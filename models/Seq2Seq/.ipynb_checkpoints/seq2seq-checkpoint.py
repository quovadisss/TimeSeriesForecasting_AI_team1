import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.autograd import Variable
import re



ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'input_mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['input_mask'])
    return enc_kwargs, dec_kwargs, kwargs


class Encoder(nn.Module):
    # enc_kwargs : input_len, input_dim, hidden_size, rnn_num_layers ,dropout 
    def __init__(self, input_len, input_dim, hidden_size=100, rnn_num_layers=1, dropout=0.2):
        super().__init__()
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = rnn_num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout
        )
        # Learning initial hidden/cell states
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1 ,self.hidden_size), requires_grad = True)
        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1 , self.hidden_size), requires_grad = True)


    def forward(self, input_seq):
        
        batch_size = input_seq.size(0)
        h0 = self.h0.repeat(1,batch_size,1)
        c0 = self.c0.repeat(1,batch_size,1)
        
        lstm_out, hidden = self.lstm(input_seq, (h0, c0))

        return lstm_out, hidden
    
    
class DecoderCell(nn.Module):
    # dec_kwargs : input_dim, hidden_size, rnn_num_layers ,dropout
    def __init__(self, input_dim, hidden_size, rnn_num_layers=1,dropout=0.2):
        super().__init__()
        self.decoder_rnn_cell = nn.LSTMCell(
            input_size=input_dim,
            hidden_size=hidden_size
        )
        self.out = nn.Linear(hidden_size, rnn_num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input,  prev_hidden):
        rnn_hidden = self.decoder_rnn_cell(dec_input, prev_hidden)
        output = self.out(rnn_hidden[0])
        return output, (self.dropout(rnn_hidden[0]),self.dropout(rnn_hidden[1]))
    
    
class Seq2SeqEncDec(nn.Module):
    def __init__(self, input_len, label_len, pred_len ,input_dim, hidden_size,rnn_num_layers, teacher_force,  **kwargs):
        super().__init__()
        
        # enc_kwargs : input_len, input_dim, hidden_size, rnn_num_layers ,dropout 
        # dec_kwargs : input_dim, hidden_size, rnn_num_layers ,dropout
        # training: input_len label_len, pred_len, teacher_force
        
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        enc_kwargs['input_len'] = input_len
        enc_kwargs['input_dim'] = dec_kwargs['input_dim']  = input_dim
        enc_kwargs['hidden_size'] = dec_kwargs['hidden_size']  = hidden_size
        enc_kwargs['rnn_num_layers'] = dec_kwargs['rnn_num_layers']  = rnn_num_layers
        
        self.encoder = Encoder(**enc_kwargs) #.cuda()
        self.decoder_cell = DecoderCell(**dec_kwargs) #.cuda()
        
        self.input_len = input_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.output_len = label_len + pred_len
        self.teacher_forcing = teacher_force # teacher_force probability
    
    
        

    def forward(self, input_seq, **kwargs):
        
        batch_size = input_seq.size(0)
        
        
        _ , encoder_hidden = self.encoder(input_seq)        
        prev_hidden = ( encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))
        
        outputs = torch.zeros(batch_size, self.pred_len).cuda()

        
        dec_input = input_seq[:, -1, :]

        for i in range(self.pred_len):
            if (torch.rand(1) < self.teacher_forcing) and (kwargs.target_seq is not None) :
                # dec_input [batch 1 input_dim] => [batch 1 input_dim]
                dec_input = kwargs.target_seq[:, i, :].unsqueeze(1)
                
            dec_output, prev_hidden = self.decoder_cell(dec_input, prev_hidden)
            dec_input =  dec_output
            outputs[:, i] = dec_input.squeeze(1)
        
        return outputs.unsqueeze(2)
    




