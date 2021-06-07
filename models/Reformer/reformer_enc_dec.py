import re
from torch import nn
from models.Reformer.reformer_pytorch import ReformerTM
from models.Reformer.autopadder import Autopadder

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





class ReformerEncDec(nn.Module):
    def __init__(self, dim, seq_len, label_len, pred_len, **kwargs):
        super().__init__()
        print('options:')
        print(kwargs)
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'return_embedding' not in enc_kwargs, 'you cannot manually set the return embeddings flag for the encoder'
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'
        print(f'dim: {dim}')
   
        enc_kwargs['seq_len'] = seq_len
        dec_kwargs['seq_len'] = label_len  + pred_len

        
        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        

        dec_kwargs['causal'] = True

        enc_kwargs.setdefault('bucket_size', 64)
        dec_kwargs.setdefault('bucket_size', enc_kwargs['bucket_size'] * 2)


        print('encoder option:')
        print(enc_kwargs)
        print('decoder option:')
        print(dec_kwargs)
        
        self.enc = Autopadder(ReformerTM(**enc_kwargs))
        self.dec = Autopadder(ReformerTM(**dec_kwargs))

    
    def generate(self, seq_in, label_len, pred_len , **kwargs):
        label_seq = seq_in[:, label_len :, :]
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec.generate(label_seq, pred_len, keys = enc_keys, **{**dec_kwargs, **kwargs})
    
    def forward(self, seq_in, seq_out, **kwargs):
        
        # seq_in =  [batch_size, input_len , seq_dim]
        # seq_out = [batch_size, label_len + pred_len, seq_dim] : target
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        
        enc_keys = self.enc(seq_in, **enc_kwargs)
        
        return self.dec(seq_out, keys = enc_keys, **dec_kwargs), __
