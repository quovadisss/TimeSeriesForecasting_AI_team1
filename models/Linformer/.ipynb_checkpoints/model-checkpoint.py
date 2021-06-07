import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.Linformer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.Linformer.decoder import Decoder, DecoderLayer
from models.Linformer.attn import FullAttention, ProbAttention, LinformerAttentionLayer
from models.Linformer.embed import DataEmbedding


class Linformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True, enc_k = None, dec_k = None,  headwise_sharing = False, key_value_sharing = False  ):
        super(Linformer, self).__init__()
        
        
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        padding = 1
        kernel_size = 3
        stride = 2
        
        
        l =  seq_len
        seq_len_list = []
        for _ in range(e_layers):
            seq_len_list.append(l)
            l = ( l + 2*padding - 1*( kernel_size -1)-1)//stride + 1 if distil else seq_len
            #l = ( seq_len + 2*padding - kernel_size) //stride +1  if distil else seq_len
            
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LinformerAttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads,  enc_k,  middle_len , mix=False, headwise_sharing = headwise_sharing, key_value_sharing = key_value_sharing),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l , middle_len in zip( range(e_layers), seq_len_list)
            ],
            [
                ConvLayer(
                    d_model
                ) for l  in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    LinformerAttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, dec_k, label_len + out_len, mix=mix, headwise_sharing = headwise_sharing, key_value_sharing = key_value_sharing),
                    LinformerAttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, dec_k, seq_len_list[-1],mix=False, headwise_sharing = headwise_sharing, key_value_sharing = key_value_sharing),
                    d_model,
                    d_ff, 
                    dropout=dropout,
                    activation=activation
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

