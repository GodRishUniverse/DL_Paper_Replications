from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

import math


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# THERE IS A BUG IN THE CODE - 


@dataclass
class ModelConfig:
    d_model: int = 512 
    dff: int = 2048
    device: str = 'cuda'
    num_heads: int = 8
    dropout: float = 0.1

    src_vocab_size: int = 100
    tgt_vocab_size: int = 100
    pad_index: int = 0

    num_encoder_layers: int = 6 # num_encoder_layers
    num_decoder_layers: int = 6 # num_decoder_layers
    max_seq_len: int =1000

class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.dff, device=config.device)
        self.linear2 = nn.Linear(config.dff, config.d_model, device=config.device)
        self.gelu = nn.GELU()
        self.device = config.device

    def forward(self, x):
        x = x.to(self.device)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x 


class MultiHeadAttention(nn.Module):
    def __init__(self,config: ModelConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.device = config.device
        # Hyperparameters passed from config
        d_model = config.d_model
        num_heads = config.num_heads
        dropout = config.dropout

        self.d_k = d_model // num_heads # using d_model and number of heads for d_k calculation
        self.num_heads = num_heads
        
        # Combined linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model,device=self.device)
        self.k_proj = nn.Linear(d_model, d_model, device=self.device)
        self.v_proj = nn.Linear(d_model, d_model, device=self.device)
        self.out_proj = nn.Linear(d_model, d_model, device= self.device)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention (can use internal implementation as well)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.to(self.device)
            # print("In MHA cust: ", mask.shape, scores.shape, "Q: ", q.shape, "K: ", k.shape, "V: ", v.shape)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_proj(context)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        d_model = config.d_model
        dff = config.dff
        device = config.device

        self.mha = MultiHeadAttention(config=config)
        self.layernorm1 = nn.LayerNorm(d_model, device=device)

        self.ffn = FFN(config=config)
        self.layernorm2 = nn.LayerNorm(d_model, device =device)

        self.device = device
    def forward(self, input_emb, mask=None):
        input_emb = input_emb.to(self.device)
        x = self.mha(input_emb, input_emb, input_emb, mask)
        x = self.layernorm1(x + input_emb)
        ffn_out = self.ffn(x).to(self.device)
        x = self.layernorm2(x + ffn_out)
        return x


class Decoder(nn.Module):
    def __init__(self,config: ModelConfig):
        super().__init__()

        d_model = config.d_model
        dropout = config.dropout
        device = config.device

        self.self_attn = MultiHeadAttention(config=config)
        self.cross_attn = MultiHeadAttention(config=config)

        self.ffn = FFN(config=config)

        self.layernorm1 = nn.LayerNorm(d_model, device=device)
        self.layernorm2 = nn.LayerNorm(d_model, device=device)
        self.layernorm3 = nn.LayerNorm(d_model, device=device)

        self.device = device

        self.dropout = nn.Dropout(dropout)
    def forward(self, x, residual, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        
        # Cross-attention
        # attn_output = self.cross_attn(x, residual, residual, src_mask) # causing error - mask size is different and doesnt match q,k,v size
        attn_output = self.cross_attn(x, residual, residual, None) # mask set to None to avoid error for the time being
        x = self.layernorm2(x + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(x)
        return self.layernorm3(x + self.dropout(ffn_output))


class Transformer(nn.Module):
    def __init__(self,  config: ModelConfig):
        super().__init__()

        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.dropout = config.dropout
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.max_seq_len = config.max_seq_len
        self.device = config.device

        self.pad_index = config.pad_index
        self.d_model = config.d_model


        # Embeddings
        self.src_embed = nn.Embedding(self.src_vocab_size, self.d_model, padding_idx=self.pad_index, device=self.device)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, self.d_model, padding_idx=self.pad_index, device=self.device)

        # Positional encoding
        self.pos_encoding = self.create_positional_encoding(self.max_seq_len, self.d_model)  # Max sequence length of 1000
        # rather than calculating this every forward pass we calculate this once and store it

        
        self.encoder = nn.ModuleList([Encoder(config=config) for _ in range(self.num_encoder_layers)])
        self.decoder = nn.ModuleList([Decoder(config=config) for _ in range(self.num_decoder_layers)])
        self.final = nn.Sequential(
            nn.Linear(self.d_model, self.tgt_vocab_size, device=self.device),
        )

        self.dropout = nn.Dropout(self.dropout) 

        self.init_parameters()

    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)

    def create_padding_mask(self, src):
        return (src != self.pad_index).unsqueeze(1).unsqueeze(2)

    def create_causal_mask(self, tgt):
        seq_len = tgt.size(1)
        return torch.tril(torch.ones(seq_len, seq_len)).bool()

    def init_parameters(self):
        # Utilizaing Xavier Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_emb, output_emb):

        output_emb = output_emb.long()

        output_emb = torch.clamp(output_emb, 0, self.tgt_vocab_size - 1) # trying to fix Assertion `srcIndex < srcSelectDimSize` failed - suggests that output_emb contains values that are >= tgt_vocab_size
        #(Used Claude for debugging)

        src_mask = self.create_padding_mask(input_emb)
        tgt_mask = self.create_causal_mask(output_emb)
        
        # Embedding and positional encoding for source
        src_embedded = self.src_embed(input_emb) * math.sqrt(self.d_model)
        src_embedded = src_embedded.to(self.device)

        temp  = (self.pos_encoding[:, :src_embedded.size(1)]).to(self.device)
        src_embedded = src_embedded + temp
        enc_output = self.dropout(src_embedded)

        for enc in self.encoder:
            enc_output = enc(enc_output, src_mask)
        
        # Embedding and positional encoding for target
        tgt_embedded = self.tgt_embed(output_emb) * math.sqrt(self.d_model)
        temp = (self.pos_encoding[:, :tgt_embedded.size(1)]).to(self.device)
        tgt_embedded = tgt_embedded + temp
        dec_output = self.dropout(tgt_embedded)

        # Decoder
        for dec in self.decoder:
            dec_output = dec(x = dec_output, residual = enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        
        return self.final(dec_output)
    
    def generate(self, inputs, max_len, sos_token, eos_token):
        self.eval()
        with torch.inference_mode():
            # Encode source sequence
            src_mask = self.create_padding_mask(inputs)
            
            src_embedded = self.src_embed(inputs) * math.sqrt(self.d_model)
            src_embedded = src_embedded + self.pos_encoding[:, :src_embedded.size(1)].to(self.device)
            enc_output = self.dropout(src_embedded)
            
            for enc in self.encoder:
                enc_output = enc(enc_output, src_mask)
            
            # Initialize target sequence with SOS token
            target = torch.full((inputs.size(0), 1), sos_token, dtype=torch.long, device=self.device)
            
            # Generate tokens one by one
            for _ in range(max_len - 1):
                tgt_mask = self.create_causal_mask(target)
                
                tgt_embedded = self.tgt_embed(target) * math.sqrt(self.d_model)
                tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt_embedded.size(1)].to(self.device)
                dec_output = self.dropout(tgt_embedded)
                
                for dec in self.decoder:
                    dec_output = dec(x = dec_output, residual = enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
                
                output = self.final_layer(dec_output)
                next_token = output[:, -1].argmax(dim=-1).unsqueeze(1)
                target = torch.cat([target, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == eos_token:
                    break
            
            return target
