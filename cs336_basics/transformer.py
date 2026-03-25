import torch
import torch.nn as nn
from cs336_basics.utils import Multihead_Self_Attention, RMSNorm, SwiGLU, Embedding, Linear
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_length,device =None,dtype = None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model,device=device)
        self.rmsnorm2 = RMSNorm(d_model,device=device)
        self.MHA = Multihead_Self_Attention(d_model,num_heads,max_seq_length,theta,device)
        self.FFN = SwiGLU(d_model,d_ff,device=device,dtype = dtype)
    def forward(self,x:torch.Tensor):
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len, device=x.device)

        x = x + self.MHA(self.rmsnorm1(x),token_positions)
        x = x + self.FFN(self.rmsnorm2(x))
        return x
    def change_weights(self, weights: dict):
        MHA_Q_proj_weights = weights['attn.q_proj.weight']
        MHA_K_proj_weights = weights['attn.k_proj.weight']
        MHA_V_proj_weights = weights['attn.v_proj.weight']
        MHA_O_proj_weights = weights['attn.output_proj.weight']
        rmsnorm1_weights = weights['ln1.weight']
        rmsnorm2_weights = weights['ln2.weight']
        FFN_W1 = weights['ffn.w1.weight']
        FFN_W2 = weights['ffn.w2.weight']
        FFN_W3 = weights['ffn.w3.weight']
        
        device = next(self.parameters()).device
        
        self.MHA.change_weights(
            MHA_Q_proj_weights.to(device),
            MHA_K_proj_weights.to(device),
            MHA_V_proj_weights.to(device),
            MHA_O_proj_weights.to(device)
        )
        self.rmsnorm1.change_weights(rmsnorm1_weights.to(device))
        self.rmsnorm2.change_weights(rmsnorm2_weights.to(device))
        self.FFN.change_weights(
            FFN_W1.to(device),
            FFN_W2.to(device),
            FFN_W3.to(device)
        )
class TransformerLM(nn.Module):
    def __init__(self,
                vocab_size: int,
                max_seq_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device = None,
                dtype = None):
        super().__init__()
        self.num_vocab = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads =  num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.embedding = Embedding(vocab_size,d_model,device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope_theta,
                max_seq_length,
                device,
                dtype
            )
            for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model,device=device,dtype = dtype)
        self.outputlayer = Linear(d_model,vocab_size,device=device,dtype=dtype)
    def forward(self,token_ids:torch.Tensor):
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x)
        result = self.outputlayer(self.rmsnorm(x))
        return result
    def change_weights(self,weights:dict):
        
        embedding_weights = weights['token_embeddings.weight']
        rms_weights = weights['ln_final.weight']
        outputlayer_weights = weights['lm_head.weight']
        self.embedding.change_weights(embedding_weights)
        self.rmsnorm.change_weights(rms_weights)
        self.outputlayer.change_weights(outputlayer_weights)
        for layer,block in enumerate(self.transformer_blocks):
            layer_weights = dict()
            layer_weights['attn.q_proj.weight'] = weights[f'layers.{layer}.attn.q_proj.weight']
            layer_weights['attn.k_proj.weight'] = weights[f'layers.{layer}.attn.k_proj.weight']
            layer_weights['attn.v_proj.weight'] = weights[f'layers.{layer}.attn.v_proj.weight']
            layer_weights['attn.output_proj.weight'] = weights[f'layers.{layer}.attn.output_proj.weight']
            layer_weights['ln1.weight']= weights[f'layers.{layer}.ln1.weight']
            layer_weights['ln2.weight']= weights[f'layers.{layer}.ln2.weight']
            layer_weights['ffn.w1.weight']= weights[f'layers.{layer}.ffn.w1.weight']
            layer_weights['ffn.w2.weight']= weights[f'layers.{layer}.ffn.w2.weight']
            layer_weights['ffn.w3.weight']= weights[f'layers.{layer}.ffn.w3.weight']    
            block.change_weights(layer_weights)
