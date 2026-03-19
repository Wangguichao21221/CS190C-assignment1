import torch
import torch.nn as nn
from cs336_basics.utils import Multihead_Self_Attention, RMSNorm, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_length,device =None,dtype = None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model,device=device)
        self.rmsnorm2 = RMSNorm(d_model,device=device)
        self.MHA = Multihead_Self_Attention(d_model,num_heads,max_seq_length,theta,device)
        self.FFN = SwiGLU(d_model,d_ff,device=device,dtype = dtype)
    def forward(self,x:torch.Tensor):
        seq_len = x.shape[1]
        token_positions = torch.arange(seq_len).unsqueeze(0).expand(x.shape[0], -1).to(x.device)

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
    def __init__(self,vocab_size: int,
                    context_length: int,
                    d_model: int,
                    num_layers: int,
                    num_heads: int,
                    d_ff: int,
                    rope_theta: float,
                    weights: dict[str, torch.Tensor],
                    device = None,
                    dtype = None):
        super().__init__()
        # TODO
    def forward(self,in_indices:torch.Tensor):
        pass