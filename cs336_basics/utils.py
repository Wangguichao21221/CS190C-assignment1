import torch.nn as nn
import torch
import math
from jaxtyping import Bool, Float, Int
class Linear(nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    """
    Construct a linear transformation module. This function should accept the following parameters
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None=None Device to store the parameters on
    dtype: torch.dtype | None=None Data type of the parameters
    """
    super().__init__()
    W = torch.empty((out_features, in_features) ,device=device,dtype=dtype)
    std = math.sqrt(2.0/(in_features + out_features))
    nn.init.trunc_normal_(W,mean=0.0,std= std,a=-3*std, b =3*std)
    self.W = nn.Parameter(W, requires_grad=True)
  def change_weights(self,weights):
    self.W = nn.Parameter(weights ,requires_grad=True)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.W.transpose(-2,-1)
    
class Embedding(nn.Module):
  def __init__(self,num_vocab, d_model,device=None,dtype=None):
    super().__init__()
    MatrixEmbeddings = torch.empty((num_vocab,d_model),device=device, dtype= dtype)
    nn.init.trunc_normal_(MatrixEmbeddings, mean=0, std=1, a=-3, b=3)
    self.ME = nn.Parameter(MatrixEmbeddings ,requires_grad=True)

  def change_weights(self,weights):
    self.ME = nn.Parameter(weights ,requires_grad=True)
  def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
    return self.ME[token_ids]
class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5,device=None, dtype=None):
    """
    Construct the RMSNorm module. This function should accept the following parameters:
    d_model: int Hidden dimension of the model
    eps: float = 1e-5 Epsilon value for numerical stability
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    """
    super().__init__()
    self.eps = eps
    g = torch.ones(d_model,device=device,dtype= dtype)
    self.g = nn.Parameter(g,requires_grad=True)
  def change_weights(self,weights):
    self.g = nn.Parameter(weights,requires_grad=True)
  def forward(self, x: torch.Tensor)-> torch.Tensor :
    """
    Process an input tensor of shape(batch_size, sequence_length, d_model) and return a tensor of the same shape
    """
    in_dtype = x.dtype
    x = x.to(torch.float32)
    x2 = x**2
    mean_x_squared = torch.mean(x2, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_x_squared + self.eps)
    result = x / rms * self.g

    return result.to(in_dtype)
class SwiGLU(nn.Module):
  def __init__(self, d_model,dff, device=None, dtype=None):
    super().__init__()
    self.W1 = Linear(d_model, dff)  
    self.W3 = Linear(d_model, dff)  
    self.W2 = Linear(dff, d_model)  
  def forward(self,x):
    gate = SiLU(self.W1(x))  
    value = self.W3(x)                   
    glu_out = gate * value              
    ffn_out = self.W2(glu_out)          
    return ffn_out
  def change_weights(self,W1,W2,W3):
    self.W1.change_weights(W1)
    self.W2.change_weights(W2)
    self.W3.change_weights(W3)

def SiLU(x: torch.Tensor):
  return x*torch.sigmoid(x)
def softmax(x: torch.Tensor, dim) -> torch.Tensor:
  maxexp = torch.max(x,dim=dim, keepdim=True)[0]
  x_shifted = x-maxexp
  softmax = torch.exp(x_shifted)/torch.sum(torch.exp(x_shifted),dim=dim,keepdim= True)
  return softmax
class RoPE(nn.Module):
  def __init__(self,theta: float,d_k:int,max_seq_len:int,device=None):
    """
    the RoPE module and create buffers if needed.
    theta:float Θ value for the RoPE
    d_k: int dimension of query and key vectors
    max_seq_len: int Maximum sequence length that will be inputted
    device: torch.device | None= None Device to store the buffer on
    """
    super().__init__()
    self.theta = theta
    self.d_k = d_k
    self.max_seq_len = max_seq_len
    self.device = device
    self.d_half = d_k//2
    positions = torch.arange(self.max_seq_len,device=self.device).unsqueeze(1)
    dims = torch.arange(self.d_half,device=self.device).unsqueeze(0)
    theta_table = positions/self.theta**(2.0*dims/self.d_k)
    self.register_buffer('cos_table',torch.cos(theta_table.unsqueeze(0)))
    self.register_buffer('sin_table',torch.sin(theta_table.unsqueeze(0)))
  def forward(self, x:torch.Tensor,token_positions:torch.Tensor):
    """
    Process an input tensor of shape (...,seq_len,d_k) and return a tensor of the same shape.
    """
    x_splited = x.reshape(*x.shape[:-1], self.d_k//2, 2)
    cos_chunk = self.cos_table[:, token_positions, :]
    sin_chunk = self.sin_table[:, token_positions, :]
    even_transform = torch.stack([cos_chunk,  -sin_chunk], dim=-1)
    odd_transform  = torch.stack([sin_chunk, cos_chunk], dim=-1)
    x_rotated_even = torch.sum(x_splited * even_transform, dim=-1)
    x_rotated_odd = torch.sum(x_splited * odd_transform, dim=-1)
    stacked_x = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = stacked_x.reshape(*stacked_x.shape[:-2], self.d_k)
    return x_rotated
def scaled_dot_product_attention(
    Q,
    K,
    V,
    mask=None,):
  """
  Q:(batch, heads, seq_len_q, d_k)
  K:(batch, heads, seq_len_k, d_k)
  V:(batch, heads, seq_len_k, d_v)
  mask: boolean tensor. True=allow, False=block
        accepted shapes:
          (seq_q, seq_k) -> broadcast to (batch, heads, seq_q, seq_k)
  """
  d_k = Q.shape[-1]
  QKT = Q @ K.transpose(-1, -2)
  attention_scores = QKT / math.sqrt(d_k)
  
  if mask is not None:
      mask = mask.to(torch.bool)
      # expand (seq_q, seq_k) to (1, 1, seq_q, seq_k)
      if mask.dim() == 2:
          mask = mask.unsqueeze(0).unsqueeze(0)
      # expand to match attention_scores shape: (batch, heads, seq_q, seq_k)
      mask = mask.expand(attention_scores.shape)
      mask = mask.to(device=Q.device)
      # where mask is True keep 0, where False set -inf
      additive = torch.where(
          mask,
          torch.tensor(0.0, device=Q.device),
          torch.tensor(float('-inf'), device=Q.device)
      )
      attention_scores = attention_scores + additive

  attention_weights = softmax(attention_scores, dim=-1)
  output = attention_weights @ V
  
  return output
class Multihead_Self_Attention(nn.Module):
  def __init__(self,d_model,num_heads,max_seq_length:int=None,theta:int=None,device=None):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model//num_heads
    self.d_v = d_model//num_heads
    self.max_seq_length = max_seq_length
    self.theta = theta
    self.token_positons = None
    self.Q_proj = Linear(d_model,self.d_k*self.num_heads,device=device)
    self.K_proj = Linear(d_model,self.d_k*self.num_heads,device=device)
    self.V_proj = Linear(d_model,self.d_v*self.num_heads,device=device)
    self.O_proj = Linear(self.d_v*self.num_heads,d_model,device=device)
    self.attention_func = scaled_dot_product_attention
    if max_seq_length is not None and self.theta is not None:
      self.rope = RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_length,device=device)
    else:
      self.rope = None
  def forward(self,x:torch.Tensor,token_positons:torch.Tensor = None):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    Q= self.Q_proj(x)
    Q=Q.reshape(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
    K= self.K_proj(x)
    K=K.reshape(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
    V= self.V_proj(x)
    V=V.reshape(batch_size,seq_len,self.num_heads,self.d_v).transpose(1,2)
    if self.rope is not None and token_positons is not None:
      self.token_positons = token_positons
      Q= self.rope(Q,self.token_positons)
      K= self.rope(K,self.token_positons)
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
    attention_output = self.attention_func(Q,K,V,mask=mask)
    attention_output = attention_output.transpose(1,2).contiguous().reshape(batch_size,seq_len,self.d_v*self.num_heads)
    output = self.O_proj(attention_output)
    return output
  def change_weights(self,Q_proj_weights,K_proj_weights,V_proj_weights,O_proj_weights):
    self.Q_proj.change_weights(Q_proj_weights)
    self.K_proj.change_weights(K_proj_weights)
    self.V_proj.change_weights(V_proj_weights)
    self.O_proj.change_weights(O_proj_weights)