import torch.nn as nn
import torch
import math
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