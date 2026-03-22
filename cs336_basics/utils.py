import torch.nn as nn
import torch
import math
from jaxtyping import Bool, Float, Int
from collections.abc import Iterable
from cs336_basics.tokenizer import Tokenizer
import numpy as np
from pathlib import Path
import os
import shutil
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
def log_softmax(x):
  max = torch.max(x, dim=-1, keepdim=True)[0]
  return x - max - torch.log(torch.sum(torch.exp(x - max), dim=-1, keepdim=True))
def cross_entropy(inputs:torch.Tensor,targets:torch.Tensor):
  log_probs = log_softmax(inputs)
  target_log_probs = log_probs[torch.arange(inputs.shape[0], device=inputs.device), targets]
  return -target_log_probs.mean()
def perplexity(inputs,targets):
  mean_cross_entropy = cross_entropy(inputs,targets)
  return torch.exp(mean_cross_entropy)
class AdamW_Optimizer(torch.optim.Optimizer):
  def __init__(self, parameters, lr: float, weight_decay: float, betas,eps):
    param_groups = [
      {
        "params":parameters,
        "lr": lr
      }
    ]
    super(AdamW_Optimizer,self).__init__(param_groups,{})
    self.weight_decay = weight_decay
    self.beta1 = betas[0]
    self.beta2 = betas[1]
    self.eps = eps
    for group in self.param_groups:
      for param in group['params']:
        self.state[param] = {
          'm':torch.zeros_like(param.data),
          'v':torch.zeros_like(param.data),
          'step':torch.tensor(0.0,device = param.device)
        }
  def step(self):
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        grad = p.grad.data
        state = self.state[p]
        m,v,step = state['m'],state['v'], state['step']
        if not isinstance(step,torch.Tensor):
          step = torch.tensor(float(step),device= p.device)
          state['step'] = self.step
        current_lr = group.get('lr')
        m = m*self.beta1+(1-self.beta1)*grad
        v = v*self.beta2+(1-self.beta2)*(grad**2)
        step = step+1
        alpha_t = current_lr*(math.sqrt(1-self.beta2**step))/(1-self.beta1**step)
        p.data = (p.data - alpha_t * (m / (torch.sqrt(v) + self.eps))) - current_lr * self.weight_decay * p.data
        self.state[p]['m']=m
        self.state[p]['v']=v
        self.state[p]['step']=step
def lr_cosine_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
  if it < warmup_iters:
    return (it/warmup_iters)*max_learning_rate
  elif it > cosine_cycle_iters:
    return min_learning_rate
  else:
    return min_learning_rate+0.5*(1+math.cos(((it-warmup_iters)/(cosine_cycle_iters-warmup_iters))*math.pi))*(max_learning_rate-min_learning_rate)
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
  grad_sum = None
  l2norm = torch.sqrt(sum(p.grad.data.norm(2)**2 for p in parameters if p.grad is not None))
  if l2norm>max_l2_norm:
    clip_factor = max_l2_norm/(l2norm+1e-6)
    for p in parameters:
      if p.grad is not None:
        p.grad.data = p.grad.data * clip_factor
class Mmap():
  def __init__(self,corpus_path,vocab_path, merge_path, special_tokens=None,chunk_size = 1024):
    self.corpus_path = corpus_path
    self.vocab_path = vocab_path
    self.merge_path = merge_path
    self.special_tokens = special_tokens
    self.chunk_size = chunk_size
    with open(self.corpus_path) as f:
      self.corpus_size = Path(self.corpus_path).stat().st_size
  def save_as_memmap(self):
      tokenizer = Tokenizer.from_files(self.vocab_path, self.merge_path, self.special_tokens)
      buffer = []
      chunk_num = 0
      length = 0

      with open(self.corpus_path,encoding='utf-8') as f:
          encoder = tokenizer.encode_iterable(f)
          for id in encoder:
              length += 1
              buffer.append(id)
              if len(buffer) >= self.chunk_size:
                  self.save_by_chunks(buffer, self.chunk_size, chunk_num)
                  chunk_num += 1
                  buffer = []
      if len(buffer) > 0:
          self.save_by_chunks(buffer, len(buffer), chunk_num)
          buffer = []

      print(f"length of corpus in tokens:{length}")
  def save_by_chunks(self, token_ids, buffer_len, chunk_num):
    chunk_dir = Path("./data") / f"{self.corpus_size}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    self.chunk_dir = chunk_dir
    fname = chunk_dir / f"encoded_tokens_chunk_{chunk_num}.dat"
    memmap_arr = np.memmap(str(fname), dtype=np.int32, mode="w+", shape=(buffer_len,))
    memmap_arr[:] = token_ids
    memmap_arr.flush()
  def load_by_range(self, start_idx, end_idx):
    chunk_size = self.chunk_size
    start_chunk = start_idx // chunk_size
    end_chunk = end_idx // chunk_size
    idx_in_start = start_idx % chunk_size
    idx_in_end = end_idx % chunk_size

    token_ids = []
    for chunk in range(start_chunk, end_chunk + 1):
        fname = self.chunk_dir / f"encoded_tokens_chunk_{chunk}.dat"
        dtype = np.int32
        memmap_arr = np.memmap(fname, dtype=dtype, mode="r")
        if start_chunk == end_chunk:
            token_ids.extend(memmap_arr[idx_in_start:idx_in_end])
        else:
            if chunk == start_chunk:
                token_ids.extend(memmap_arr[idx_in_start:])
            elif chunk > start_chunk and chunk < end_chunk:
                token_ids.extend(memmap_arr[:])
            else:
                token_ids.extend(memmap_arr[:idx_in_end])
    return token_ids
class Batch_Random_Sampler:
  def __init__(self, mmap: Mmap = None):
    self.mmap = mmap
  def get_batch(self, bsz, seq_len, dataset_length, device=None):
    max_start_idx = dataset_length - seq_len
    start_indices = np.random.randint(0, max_start_idx, bsz)
    x = np.array([range(i,i+seq_len) for i in start_indices], dtype=np.int64)
    y = np.array([range(i+1,i+seq_len+1) for i in start_indices], dtype=np.int64)
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return (x, y)
  def get_batch_mmap(self, bsz, seq_len, dataset_length, device=None):
    max_start_idx = dataset_length - seq_len
    start_indices = np.random.randint(0, max_start_idx, bsz)
    x = np.array([self.memmap_manager.load_by_range(i, i+seq_len) for i in start_indices], dtype=np.int64)
    y = np.array([self.memmap_manager.load_by_range(i+1, i+seq_len+1) for i in start_indices], dtype=np.int64)
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return (x, y)
    
class Checkpoint_Manager:
  def save(self, model, optimizer, iteration, save_path):
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      state_model = model.state_dict()
      state_optimizer = optimizer.state_dict()
      checkpoint = {
          "model": state_model,
          "optimizer": state_optimizer,
          "iteration": iteration
      }
      torch.save(checkpoint, save_path)
  def load(self, src_path, model, optimizer=None):
    checkpoint = torch.load(src_path)
    state_model = checkpoint["model"]
    if optimizer is not None:
        print(f"optimizer is not none")
        state_optimizer = checkpoint["optimizer"]
    iteration = checkpoint["iteration"]
    model.load_state_dict(state_model)
    if optimizer is not None:
        optimizer.load_state_dict(state_optimizer)
    return iteration
