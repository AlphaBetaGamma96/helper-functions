import torch
#check of https://www.tensorflow.org/api_docs/python/tf/math/cumsum
def exclusive_cumsum(x: torch.Tensor, dim: int):
  res = x.cumsum(dim=dim).roll(1) #roll(1) and res[...,0]=0. make it exclusive btw
  res[...,0]=0.
  return res
  
def reverse_exclusive_cumsum(x: torch.Tensor, dim: int):
  res = x.flip(dim).cumsum(dim=dim)
  res[...,-1]=0.
  res=res.roll(1).flip(-1) #was roll(1,0) although no change in behaviour...
  return res

#modified from https://github.com/deepmind/ferminet/blob/tf/ferminet/networks.py#L143-L192
def get_log_sigma(sigma: torch.Tensor):
  return torch.log(sigma)
  
def get_log_gamma(log_sigma: torch.Tensor):
  lower = exclusive_cumsum(log_sigma, dim=-1)
  upper = reverse_exclusive_cumsum(log_sigma, dim=-1)
  log_gamma = lower + upper

  return log_gamma

def get_log_rho(log_sigma: torch.Tensor):
  lower_cumsum = exclusive_cumsum(log_sigma, dim=-1)
  upper_cumsum = reverse_exclusive_cumsum(log_sigma, dim=-1)
  
  size = len(log_sigma.shape)
  v = lower_cumsum.unsqueeze(size) + upper_cumsum.unsqueeze(size-1)
  
  s_mat = torch.transpose(torch.tile(log_sigma.unsqueeze(size), [1] * size + [log_sigma.shape[-1]]), -2, -1)
  
  triu_s_mat = torch.triu(s_mat, diagonal=1)
  
  z = exclusive_cumsum(triu_s_mat, -1)
  
  r = torch.triu(torch.exp(z + v), diagonal=1)
  log_rho = torch.log(r + r.transpose(-2,-1))
  
  log_gamma = (lower_cumsum + upper_cumsum)
  log_rho = torch.diagonal_scatter(log_rho, log_gamma, offset=0, dim1=-2, dim2=-1)
  return log_rho

#S = torch.arange(1,5,1)
S = torch.Tensor([[1,2,3,4],
                  [1,2,3,4]])
log_S = get_log_sigma(S)
log_gamma = get_log_gamma(log_S)
log_rho = get_log_rho(log_S)
print("log_sigma: \n",log_S)
print("log_gamma: \n",log_gamma)
print("log_rho: \n",log_rho)

"""
exact log rho:
tensor([[[3.1781, 2.4849, 2.0794, 1.7918],
         [2.4849, 2.4849, 1.3863, 1.0986],
         [2.0794, 1.3863, 2.0794, 0.6931],
         [1.7918, 1.0986, 0.6931, 1.7918]]])
"""
