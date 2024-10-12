import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim
import scipy.io

class Trajectory_ODE_time(nn.Module):
  def __init__(self, seq_len, t):
    
    super().__init__()
    #those learnable parameters force model to better fit at Trajectory Layer
    #seq_len is the like the nụmbers
    #t is time steps 
    
    self.seq_len = seq_len
    self.r_ex = nn.Parameter(torch.tensor(1.0))
    self.eta_y_ex = nn.Parameter(torch.tensor(1.0))
    self.b1 = nn.Parameter(torch.tensor(1.0))
    self.b2 = nn.Parameter(torch.tensor(1.0))
    self.plus = nn.Parameter(torch.tensor(0.1))
    self.t = t

  def forward(self, input, params):
      
    #input = (batch, num_channels, time_steps)

    batch = input.shape[0] #(512)
    r = params[0]

    eta_y = params[1] + self.plus
    
    a1 = input[:, :, 0].float() #(512,1)
    a2 = input[:, :, 1].float() #(512,1)
    

    term1 = ((eta_y * self.eta_y_ex) ** (1 - r * self.r_ex ) + (a2 * self.b2) / (a1 * self.b1) ) \
        * torch.exp(self.b1 * a1 * (1 - r * self.r_ex) * self.t) - (a2 * self.b2) / (a1 * self.b1)

    y_esti = term1 ** (1 / (1 - r * self.r_ex))

    x = self.b1 * a1 * y_esti + self.b2 * a2 * y_esti ** (r * self.r_ex)

    return x.unsqueeze(1), y_esti.unsqueeze(1)



class Trajectory_IDE_time(nn.Module):
  def __init__(self, t):
    super().__init__()
    
    self.r_ex = nn.Parameter(torch.tensor(1.0))
    self.eta_y_ex = nn.Parameter(torch.tensor(1.0))   
    self.b1 = nn.Parameter(torch.tensor(1.0))
    self.b2 = nn.Parameter(torch.tensor(1.0))
    self.plus = nn.Parameter(torch.tensor(0.1))
    self.t = t

  def forward(self, input, params):
      
    #input = (batch, num_channels, time_steps)
    # params = [r, ny] learnable parameters
    
    r = params[0]
    eta_y = params[1] + self.plus
    a1 = input[:, :, 0].float() #(512,1)
    a2 = input[:, :, 1].float() #(512,1)

        
    term1 = (self.b1 * a1 * (self.eta_y_ex * eta_y) ** (1 - r * self.r_ex) + \
      self.b2 * a2) * torch.exp(self.b1 * a1 * (1 - r * self.r_ex ) * self.t)
    
    term2 = (((self.eta_y_ex * eta_y) ** ( 1 - r * self.r_ex) + self.b2 * a2 /(self.b1 * a1)) * \
             torch.exp(a1 * (1 - r * self.r_ex) * self.t) - self.b2 * a2/(self.b1 * a1)) ** ((self.r_ex * r) / (1 - self.r_ex * r))

    x = term1 * term2

    return x.unsqueeze(1), None

