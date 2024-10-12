import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim
import scipy.io
from Networks import *
from Networks.VarProjNetwork import *
from Networks.VPnetwork_times import *
from Networks.Trajectory import *
from Networks.Trajectory_time import *
from Data.data_processing import *

class FCVPNN(nn.Module):
    def __init__(self, seq_len, params_init=None):
        super().__init__()

        self.vp = VPLayer(params_init = params_init) #r
        self.traj = Trajectory_IDE(seq_len) #output (batch,1,num_samples) = (batch,1,24)

        ######################################################################

        self.fc1 = nn.Linear(24,48)
        self.fc2 = nn.Linear(48,24)
        
    def forward(self, input_x, estimation = False):
        
        [coeffs, x_hat, res, r2], params = self.vp(input_x)

        x1, _ = self.traj(coeffs, params) #(x1 is like the approximation U-shape inverted)

        ##################################################################

         
        x_fc = F.relu(self.fc1(x1))
        x_fc = self.fc2(x_fc) # keep Linear activation for handling the noise
        
        # x_fc = self.fc1(x1)
        # x_fc = self.activation(x_fc)
        # x_fc = self.fc2(x_fc)

        # # Add noise to original signal (skip connection)
        x = x1 + x_fc  # Output shape: (batch, 1, 24)

        if estimation:
          return x1

        return x, x1





class FCVPNN_time(nn.Module):
    def __init__(self, t, params_init):
        super().__init__()
        
        self.t = t

        self.vp = VPLayer_time(self.t, params_init) #r
        self.traj = Trajectory_IDE_time(self.t) #output (batch,1,num_samples) = (batch,1,24)

        ######################################################################

        self.fc1 = nn.Linear(len(self.t),64)
        self.fc2 = nn.Linear(64, len(self.t))
        
    def forward(self, input_x, estimation = False):
        
        [coeffs, x_hat, res, r2], params = self.vp(input_x)

        x1, _ = self.traj(coeffs, params) #(x1 is like the approximation U-shape inverted)

        ##################################################################

         
        x_fc = F.relu(self.fc1(x1))
        x_fc = self.fc2(x_fc) # keep Linear activation for handling the noise
        
        # x_fc = self.fc1(x1)
        # x_fc = self.activation(x_fc)
        # x_fc = self.fc2(x_fc)

        # # Add noise to original signal (skip connection)
        x = x1 + x_fc  # Output shape: (batch, 1, 24)

        if estimation:
          return x1

        return x, x1
