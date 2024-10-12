import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.optim as optim
import scipy.io

def cusum1(x, h):
    y = torch.zeros_like(x)

    y[...,0] = x[...,0]
    for i in range(1, x.shape[-1]):
      y[...,i] = y[...,i-1] + h * x[...,i]

    return y
  
  
def cusum(x,h):
    y = torch.zeros_like(x) # (512,1,100)

    y[...,0] = 0
    for t in range(1, x.shape[-1]):
        y[...,t] = y[...,t-1] + 0.5 * h * x[...,t-1] + 0.5 * h * x[...,t]

    return y

class VPFun(Function):
  @staticmethod
  def forward(ctx, input, params):
    x = input[:,:,1:] #(512,1,100)
    phi, dphi = VPFun.bernouli(input, params) #(512,4,100) , #(512,2,4,100)
    phip = torch.linalg.pinv(phi) #(512,)
    coeffs = x @ phip #(512,1,100) @ (512,100,4) = (512,1,4)
    x_hat = coeffs @ phi #(512,1,4)@(512,4,100)
    res = x - x_hat #(512,1)
    r2 = (res ** 2).sum(dim = -1)
    ctx.save_for_backward(phi, phip, dphi, coeffs, res)

    return coeffs, x_hat, res, r2


  @staticmethod
  def backward(ctx, d_coeff, d_x_hat,
                 d_res, d_r2):
        '''
        Computes the backpropagation gradients.

        Input:
            d_coeffs: torch.Tensor  Backpropagated gradient of coeffs.
                                    Size: (batch,*,num_coeffs)
            d_x_hat: torch.Tensor   Backpropagated gradient of x_hat.
                                    Size: (batch,*,num_samples)
            d_res: torch.Tensor     Backpropagated gradient of res.
                                    Size: (batch,*,num_samples)
            d_r2: torch.Tensor      Backpropagated gradient of r2.
                                    Size: (batch,*)
        Output:
            dx: torch.Tensor        Gradient of input x.
                                    Size: (batch,*,num_samples)
            d_params: torch.Tensor  Gradient of params.
                                    Size: (num_params)
            None                    [Argument if not differentiable.]
        '''
        phi, phip, dphi, coeffs, res = ctx.saved_tensors
        # Intermediate Jacobians:
        #   Jac1 = dPhi coeff
        #   Jac2 = Phi^+^T dPhi^T res
        #   Jac3 = dPhi^T Phi^+^T c
        jac1 = coeffs @ dphi.permute(1, 0, 2, 3) #(512,1,4) @ (2,512,4,100)   # (num_params,batch,*,num_samples)
        jac2 = res @ dphi.permute(1,0,3,2) @ phip.permute(0,2,1) #(512,1,100)@(2,512,100,4)@
            # (num_params,batch,*,num_samples)
        jac3 = coeffs @ phip.permute(0,2,1) @ dphi.permute(1,0,3,2)  # (num_params,batch,*,num_coeffs)
            # (512,1,4) @ (512,4,100) @ (2,512,100,4) = (2,512,1,4)

        # Jacobians
        jac_coeff = jac3 + (-jac1 + jac2 - jac3 @ phi) @ phip   # (num_params,batch,*,num_coeffs)
        jac_x_hat = jac1 - jac1 @ phip @ phi + jac2             # (num_params,batch,*,num_samples)
        jac_res = -jac_x_hat                                    # (num_params,batch,*,num_samples)
        jac_r2 = -2 * (jac1 * res).sum(dim=-1)                  # (num_params,batch,*)
        # gradients
        dx = d_coeff @ phip.permute(0,2,1) + \
             d_x_hat @ phip @ phi + \
             d_res - d_res @ phip @ phi + \
             2 * d_r2.unsqueeze(-1) * res

        d_params = (jac_coeff * d_coeff).flatten(1).sum(dim=1) + \
                   (jac_x_hat * d_x_hat).flatten(1).sum(dim=1) + \
                   (jac_res * d_res).flatten(1).sum(dim=1) + \
                   (jac_r2 * d_r2).flatten(1).sum(dim=1)


        return dx, d_params, None

  @staticmethod
  def bernouli(x, params):
    #params = [r, ny] which is trainable parameter

    r = params[0]
    ny = params[1]
    y_cusum = cusum(x, 1) #(batch,num_channels, num_samples)

    Theta, dTheta, dTheta1, dTheta2 = [],[],[],[]

    for y in y_cusum.squeeze(1):

      theta = torch.stack([
                torch.tensor([ny + y[i], ((ny + y[i]) ** r)])
                for i in range(1, y.squeeze().shape[0]) #(1,100)
            ])
      Theta.append(theta)
       
      #dtheta1 is derivative matrix correspond to variable r
      dtheta1 = torch.stack([
                torch.tensor([0, ((ny + y[i]) ** r) * torch.log(ny + y[i])])
                for i in range(1, y.squeeze().shape[0]) #(1,100)
            ])
      

      dtheta2 = torch.stack([
         torch.tensor([1, r * (ny + y[i])**(r-1) ])
          for i in range(1, y.squeeze().shape[0])
          ])

      dTheta1.append(dtheta1)
      dTheta2.append(dtheta2)

    dTheta1 = torch.stack(dTheta1)
    dTheta2 = torch.stack(dTheta2)

    dTheta = torch.stack([dTheta1, dTheta2]).permute(1,0,3,2)

    Theta = torch.stack(Theta, dim=0).transpose(1,2)
    return Theta, dTheta
  
  




class VPLayer(nn.Module):
  def __init__(self,params_init):
    super().__init__()

    self.params = nn.Parameter(params_init.clone().detach().requires_grad_(True))
    # [r, ny]

  def forward(self, x):
    return VPFun.apply(x, self.params), self.params


class VPLoss(nn.Module):
  def __init__(self,criterion, vp_penalty):
    super().__init__()
    if vp_penalty is None:
      vp_penalty = 0.5
    self.criterion = criterion
    self.vp_penalty = vp_penalty

  def forward(self, outputs, target):
    y, x1 = outputs

    return self.criterion(y, target) + self.vp_penalty * self.criterion(x1, target)

  def extra_repr(self) -> str:
    return f'(vp_penalty): {self.vp_penalty}'








