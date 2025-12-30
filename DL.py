import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
def batch_norm(x,beta,gamma,eps=1e-5,global_mean=None,global_var=None,momentum=0.9):
    if not torch.is_grad_enabled():
        # eval mode
        x_hat =  (x - global_mean) / torch.sqrt(global_var + eps)
    else:
        assert x.dim() in [2,4]
        if x.dim() == 2: # (B, D) fc layer
            mean = x.mean(dim = 0, keepdim=True) # (1, D)
            var = x.var(dim = 0, keepdim=True)   # (1, D)
        else: # (B, C, H, W) conv layer
            mean = x.mean(dim=(0,2,3),keepdim=True) # (1, C, 1, 1)
            var = x.var(dim=(0,2,3),keepdim=True)   # (1, C, 1, 1)
        x_hat =  (x - mean) / torch.sqrt(var + eps)
        global_mean = momentum * global_mean + (1-momentum) * mean
        global_var = momentum * global_var + (1-momentum) * var
    out = gamma * x_hat + beta
    return out,global_mean.data,global_var.data

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims=2):
        super().__init__()
        if num_dims == 2:
            shape = (1,num_features)
        else:
            shape = (1,num_features,1,1)
        self.beta = nn.Parameter(torch.zeros(shape))# Parameter是可以被训练的变量
        self.gamma = nn.Parameter(torch.ones(shape))
        self.global_mean = torch.zeros(shape)
        self.global_var = torch.ones(shape)
    def forward(self,x):
        if(self.global_mean.device != x.device):
            self.global_mean = self.global_mean.to(x.device)
            self.global_var = self.global_var.to(x.device)
        out,global_mean,global_var = batch_norm(x,self.beta,self.gamma,eps=1e-5,
                                                global_var=self.global_var,global_mean=self.global_mean,momentum=0.9)
        return out
    

if __name__ == "__main__":
    
    # 假设输入图片尺寸为 32x32
    hight = 32
    width = 32
    last_channel_output = 32
    fc_in_features = hight * width * last_channel_output
    net = nn.Sequential(
        nn.Conv2d(input=3,output=16,kernel_size = 3,stride=1,padding=1),
        BatchNorm(num_features=16,num_dims=4),
        nn.ReLU(),
        nn.Conv2d(input=16,output=last_channel_output,kernel_size = 3,stride=1,padding=1),
        BatchNorm(num_features=last_channel_output,num_dims=4),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_feature=fc_in_features,out_feature=10),
        BatchNorm(num_features=10,num_dims=2)
    )
