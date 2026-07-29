import torch
import torch.nn as nn

class LayerNormal(nn.Module):
    def __init__(self, feature_size, epsilon=1e-6):
        super().__init__()
        self.feature_size = feature_size
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.full(size=(self.feature_size, 1), fill_value=1.0))
        self.beta = nn.Parameter(torch.full(size=(self.feature_size, 1), fill_value=0.0))
    '''
    x (batch_size,feature_size)
    '''
    def forward(self, x):
        mean_value = torch.unsqueeze(torch.mean(x, dim=-1), -1)  # (batch_size,1)
        sigma = torch.sqrt(torch.mean(torch.pow(x - mean_value, 2), dim=-1) + self.epsilon)  # (batch_size,)
        sigma = torch.unsqueeze(sigma, -1)  # (batch_size,1)
        gamma_expand = torch.transpose(self.gamma, 0, 1).expand(x.shape[0], -1)  # (batch_size,feature_size)
        beta_expand = torch.transpose(self.beta, 0, 1).expand(x.shape[0], -1)  # (batch_size,feature_size)
        return gamma_expand * (x - mean_value) / sigma + beta_expand  # (batch_size,feature_size)


class LayerNormalV2(nn.Module):
    def __init__(self, feature_size, epsilon=1e-6):
        super().__init__()
        self.feature_size = feature_size
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.full(size=(self.feature_size,), fill_value=1.0))
        self.beta = nn.Parameter(torch.full(size=(self.feature_size,), fill_value=0.0))

    '''
    x (batch_size,feature_size)
    '''

    def forward(self, x):
        mean_value = torch.unsqueeze(torch.mean(x, dim=-1), -1)  # (batch_size,1)
        sigma = torch.sqrt(torch.mean(torch.pow(x - mean_value, 2), dim=-1) + self.epsilon)  # (batch_size,)
        sigma = torch.unsqueeze(sigma, -1)  # (batch_size,1)
        return self.gamma * (x - mean_value) / sigma + self.beta  # (batch_size,feature_size)

class RMSNormal(nn.Module):
    def __init__(self,feature_size,epsilon=1e-6):
        super().__init__()
        self.feature_size=feature_size
        self.epsilon=epsilon
        self.gamma=nn.Parameter(torch.full(size=(self.feature_size,),fill_value=1.0))
    '''
    x (batch_size,feature_size)
    '''
    def forward(self,x):
        sigma=torch.sqrt(torch.mean(torch.pow(x,2),dim=-1)+self.epsilon)#(batch_size,)
        sigma=torch.unsqueeze(sigma,-1) #(batch_size,1)
        return self.gamma*(x)/sigma #(batch_size,feature_size)

