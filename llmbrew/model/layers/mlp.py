import torch
import torch.nn as nn
import torch.nn.functional as F
from llmbrew.model.layers import RMSNormal
from llmbrew.model.layers import init_embedding
# defined a general MLP block
class MLPBlock(nn.Module):
    def __init__(self,
                 *,
                 input_dim,
                 hidden_layers=(128, 64, 32),
                 norm=False,
                 dropout:float=0.0
                 ):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_layers=hidden_layers
        self.norm=norm
        self.dropout=dropout
        module_list=[]
        pre_layer=self.input_dim
        for hidden_layer in self.hidden_layers:
            module_list.append(nn.Linear(pre_layer,hidden_layer))
            module_list.append(nn.ReLU())
            if self.norm:
                module_list.append(RMSNormal(hidden_layer))
            if self.dropout>0:
                module_list.append(nn.Dropout(self.dropout))
            pre_layer=hidden_layer
        self.layers=nn.ModuleList(module_list)
    def forward(self,x):
        for layer  in  self.layers:
            x=layer(x)
        return x
'''
SwigluMLP is just for transformer MLP that is for decoder only
for swiglu,intermediate size should be the 8/3 times of hidden_dim.
'''
class SwiGluMLP(nn.Module):
    def __init__(self, hidden_dim,
                 intermediate_dim:int=None):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.intermediate_size=intermediate_dim if intermediate_dim is not None else int(8 / 3 * self.hidden_dim)
        self.gate_layer=nn.Linear(self.hidden_dim,self.intermediate_size)
        self.value_layer=nn.Linear(self.hidden_dim,self.intermediate_size)
        self.out_proj_layer=nn.Linear(self.intermediate_size, self.hidden_dim)
    '''
    x:(batch_size,seq_len,hidden_dim)
    return:(batch_size,seq_len,hidden_dim)
    '''
    def forward(self,x):
        gate_x=self.gate_layer(x)
        gate_x=gate_x*F.sigmoid(gate_x)
        value_x=self.value_layer(x)
        output=gate_x*value_x
        return self.out_proj_layer(output)