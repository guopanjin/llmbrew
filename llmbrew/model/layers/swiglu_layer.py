import torch
import torch.nn as nn
import torch.nn.functional as F
'''
This is for a common swiglu,you don't need to specify the intermediate size for the
swiglu,Because the swiglu layer will automatically check the last dimension and based on this 
to initialize the parameter of matrix w1 and w2.
processing as following.
gate_x=x*w1
value_x=x*w2
output=silu(gate_x)*value_x # output shape should keep consistent with the input.
x=(batch_size,feature_size)
you should tell swigluLayer the feature_size
swigluLayer=SwigluLayer(x.shape[-1])
output=swigluLayer(x) #(batch_size,feature_size)
'''
class SwigluLayer(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim=input_dim
        self.gate_layer=nn.Linear(self.input_dim,self.input_dim)
        self.value_layer=nn.Linear(self.input_dim,self.input_dim)

    def forward(self,x):
        gate_x=self.gate_layer(x)
        gate_x=gate_x*F.sigmoid(gate_x)
        value_x=self.value_layer(x)
        return gate_x*value_x