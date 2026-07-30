import torch
import torch.nn as nn
import torch.nn.functional as F
from llmbrew.model.layers import RMSNormal
from llmbrew.model.layers import MultiHeadAttention
from llmbrew.model.layers import SwiGluMLP
'''
one decoder layer:
input:x (batch_size,seq_len,embedding_dim)
x->Rmsnorm(x)-->mha-->x1-->x1+x-->x2
x2->Rmsnorm(x2)--->swiglumlp-->x3
output:x3+x2
'''
class DecoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim:int,
                 num_heads:int,
                 intermediate_dim:int=None,
                 causal_mask:bool=True
                 ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.intermediate_dim=intermediate_dim
        self.causal_mask=causal_mask
        self.mha=MultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            causal_mask=self.causal_mask)
        self.swiglu_mlp=SwiGluMLP(hidden_dim=self.hidden_dim,intermediate_dim=self.intermediate_dim)
        self.mha_rms_normal=RMSNormal(self.hidden_dim)
        self.mlp_rms_normal = RMSNormal(self.hidden_dim)
    def forward(self,x):
        mha_norm=self.mha_rms_normal(x)
        output_mha=self.mha(mha_norm) +x #mha + residual
        mlp_norm=self.mlp_rms_normal(output_mha)
        mlp_output=self.swiglu_mlp(mlp_norm)+output_mha #mlp +residual
        return mlp_output