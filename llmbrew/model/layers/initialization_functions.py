import torch
import torch.nn as nn
import math
def init_embedding(module:nn.Module,
                std=0.01
               ):
    if isinstance(module,nn.Embedding):
        nn.init.normal_(module.weight,std=std)
        with torch.no_grad():
            if module.padding_idx is not None:
                module.weight[module.padding_idx].fill_(0.0)
def init_module(module:nn.Module,
                init_name='xavier',
                std=0.02
               ):
    if isinstance(module,nn.Linear):
        if init_name=='xavier':
            nn.init.xavier_normal_(module.weight)
        elif init_name=='kaiming':
            nn.init.kaiming_normal_(module.weight,nonlinearity='relu')
        else:
            raise ValueError("does not support this type of initilization method")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module,nn.Embedding):
        nn.init.normal_(module.weight,std=std)
def  init_residual_layer(model:nn.Module,
                         layer_numbers:int,
                         match_name=["out_proj","fc2"],
                         residual_number_per=2
                        ):

    scale= math.sqrt(1/(layer_numbers*residual_number_per))
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if name.endswith(".weight") and any(k in name  for k in match_name):
                parameter.mul_(scale)