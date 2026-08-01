import torch
import torch.nn as nn

'''
m=beta*m+g
w=w-lr*m
'''
def build_sgd_momentum(model:nn.Module,
                       lr=0.1,
                       momentum=0.9,
                       weight_decay=1e-4
                       ):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    return optimizer



def build_adamw(model:nn.Module,
                lr=1e-3,
                weight_decay=1e-4
                ):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)
    return optimizer
'''
different parameters use different weightdecay
'''
def build_adamw_with_decay_groups(
              module:nn.Module,
              lr=1e-3,
              weight_decay=0.001
             ):
    weight_decay_parameters=[]
    no_weight_decay_parameters=[]
    for name,parameter in module.named_parameters():
        if len(parameter.shape)<2 or ".bias" in name:
            no_weight_decay_parameters.append(parameter)
        else:
            weight_decay_parameters.append(parameter)
    optimizer=torch.optim.AdamW([
        {
        "params":weight_decay_parameters,
        "weight_decay":weight_decay},
        {
         "params":no_weight_decay_parameters,
        "weight_decay":0}
    ],lr=lr)
    return optimizer