import torch

def get_linear_scheduler(*,
                         optimizer,
                         start_factor=0.1,
                         end_factor=1.0,
                         warmup_steps=1000
                         ):
    warm_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                       start_factor=start_factor,
                                                       end_factor=end_factor,
                                                       total_iters=warmup_steps)
    return warm_scheduler

def get_cosineAnnealingLR(*,
                          optimizer,
                          decay_steps,
                          min_lr=1e-4
                          ):
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                 T_max=decay_steps,
                                                                 eta_min=min_lr)
    return decay_scheduler


def get_constantLR(*,
                   optimizer,
                   base_lr=1e-3,
                   min_lr=1e-4,
                   constant_steps=10**9
                   ):
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                             factor=min_lr / base_lr,
                                                             total_iters=constant_steps)
    return constant_scheduler

class StreamingScheduler:
    def __init__(self,
                 optimizer,
                 min_lr=1e-4,
                 warmup_steps=1000,
                 hold_steps=100000,
                 decay_interval=10000,
                 decay_rate=0.95
                ):
        self.optimizer=optimizer
        self.min_lr=min_lr
        self.warmup_steps=warmup_steps
        self.hold_steps=hold_steps
        self.decay_interval=decay_interval
        self.decay_rate=decay_rate
        self.base_lr=optimizer.param_groups[0]['lr']
        self.step_number=0

    def get_lr(self,step_number:int)->float:
        if step_number <=self.warmup_steps:
            lr=self.base_lr*(step_number/self.warmup_steps)
            return lr
        elif step_number <=self.hold_steps:
            return self.base_lr
        else:
            exp=(self.step_number-self.warmup_steps)//self.decay_interval
            lr=max(self.min_lr, self.base_lr*(self.decay_rate**exp))
            return lr

    def step(self):
        self.step_number+=1
        lr=self.get_lr(self.step_number)
        #update lr
        self.optimizer.param_groups[0]['lr']=lr
        return lr
