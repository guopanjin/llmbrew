from llmbrew.utils.log_utils import Logger
from llmbrew.utils import get_device
from llmbrew.model.optimizers import build_adamw_with_decay_groups
from llmbrew.model.optimizers import get_linear_scheduler
import torch
import torch.nn as nn
import os
import glob
import numpy as np
logger=Logger.get_logger()
import time
'''
1:train model
2:save model
3:load model
4:predict model

train_dataloader:
input:(
   feature_dict:(batch_size,seq_len),
   label_dict:label:(batch_size,seq_len)
)
'''
class PretrainTrainer:
    def __init__(self,
                 *,
                 model_name:str,
                 model:nn.Module,
                 loss_fn=None,
                 device=None,
                 train_dataloader,
                 validation_dataloader,
                 checkpoints_model_dir,
                 best_model_dir,
                 final_model_dir,
                 optimizer:torch.optim.Optimizer=None,
                 epoch_number=1,
                 use_warm_up=False,
                 warm_up_steps=100,
                 validation_steps=500,
                 log_steps=10,
                 checkpoints_step:int=1000,
                 max_checkpoints_to_keep:int=3,
                 ema_loss_alpha=0.1,
                 customize_initialization=False,
                 use_early_stop=False,
                 step_early_stop=False,
                 num_classes:int=None,
                 grad_max_norm:float=1.0
                 ):
        self.model_name=model_name
        self.model=model
        self.loss_fn=loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.optimizer=optimizer if optimizer is not None else build_adamw_with_decay_groups(self.model)
        self.use_warm_up=use_warm_up
        self.warm_up_steps=warm_up_steps
        self.train_dataloader=train_dataloader
        self.validation_dataloader=validation_dataloader
        self.checkpoints_model_dir=checkpoints_model_dir
        self.best_model_dir=best_model_dir
        self.final_model_dir=final_model_dir
        self.epoch_number=epoch_number
        self.device=device if device is not None else get_device()
        self.validation_steps=validation_steps
        self.log_steps=log_steps
        self.ema_loss_alpha=ema_loss_alpha
        self.customize_initialization=customize_initialization
        self.use_early_stop=use_early_stop
        self.step_early_stop=step_early_stop
        self.num_classes = num_classes
        self.grad_max_norm=grad_max_norm
        self.global_tokens=0
        self.global_size=0
        self.global_step=0
        self.checkpoints_step=checkpoints_step
        self.max_checkpoints_to_keep=max_checkpoints_to_keep
        self.best_validation_loss=float("inf")
        if self.use_warm_up:
            self.scheduler = get_linear_scheduler(optimizer=self.optimizer, warmup_steps=self.warm_up_steps)
        logger.info(f"device:{self.device}")
        logger.info(f"{self.model_name} structure:\n {self.model}")
        logger.info(f"pretrain trainer parameters config:{self.__dict__}")
    def train(self):
        early_stop=None
        if self.use_early_stop:
            early_stop=EarlyStop()
        model=self.model.to(self.device)
        model.train()
        #the merics we need to monitor
        step_size=0
        step_loss=0
        ema_loss=None
        start_time=time.time()
        for epoch in range(self.epoch_number):
            if self.use_early_stop:
                if early_stop.stopped():
                    logger.info(f"early stop trigged,epoch {epoch}")
                    break;
            for inputids,labels in self.train_dataloader:
                inputids=inputids.to(self.device)
                labels=labels.to(self.device)
                step_size = labels.shape[0]
                seq_len=labels.shape[1]
                labels=torch.reshape(labels,(-1,))
                logits=model(inputids) # (batch_size,seq_len,vocab_size)
                logits=torch.reshape(logits,(-1,logits.shape[-1]))#(batch_size*seq_len,vocab_size)
                loss=self.loss_fn(input=logits,target=labels)
                self.optimizer.zero_grad() #clear gradient
                loss.backward() # get the gradient
                #Add grad clip,in case some batches genearate abnormal graditent,and will break the normal parameters.
                #Retrun the raw grad norm before clipping
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=self.grad_max_norm)
                self.optimizer.step()
                ####evaluation part
                self.global_size+=step_size
                self.global_tokens+=seq_len*step_size
                self.global_step+=1
                step_loss=loss.detach().cpu().item()
                if ema_loss is None:
                    ema_loss=step_loss
                else:
                    ema_loss=self.ema_loss_alpha*step_loss + (1-self.ema_loss_alpha)*ema_loss
                if self.global_step % self.log_steps ==0:
                    end_time=time.time()
                    logger.info({
                        "model":self.model_name,
                        "duration":str(np.round((end_time-start_time)/60,3))+"min",
                        "stage":"training",
                        "epoch":epoch,
                        "step_size":step_size,
                        "step_loss":step_loss,
                        "ema_loss":ema_loss,
                        "global_size":self.global_size,
                        "global_tokens":self.global_tokens,
                        "global_step":self.global_step,
                        "grad_norm":grad_norm.detach().cpu().item(),
                        "num_classes":self.num_classes,
                        "step_loss_ppl": np.round(np.exp(step_loss),4),
                        "ema_loss_ppl":np.round(np.exp(ema_loss),4)
                    })
                if self.global_step % self.validation_steps ==0:
                    validation_loss=self._evaluation(epoch=epoch,model_name=self.model_name)
                    if self.use_early_stop and self.step_early_stop:
                        if early_stop.step(validation_loss=validation_loss,
                                         model=self.model
                                        ):
                            break;
                if self.use_warm_up:
                    self.scheduler.step()
                #save checkpoints
                if self.global_step>0 and self.global_step%self.checkpoints_step==0:
                    self._save_checkpoints()
            validation_loss=self._evaluation(epoch=epoch,model_name=self.model_name)
            if self.use_early_stop and not self.step_early_stop:
                if early_stop.step(validation_loss=validation_loss,
                                model=self.model
                                ):
                    break;

        if  self.use_early_stop and early_stop.best_state!=None:
            #restore the best model to self.model
            self.model.load_state_dict(early_stop.best_state) # in place operation
            logger.info("restore the best model weight to the current model")
        self._save_final_model()
    def _save_best_model(self):
        os.makedirs(os.path.expanduser(self.best_model_dir), exist_ok=True)
        checkpoint_state={
            "model_state_dict":self.model.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict(),
            "global_tokens":self.global_tokens,
            "global_size":self.global_size,
            "global_step":self.global_step,
            "best_loss":self.best_validation_loss
        }
        model_path=os.path.join(os.path.expanduser(self.best_model_dir),f"{self.model_name}_best.pt")
        torch.save(checkpoint_state,model_path)
    def _save_checkpoints(self):
        os.makedirs(os.path.expanduser(self.checkpoints_model_dir), exist_ok=True)
        checkpoint_state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_tokens": self.global_tokens,
            "global_size": self.global_size,
            "global_step": self.global_step
        }
        #save latest checkpoint
        model_path = os.path.join(os.path.expanduser(self.checkpoints_model_dir),
                                  f"{self.model_name}_{self.global_step}.pt")
        torch.save(checkpoint_state, model_path)
        #remove the old checkpoint
        paths_list=glob.glob(os.path.join(os.path.expanduser(self.checkpoints_model_dir),f"{self.model_name}_*.pt"))
        paths_tuple=[(path,int(path.split("_")[-1].split(".")[0])) for path in paths_list]
        sorted_paths_list=sorted(paths_tuple,key=lambda x:x[1],reverse=False)
        remove_paths=sorted_paths_list[:-self.max_checkpoints_to_keep]
        for remove_path in remove_paths:
            os.remove(remove_path[0])
            logger.info(f"remove old model {remove_path[0]}")

    def _save_final_model(self):
        os.makedirs(os.path.expanduser(self.final_model_dir),exist_ok=True)
        model_path=os.path.join(os.path.expanduser(self.final_model_dir),f"{self.model_name}.pt")
        final_model_state={"model_state_dict":self.model.state_dict(),
                           "config":self.model.llmbrewconfig,
                           "num_tokens":self.global_tokens
                           }
        torch.save(final_model_state,model_path)
        logger.info(f"successfully save model to the path {model_path}")
    @torch.no_grad()
    def _evaluation(self,*,epoch,model_name):
        self.model.eval()
        loss_sum=0
        size_sum=0
        for inputids,labels  in self.validation_dataloader:
            inputids = inputids.to(self.device)
            labels = labels.to(self.device)
            step_size=labels.shape[0]
            labels = torch.reshape(labels, (-1,))
            logits = self.model(inputids)  # (batch_size,seq_len,num_items+1)
            logits = torch.reshape(logits, (-1, logits.shape[-1]))
            # ignore_index,will ignore the value of label is 0
            loss = self.loss_fn(input=logits, target=labels)
            loss_sum+=loss.cpu().item()*step_size
            size_sum+=step_size
        validation_loss=np.round(loss_sum/size_sum,4)
        if validation_loss<self.best_validation_loss:
            logger.info(f"saved the best loss {validation_loss} old loss:{self.best_validation_loss}")
            #update best_validation_loss
            self.best_validation_loss=validation_loss
            self._save_best_model()
        normal_loss=0.0
        if self.num_classes is not None and self.num_classes>0:
            normal_loss = np.round(validation_loss/np.log(self.num_classes),4)
        logger.info({
            "stage":"validation",
            "model_name":model_name,
            "epoch":epoch,
            "validation_number":size_sum,
            "validation_loss":validation_loss,
            "normal_loss":normal_loss,
            "validation_ppl":np.round(np.exp(validation_loss),4),
            "num_classes":self.num_classes
        })
        self.model.train()
        return validation_loss
class EarlyStop:
    def __init__(self,
                 patience:int =5,
                 min_delta=0.0005
                 ):
        self.patience=patience
        self.min_delta=min_delta
        self.best_state=None
        self.bad_round=0
        self.best_loss=float("inf")
        self.is_stop=False
    def step(self,validation_loss,model:nn.Module):
        if  validation_loss < self.best_loss - self.min_delta:
            self.best_loss=validation_loss
            self.bad_round=0
            #becase model.state_dict() is reference not copy,so still changing,so need to copy that
            self.best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_round+=1
        if self.bad_round>=self.patience:
            logger.info(f"early stop,stop training,best_loss:{self.best_loss}, bad_round:{self.bad_round}, min_delta:{self.min_delta}")
            self.is_stop=True
            return True
        return False
    def stopped(self):
        return self.is_stop