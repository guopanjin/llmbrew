from llmbrew.model import LLMBrewModel,LLMBrewConfig
from llmbrew.config import data_config,model_artifacts_config
from llmbrew.trainer import SFTTrainer
from llmbrew.utils import get_device,prevent_sleep
from llmbrew.dataset import SFTDataset
from torch.utils.data import Dataset,DataLoader
from llmbrew.model.optimizers import build_adamw_with_decay_groups
from llmbrew.model.optimizers import get_linear_scheduler
import torch
import os
from llmbrew.utils import Logger

logger=Logger.get_logger()
'''

'''
#data path
train_data_path=data_config.llmbrew_sft_train_path
validation_data_path=data_config.llmbrew_sft_validation_path
checkpoints_model_dir=model_artifacts_config.model_outputs_sft_checkpoints_path
best_model_dir =model_artifacts_config.model_outputs_sft_best_path
final_model_dir=model_artifacts_config.model_outputs_sft_final_model_path

##trainer config
train_batch_size=100
validation_batch_size=100
epoch_number=2
use_warm_up=True
warm_up_steps=100
checkpoints_step=500 #
max_checkpoints_to_keep=3
lr=2e-5

log_steps=5
validation_steps=200
use_early_stop=True
num_classes=16000
device="cpu"
#load model from artifacts
model_file_path=model_artifacts_config.model_outputs_pretrain_final_model_path+"pretrain_model.pt"
model_states=torch.load(os.path.expanduser(model_file_path),weights_only=False)
logger.info(f"load mode from {model_file_path}")
model_config=model_states["config"].__dict__
num_tokens=model_states["num_tokens"]
model_state_dict=model_states["model_state_dict"]
print("num_tokens",num_tokens)
print(model_config)

llmbrewconfig= LLMBrewConfig(**model_config)
model=LLMBrewModel(llmbrewconfig=llmbrewconfig)
model.load_state_dict(state_dict=model_state_dict)
model=model.to("cpu")

model_name="sft_model"
def train_main():
    train_dataset=SFTDataset(
        data_path=train_data_path,
        context_length=512,
        batch_size=train_batch_size,
    )
    train_dataloader=DataLoader(train_dataset,batch_size=None)

    validation_dataset = SFTDataset(
        data_path=validation_data_path,
        context_length=512,
        batch_size=validation_batch_size,
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=None)

    trainer=SFTTrainer(
        model_name=model_name,
        model=model,
        device=device,
        lr=lr,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch_number=epoch_number,
        use_warm_up=use_warm_up,
        warm_up_steps=warm_up_steps,
        validation_steps=validation_steps,
        log_steps=log_steps,
        use_early_stop=use_early_stop,
        num_classes=num_classes,
        checkpoints_step=checkpoints_step,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoints_model_dir=checkpoints_model_dir,
        best_model_dir=best_model_dir,
        final_model_dir=final_model_dir
    )
    trainer.train()


if __name__ == '__main__':
    prevent_sleep()
    train_main()
    pass