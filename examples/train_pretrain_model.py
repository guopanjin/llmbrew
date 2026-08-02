from llmbrew.model import LLMBrewModel,LLMBrewConfig
from llmbrew.config import data_config,model_artifacts_config
from llmbrew.trainer import PretrainTrainer
from llmbrew.utils import get_device,prevent_sleep
from llmbrew.dataset import PretrainDataset
from torch.utils.data import Dataset,DataLoader
#data path
train_data_path=data_config.train_bin_path
validation_data_path=data_config.validation_bin_path

# model config
vocab_size: int = 16000
hidden_dim: int = 320
num_heads: int = 5
intermediate_dim: int = 864
num_decoder_layers: int = 4
causal_mask: int = True
context_length=512

##trainer config
train_batch_size=100
validation_batch_size=100
epoch_number=2
use_warm_up=True
warm_up_steps=200
checkpoints_step=500
max_checkpoints_to_keep=3
checkpoints_model_dir=model_artifacts_config.model_outputs_pretrain_checkpoints_path
best_model_dir =model_artifacts_config.model_outputs_pretrain_best_path
final_model_dir=model_artifacts_config.model_outputs_pretrain_final_model_path
log_steps=20
validation_steps=500
use_early_stop=False
num_classes=vocab_size




llmbrew_config=LLMBrewConfig(
             vocab_size = vocab_size,
             hidden_dim  = hidden_dim,
             num_heads  =num_heads,
             intermediate_dim = intermediate_dim,
             num_decoder_layers  = num_decoder_layers,
             causal_mask = causal_mask)
model=LLMBrewModel(llmbrew_config)
device="cpu"
model_name="pretrain_model"
def train_main():
    train_dataset=PretrainDataset(
        data_path=train_data_path,
        context_length=context_length,
        batch_size=train_batch_size,
        is_train=True
    )
    train_dataloader=DataLoader(train_dataset,batch_size=None)

    validation_dataset = PretrainDataset(
        data_path=validation_data_path,
        context_length=context_length,
        batch_size=validation_batch_size,
        is_train=False
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=None)

    trainer=PretrainTrainer(
        model_name=model_name,
        model=model,
        device=device,
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