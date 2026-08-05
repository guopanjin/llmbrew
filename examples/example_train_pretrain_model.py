from llmbrew.model import LLMBrewModel,LLMBrewConfig
from llmbrew.config import data_config,model_artifacts_config
from llmbrew.trainer import PretrainTrainer
from llmbrew.utils import get_device,prevent_sleep
from llmbrew.dataset import PretrainDataset
from torch.utils.data import Dataset,DataLoader

'''
2026-08-02 08:38:52 | INFO | pretrain_trainer.py:124 | {'model': 'pretrain_model', 'duration': '444.656min', 'stage': 'training', 'epoch': 1, 'step_size': 100, 'step_loss': 4.024149417877197, 'ema_loss': 3.9061509527569775, 'global_size': 432000, 'global_tokens': 221184000, 'global_step': 4320, 'grad_norm': tensor(0.3005), 'num_classes': 16000, 'step_loss_ppl': 55.9327, 'ema_loss_ppl': 49.7073}
2026-08-02 08:40:57 | INFO | pretrain_trainer.py:124 | {'model': 'pretrain_model', 'duration': '446.742min', 'stage': 'training', 'epoch': 1, 'step_size': 100, 'step_loss': 3.9475033283233643, 'ema_loss': 3.8866857063467353, 'global_size': 434000, 'global_tokens': 222208000, 'global_step': 4340, 'grad_norm': tensor(0.3015), 'num_classes': 16000, 'step_loss_ppl': 51.8059, 'ema_loss_ppl': 48.749}
2026-08-02 08:43:02 | INFO | pretrain_trainer.py:124 | {'model': 'pretrain_model', 'duration': '448.824min', 'stage': 'training', 'epoch': 1, 'step_size': 100, 'step_loss': 3.7992281913757324, 'ema_loss': 3.8530637970729438, 'global_size': 436000, 'global_tokens': 223232000, 'global_step': 4360, 'grad_norm': tensor(0.2963), 'num_classes': 16000, 'step_loss_ppl': 44.6667, 'ema_loss_ppl': 47.1373}
2026-08-02 08:44:57 | INFO | pretrain_dataset.py:70 | data_stage:validataion_stage,start to read data from /Users/jinguopan/.llmbrew/data/validation/v1/validation.bin,size=937236
2026-08-02 08:45:42 | INFO | pretrain_dataset.py:74 | read_end_index >size-1 break,read_end_index:974700,size-1:937235
2026-08-02 08:45:42 | INFO | pretrain_trainer.py:227 | saved the best loss 3.9727 old loss:3.9929
2026-08-02 08:45:42 | INFO | pretrain_trainer.py:234 | {'stage': 'validation', 'model_name': 'pretrain_model', 'epoch': 1, 'validation_number': 1800, 'validation_loss': 3.9727, 'normal_loss': 0.4104, 'validation_ppl': 53.1278, 'num_classes': 16000}
2026-08-02 08:45:43 | INFO | pretrain_trainer.py:208 | successfully save model to the path /Users/jinguopan/.llmbrew/model_outputs/pretrain/final_model/pretrain_model.pt


'''



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
checkpoints_step=500 #
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