from llmbrew.model import LLMBrewModel,LLMBrewConfig
from llmbrew.config import data_config
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
warm_up_steps=1000

validation_steps=1000
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
        batch_size=train_batch_size)
    train_dataloader=DataLoader(train_dataset,batch_size=None)

    validation_dataset = PretrainDataset(
        data_path=validation_data_path,
        context_length=context_length,
        batch_size=validation_batch_size)
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
        use_early_stop=use_early_stop,
        num_classes=num_classes)
    trainer.train()


if __name__ == '__main__':
    prevent_sleep()
    train_main()
    pass