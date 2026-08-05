import os
from pathlib import Path
def resolve_path(path):
    abs_path=os.path.expanduser(path)
    return abs_path

#tokenizer training raw data path
llmbrew_corpus_tokenizer_en_v1_path= "~/.llmbrew/data/llmbrew_corpus_tokenizer_en_v1.jsonl"
llmbrew_corpus_tokenizer_zh_v1_path= "~/.llmbrew/data/llmbrew_corpus_tokenizer_zh_v1.jsonl"
llmbrew_corpus_tokenizer_code_markdown_v1_path= "~/.llmbrew/data/llmbrew_corpus_tokenizer_code_markdown_v1.jsonl"

#pretraining raw data path
llmbrew_pretrain_en_v1_path="~/.llmbrew/data/llmbrew_pretrain_en_v1.jsonl"
llmbrew_pretrain_zh_v1_path= "~/.llmbrew/data/llmbrew_pretrain_zh_v1.jsonl"
llmbrew_pretrain_code_v1_path= "~/.llmbrew/data/llmbrew_pretrain_code_markdown_v1.jsonl"

#project root path
root_path=Path(__file__).resolve().parents[2]

#tokenizer path
llmbrew_tokenizer_v1=f"{root_path}/artifacts/tokenizer/llmbrew_tokenizer_v1"

#bin path
train_bin_path="~/.llmbrew/data/train/v1/"
validation_bin_path="~/.llmbrew/data/validation/v1/"

## sft raw data
llmbrew_sft_raw_data_en_file=resolve_path("~/.llmbrew/data/sft/raw_data/en_sft_rawdata/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
llmbrew_sft_raw_data_zh_file=resolve_path("~/.llmbrew/data/sft/raw_data/zh_sft_rawdata/alpaca_gpt4_data_zh.json")

# sft train and validation data
llmbrew_sft_train_path=resolve_path("~/.llmbrew/data/sft/v1/train/")
llmbrew_sft_validation_path=resolve_path("~/.llmbrew/data/sft/v1/validation/")