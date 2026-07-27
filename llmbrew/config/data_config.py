import os
from pathlib import Path

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
