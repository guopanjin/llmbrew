import os
from llmbrew.config import data_config
import json

'''
##token
~/.llmbrew/data/llmbrew_corpus_tokenizer_en_v1.jsonl
documents: 4272
characters: 20071702
~/.llmbrew/data/llmbrew_corpus_tokenizer_zh_v1.jsonl
documents: 14267
characters: 29952897
~/.llmbrew/data/llmbrew_corpus_tokenizer_code_markdown_v1.jsonl
documents: 2177
characters: 7199952



'''

tokenizer_data_paths=[data_config.llmbrew_corpus_tokenizer_en_v1_path,
                     data_config.llmbrew_corpus_tokenizer_zh_v1_path,
                     data_config.llmbrew_corpus_tokenizer_code_markdown_v1_path
                     ]

all_pretrain_data_paths = [data_config.llmbrew_pretrain_en_v1_path,
                      data_config.llmbrew_pretrain_zh_v1_path,
                      data_config.llmbrew_pretrain_code_v1_path
                      ]


def verify_num_chars(data_paths):
    for data_path in data_paths:
        total_chars = 0
        total_docs = 0

        with open(os.path.expanduser(data_path), "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                if isinstance(text, str) and text.strip():
                    total_chars += len(text)
                    total_docs += 1

        print(data_path)
        print("documents:", total_docs)
        print("characters:", total_chars)
    pass

if __name__ == '__main__':
    verify_num_chars(all_pretrain_data_paths)