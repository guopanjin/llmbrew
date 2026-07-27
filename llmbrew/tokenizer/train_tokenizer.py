'''
pip install tokenizers  #for tokenizer training
pip install transformers  #for wrapping the tokenizer
'''
import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from llmbrew.config import data_config
from llmbrew.utils import  prevent_sleep
from transformers import PreTrainedTokenizerFast,AutoTokenizer
from llmbrew.config import data_config

tokenizer_data_paths=[data_config.llmbrew_corpus_tokenizer_en_v1_path,
                     data_config.llmbrew_corpus_tokenizer_zh_v1_path,
                     data_config.llmbrew_corpus_tokenizer_code_markdown_v1_path
                     ]
'''
Get raw text data
'''
def streaming_raw_data():
    global_line=0
    for data_path in tokenizer_data_paths:
        data_path=os.path.expanduser(data_path)
        with open(data_path,"r",encoding="utf-8") as f:
            for line in f:#streaming read line from local disk
                raw_text=json.loads(line)["text"]
                raw_text=raw_text.strip()
                global_line+=1
                if global_line%1000000==0:
                    print(f"global_line:{global_line}",flush=True)
                yield raw_text
def tokenizer_training(streaming_data,
                      vocab_size = 16000,
                      min_frequency = 2
                      ):
    #define tokenizer BPE trainer
    trainer=BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=ByteLevel.alphabet(),#setup default 256 byte
        special_tokens=[
            "<pad>",
            "<bos>",
            "<eos>",
            "<system>",
            "<user>",
            "<assistant>",
            "<unk>"
        ]
    )
    #define tokenizer
    tokenizer=Tokenizer(BPE(unk_token="<unk>"))#setup BPE model for tokenizer
    #convert text to bytes
    tokenizer.pre_tokenizer=ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(
        streaming_data,
        trainer=trainer
    )
    return tokenizer

def tokenizer_main():
    streaming_data=streaming_raw_data()
    tokenizer_path="../../artifacts/tokenizer/"
    #recursion create folder
    os.makedirs(tokenizer_path,exist_ok=True)
    tokenizer=tokenizer_training(streaming_data=streaming_data,
                       vocab_size=16000,
                       min_frequency=2
                       )
    tokenizer.save(os.path.join(tokenizer_path,"llmbrew_tokenizer_v1.json"))

def verify_tokenizer():
    tokenizer=Tokenizer.from_file("../../artifacts/tokenizer/llmbrew_tokenizer_v1.json")
    #1:verify the vocab_size
    vocab_size=tokenizer.get_vocab_size(with_added_tokens=True)
    print(f"vocab_size:{vocab_size}")
    #2:verify the special tokens
    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<system>",
        "<user>",
        "<assistant>",
        "<unk>"
    ]
    for special_token in special_tokens:
        print(f"{special_token}:{tokenizer.token_to_id(special_token)}")
    tokenizer.decoder = ByteLevelDecoder() # To convert byte to utf-8
    #verify the
    TEST_TEXTS = [
        "Hello, world!",
        "机器学习是人工智能的重要方向。",
        "Hello，中国！",
        "def forward(self, input_ids):\n"
        "    return self.embedding(input_ids)\n",
        "空格  测试\tTab\n换行",
        "😊🚀🔥",
        "𠮷野家",  # 生僻汉字
        "café naïve résumé",
        "数学符号：∑ ∫ √ ∞ ≠",
        "URL: https://example.com/a?id=123&x=y",
    ]
    for text in TEST_TEXTS:
        encoding=tokenizer.encode(text,add_special_tokens=False)
        decoded=tokenizer.decode(encoding.ids,skip_special_tokens=False)
        print(encoding.tokens)
        print(encoding.ids)
        print(f"text:{repr(text)}")# repr will show all character of str including \t \n
        print(f"decoded:{repr(decoded)}")

'''
wrap it by transformer,and then we can load based on from_pretrained
tokenizer = AutoTokenizer.from_pretrained("path")
'''
def wrap_tokenizer_with_transformer():
    hf_tokenizer_path="../../artifacts/tokenizer/llmbrew_tokenizer_v1/"
    os.makedirs(hf_tokenizer_path,exist_ok=True)
    tokenizer=Tokenizer.from_file("../../artifacts/tokenizer/llmbrew_tokenizer_v1.json")
    tokenizer.decoder=ByteLevelDecoder()
    hf_tokenizer=PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=[
            "<system>",
            "<user>",
            "<assistant>",
        ])
    hf_tokenizer.save_pretrained(hf_tokenizer_path)

def verify_transformer_tokenizer():
    tokenizer_path="../../artifacts/tokenizer/llmbrew_tokenizer_v1"
    hf_tokenizer:PreTrainedTokenizerFast=AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
    print(f"hf_tokenizer_type:{type(hf_tokenizer)}")
    print(f"is fast:{hf_tokenizer.is_fast}")
    print(f"vocab size:{hf_tokenizer.vocab_size}")
    print(f"specail token:{hf_tokenizer.special_tokens_map}")
    print(f"decoder:{hf_tokenizer.backend_tokenizer.decoder}")
    #verify encode and decode
    test_texts = [
        "Hello, world!",
        "机器学习是人工智能的重要方向。",
        "Hello，中国！",
        (
            "def forward(self, input_ids):\n"
            "    return self.embedding(input_ids)\n"
        ),
        "空格  测试\tTab\n换行",
        "😊🚀🔥",
        "𠮷野家",
        "café naïve résumé",
        "数学符号：∑ ∫ √ ∞ ≠",
        "URL: https://example.com/a?id=123&x=y",
    ]
    for text in test_texts:
        tokenids=hf_tokenizer.encode(text,add_special_tokens=False)
        decoded=hf_tokenizer.decode(tokenids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False
                            )
        print(f"raw text:{repr(text)}")
        print(f"decoded:{repr(decoded)}")
'''
Show all raw tokens to see if it meet our requirements.
'''
def show_raw_token():
    tokenizer_path = "../../artifacts/tokenizer/llmbrew_tokenizer_v1"
    hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    for token_id in range(len(hf_tokenizer)):
        token=hf_tokenizer.decode([token_id],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        print(f"{token_id}:{repr(token)}")
    pass
def get_specific_token_id(text="<eos>"):
    tokenizer_path = data_config.llmbrew_tokenizer_v1
    hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    token_ids=hf_tokenizer.encode(text)
    print(token_ids)
def get_special_token_id():
    tokenizer_path = data_config.llmbrew_tokenizer_v1
    hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    print(hf_tokenizer.eos_token_id)


if __name__ == '__main__':
    #tokenizer_main() #train the tokenizer model
    #verify_tokenizer() #verify the tokenizer
    #wrap_tokenizer_with_transformer()# warp by transformer
    #verify_transformer_tokenizer()
    #show_raw_token()
    #get_specific_token_id()
    get_special_token_id()
    pass
