import os
from llmbrew.utils import get_project_path
from transformers import PreTrainedTokenizerFast,AutoTokenizer
from llmbrew.utils import Logger
from llmbrew.constants.constant import Constant
logger=Logger.get_logger()
class TokenizerUtil:
    def __init__(self,tokenizer_path=None):
        if tokenizer_path is None:
            self.tokenizer_path=os.path.join(get_project_path(),"artifacts/tokenizer/llmbrew_tokenizer_v1/")
        else:
            self.tokenizer_path=tokenizer_path
        if not os.path.isdir(self.tokenizer_path):
            raise ValueError(f"the tokenizer path has some issue,path:{self.tokenizer_path}")
        self.hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.eos_id=self.convert_tokens_ids(Constant.SpecialToken.EOS)
        logger.info(f"successfully loaded tokenizer file {self.tokenizer_path}")
    def encode(self, text):
        tokenids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        return tokenids

    def decode(self, tokenids:list[int], is_repr:bool=False):
        decoded = self.hf_tokenizer.decode(tokenids,
                                      skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False
                                      )
        if is_repr:
            decoded=repr(decoded)
        return decoded
    def convert_tokens_ids(self,token):
        token_id=self.hf_tokenizer.convert_tokens_to_ids(token)
        return token_id