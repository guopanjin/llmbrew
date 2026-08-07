from llmbrew.tokenizer import TokenizerUtil
from llmbrew.constants import Constant

tokenizer_util=TokenizerUtil()
tokenids=tokenizer_util.encode("what is the capital of china?")
text=tokenizer_util.decode(tokenids)
print(tokenids)
print(text)

pad_id=tokenizer_util.convert_tokens_ids(Constant.SpecialToken.PAD)
print(pad_id)

