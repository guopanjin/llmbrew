from llmbrew.utils import xxhash_encoder


print(xxhash_encoder("454545")%1000)