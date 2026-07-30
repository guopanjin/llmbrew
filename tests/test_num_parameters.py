
hidden_dim=320
intermediate=864
num_decoder_layer=4
embedding_parameter=16000*320
mha_parameter=(hidden_dim*hidden_dim)*4
swiglu_mlp_paramter=(hidden_dim*intermediate)*3
one_decoder_paramter=mha_parameter +swiglu_mlp_paramter
print(embedding_parameter)
print(mha_parameter)
print(swiglu_mlp_paramter)
print(one_decoder_paramter)
print(num_decoder_layer*one_decoder_paramter+embedding_parameter)