block_indexs=[0,1,2,3,4]
context_length=512
for block_index in block_indexs:
    start_index =block_index *(context_length +1)
    end_index =start_index +context_length +1
    print(start_index,end_index)