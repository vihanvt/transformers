#implementing self attention to practice
import torch
import torch.nn as nn
print()
print("-------Preprocessing starts------",end="\n\n")
sentence="Life is short,eat shit first"
print(sentence)
dict={s:i for i,s in enumerate(sorted(sentence.replace(",",", ").split()))}
print(dict)
#Next, we use this dictionary to assign an integer index to each word:
tensed=torch.tensor([dict[sen] for sen in sentence.replace(",",", ").split()])
print(tensed,end="\n\n")
print("------Preprocessing over------",end="\n\n")

#embedding layer
vocab_size=100000

embedded=nn.Embedding(vocab_size,3)#(num_embedding,embedding_dim)
#dimension:(input_shape,embedding_dim)
embedded_sentence=embedded(tensed).detach()
print("This is the word embedding matrix ",end="\n\n")
print(embedded_sentence)
print("The shape of the matrix is",embedded_sentence.shape,end="\n\n")

'''the first q,k,v are created by multiplying
the weight matrix with the input embedding for the first input
i.e-> w_q*x^i,but after 1st layer we use the output from previous
layers to do the same'''

#lets implement for x^1
x_1=embedded_sentence[1]
q_dim=4 
embedding_dim=3
#w_query=torch.randn(embedding_dim,q_dim)
x_1=embedded_sentence[1]
d_q,d_k,d_v=2,2,4
w_q=torch.nn.Parameter(torch.rand(x_1.shape[0],d_q))
w_k=torch.nn.Parameter(torch.rand(x_1.shape[0],d_k))
w_v=torch.nn.Parameter(torch.rand(x_1.shape[0],d_v))

q_1=torch.matmul(x_1,w_q)
k_1=torch.matmul(x_1,w_k)
v_1=torch.matmul(x_1,w_v)
print("The query matrix initialised is", q_1,end="\n\n")
print("with dimensions", q_1.shape,end="\n\n")
print("Old Dimensions:(num_words,dim)---> New Dimensions: (num_words,d_q/k/v) where d_q,d_k,d_v are acting as the dimension of model not head splitting",end="\n\n")
print("In Generality:")
keys=torch.matmul(embedded_sentence,w_k)
values=torch.matmul(embedded_sentence,w_v)
print("The final keys and values tensors are of dimension(6,2)->keys and (6,4)-> values: ",end="\n\n")
print(keys)
print(values)
'''Input Embedding (num_of_words,dim 3)
   └─ W_query (3x2) → Query vector (num_of_wordsdim 2)
   └─ W_key   (3x2) → Key vector (num_of_words,dim 2)
   └─ W_value (3x4) → Value vector (num_of_words,dim 4)
   FOR LEARNING PURPOSES THE DMODEL IS SET TO 3 EARLIER AND
   THE DIMENSIONS FOR W_Q,W_K,W_V ARE ALSO VARIED TO 2,2,4
   OTHERWISE THE DIMENSIONS WOULD BE:
   input embedding: (seq,dmodel=512)
   W_q=(dmodel,dmodel=512)
   W_k=(dmodel,dmodel)
   W_v=(dmodel,dmodel)
   Q(query)=(seq,dmodel)'''

#now lets calculate the attention 
