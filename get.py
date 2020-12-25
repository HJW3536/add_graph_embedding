import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def getgraphvec():
    f = open('node2vec_256.pkl','rb')  # 256
    t = open('node2vec_512.pkl','rb')  # 512

    data1 = pickle.load(f)
    d = torch.Tensor(data1['solver']['vertex_embeddings'])
    # print(d.shape)

    data2 = pickle.load(t)
    c = torch.Tensor(data2['solver']['vertex_embeddings'])

    g = torch.cat((d,c),dim=1)
    gg = torch.zeros(1, 768)
    g = torch.cat((g, gg), 0)
    # print(g.shape)
    h = data1['graph']['name2id']
    # print(h)

    return h, g
def getembed(input_idss):
    graph_embeddings = nn.Embedding(1248, 768)
    pretrained_weight = np.array(g)  # 已有词向量的numpy形式
    graph_embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))  # from_numpy将其转化为tensor
    # graph_embeddings.weight.requires_grad = False
    k = graph_embeddings(Variable(torch.LongTensor(input_idss)))
    # k = k.unsqueeze(0)
    # print(k.shape) # [2, 768]
    return k

input_idss = []
name2id, g = getgraphvec()
tokens = [['本', '纯'], ['纯', '本'], ['本', '我'], ['我', '院']]  # inputs

for i in range(len(tokens)):
    temp = []
    # print(tokens[i])
    for j, k in enumerate(tokens[i]):
        if k in name2id:
            temp.append(name2id.get(k))
        else:
            temp.append(1247)
    input_idss.append(temp)
# print(input_idss)
embed = torch.zeros(0, 2, 768)
for i in range(len(input_idss)):
    E = getembed(input_idss[i])  # [1, 2, 768]
    E = E.unsqueeze(0)
    embed = torch.cat((embed, E), 0)
print(embed)
print(embed.shape)

# print(embed[0][0])
