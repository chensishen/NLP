import torch
import torch.nn as nn
import torch.nn.functional as F

def t0():
    vocal_size = 2000

    embedding_layer = nn.Embedding(vocal_size, 300)
    for name, param in embedding_layer.named_parameters():
        print(name, "--->", param.shape)

    token_ids = torch.randint(0,vocal_size,size = (200,100))

    # 将token id转换为token的向量
    token_emb = embedding_layer(token_ids)
    print(token_emb.shape)

    print(torch.max(torch.abs(token_emb[0,1] - embedding_layer.weight[token_ids[0,1]])))

    # 等价先做哑编码，再做全连接
    token_onehot = F.one_hot(token_ids,num_classes=vocal_size)
    print(token_onehot.shape,token_onehot.dtype)
    token_onehot = token_onehot.to(embedding_layer.weight.dtype)
    token_onehot_emb = torch.matmul(token_onehot, embedding_layer.weight)
    print(torch.max(torch.abs(token_emb - token_onehot_emb)))

def t1():
    vocal_size = 1000
    embedding_layer = nn.Embedding(vocal_size, 128)
    conv1d = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

    token_ids = torch.randint(0,vocal_size,size=(200,100))

    # 1. 将token id转换为token向量 [bs,t] --> [bs,t,c]
    token_ems = embedding_layer(token_ids)
    print(token_ems.shape, token_ems.dtype)

    # 2. 卷积提取特征 N C W D
    conv1d_emds = conv1d(torch.transpose(token_ems, dim0=-1, dim1=-2))
    conv1d_emds = torch.permute(conv1d_emds, dims=(0, 2, 1))
    print(conv1d_emds.shape)

if __name__ == '__main__':
    # t0()
    t1()

    pass

























