import torch
import torch.nn as nn
import torch.nn.functional as F

def t0():
    bs, t, e = 20, 10, 128
    v = 256
    gru = nn.GRU(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=False,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的GRU结构
    )
    print("GRU内部的参数shape:")
    for name, param in gru.named_parameters():
        print(name, "--->", param.shape)

    # 上一个模块的输出特征向量(Embedding模块)
    token_embs = torch.randn(bs, t, e)
    print(f"GRU输入的特征向量维度:{token_embs.shape}")

    # 调用gru
    gru_output, gru_h_n = gru(token_embs)
    print(f"GRU每个时刻的输出特征向量:\n\t{gru_output.shape}")
    print(f"GRU最后一个时刻的输出特征向量ht:\n\t{gru_output[:, -1]}")
    print(f"GRU第一个时刻的输出特征向量ht:\n\t{gru_output[:, 0]}")
    print(f"GRU最后一个时刻的输出特征向量ht:\n\t{gru_h_n}")


if __name__ == '__main__':
    t0()