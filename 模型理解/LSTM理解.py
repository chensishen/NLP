import torch
import torch.nn as nn
import torch.nn.functional as F

def t0():
    bs, t, e = 20, 10, 128
    v = 2*e
    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=True,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    # 上一个模块的输出特征向量(Embedding模块)
    token_embs = torch.randn(bs, t, e)
    print(f"LSTM输入的特征向量维度:{token_embs.shape}")

    # 调用lstm
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)
    print(f"LSTM每个时刻的输出特征向量:\n\t{lstm_output.shape}")
    print(f"LSTM最后一个时刻的输出特征向量ht:\n\t{lstm_output[:, -1, :].shape}")
    print(f"LSTM最后一个时刻的输出特征向量ht:\n\t{lstm_h_n.shape}")
    print(f"LSTM最后一个时刻的状态向量ct:\n\t{lstm_c_n.shape}")

    print(torch.max(torch.abs(lstm_output[:, -1, :][None,:]-lstm_h_n)))

def t1():
    """
    LSTM过程公式拆解
    :return:
    """
    bs, t, e = 20, 10, 128
    v = 2*e
    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=False,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    # 上一个模块的输出特征向量(Embedding模块)
    token_embs = torch.randn(bs, t, e)
    print(f"LSTM输入的特征向量维度:{token_embs.shape}")

    u_it, u_ft, u_ct, u_ot = torch.split(lstm.weight_ih_l0.T, split_size_or_sections=v, dim=1)
    w_it, w_ft, w_ct, w_ot = torch.split(lstm.weight_hh_l0.T, split_size_or_sections=v, dim=1)

    # 5. 解决全连接的特征问题: 全连接提取特征的时候仅考虑当前时刻的token输入，不考虑序列的特征
    new_token_embs_list = []
    ht = torch.zeros((bs, v))
    ct = torch.zeros((bs, v))
    for _t in range(t):
        # 遗忘门
        ft = F.sigmoid(torch.matmul(token_embs[:, _t, :], u_ft) + torch.matmul(ht, w_ft))
        # 更新门
        it = F.sigmoid(torch.matmul(token_embs[:, _t, :], u_it) + torch.matmul(ht, w_it))
        cur_ct = F.tanh(torch.matmul(token_embs[:, _t, :], u_ct) + torch.matmul(ht, w_ct))
        # 输出门
        ot = F.sigmoid(torch.matmul(token_embs[:, _t, :], u_ot) + torch.matmul(ht, w_ot))

        # 更新当前时刻对应的状态信息
        ct = ct * ft + cur_ct * it

        # 获取当前输出
        ht = ot * F.tanh(ct)

        oi = ht[:, None]  # 增加一个维度 [bs,64] -> [bs,1,64]
        new_token_embs_list.append(oi)

    new_token_embs = torch.concat(new_token_embs_list, dim=1)
    print(new_token_embs.shape)

    # LSTM的结果
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)

    print(torch.max(torch.abs(new_token_embs - lstm_output)))

def t3():
    e = 4
    v = 2*e
    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=True,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    # 上一个模块的输出特征向量(Embedding模块)
    token_embs = torch.randn(1, 10, e)
    token_embs = torch.tile(token_embs, dims=(2, 1, 1))  # 参数重复
    token_embs[0, 0] = torch.rand(e)
    print(token_embs)
    print(f"LSTM输入的特征向量维度:{token_embs.shape}")

    # 调用lstm
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)
    print(f"最后一个时刻的输出（最后一层）:\n\t{lstm_output[:, -1, :]}")
    print(f"第一个时刻的输出（第一层:\n\t{lstm_output[:, 0, :]}")
    print(f"状态信息输出:\n\t{lstm_h_n}")



if __name__ == '__main__':
    # t0()
    # t1()
    t3()

