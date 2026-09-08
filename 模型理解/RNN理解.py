import torch
import torch.nn as nn
import torch.nn.functional as F

def t0():
    """
    文本分类的通用的简化过程
    :return:
    """
    token2idx = {
        # .......
        '从': 23,
        # .......
        '这里': 58,
        # .......
        '怎么': 102,
        # .......
        '回家': 203,
        # .......
    }
    embedding_layer = nn.Embedding(num_embeddings=300,embedding_dim=8)

    # 1. 最原始数据
    text = "从这里怎么回家"

    # 2. 分词
    tokens = ['从', '这里', '怎么', '回家']

    # 3. 基于构建好的单词id映射mapping将每个token转换为id
    token_ids = [
        [23, 58, 102, 203]
    ]

    # 4. token id做embedding得到对应的token特征向量
    # [bs,t,e] --> bs个样本，每个样本由t个token组成，每个token对应的稠密特征向量的维度大小为e
    token_embs = embedding_layer(torch.tensor(token_ids))
    print(token_embs.shape)

    # 5. 接下来的网络结构应该就是基于token的特征向量进一步的进行特征的组合/提取 --> 最终得到文本向量 ---> 基于文本向量进行最终的决策输出，得到属于各个类别的置信度信息


    pass

def t1_fc():
    """
    基于全连接提取token特征向量：
        问题：token之间没有进行特征的交互，token的提取仅基于当前token，没有考虑上下文
    :return:
    """
    bs, t, e = 20, 5, 128
    dtype = torch.float32
    linear = torch.nn.Linear(in_features=e,out_features=2*e,dtype=dtype)
    token_embs = torch.randn(bs,t,e,dtype=dtype)

    print(linear.weight.shape,linear.bias.shape)

    # 5. 全连接提取特征向量
    token_new_embs = linear(token_embs)
    print('token_new_embs.shape',token_new_embs.shape)

    ## PS: 将全连接的过程拆开写
    token_new_embs2 = []
    for i in range(t):
        # 当前时刻的输入特征向量
        xi = token_embs[:, i, :]  # [bs,e]
        # 当前时刻的输入特征向量进行提取转换
        hi = torch.matmul(xi, linear.weight.T) + linear.bias
        # 合并输出
        oi = hi[:, None]  # [bs,2e] --> [bs, 1, 2e]
        token_new_embs2.append(oi)
    token_new_embs2 = torch.concat(token_new_embs2, dim=1)
    print(torch.max(torch.abs(token_new_embs2 - token_new_embs)))

    # 6. 文本特征向量
    text_emb = torch.mean(token_new_embs, dim=1)
    print('text_emb.shape',text_emb.shape)

def t2_conv1d():
    """
    基于1D卷积提取token特征向量：
        问题：token的特征向量融合只有局部的融合，没有全局的
    :return:
    """
    bs, t, e = 20, 5, 128
    dtype = torch.float32
    conv1d = torch.nn.Conv1d(in_channels=e,out_channels=2*e,kernel_size=3,stride=1,padding=1,dtype=dtype)
    token_embs = torch.randn(bs,t,e,dtype=dtype)

    for name, param in conv1d.named_parameters():
        print(name, param.shape)

    # 5. 提取特征向量
    token_embs = torch.permute(token_embs, (0, 2, 1)) # [bs,t,e] --> [bs,e,t]
    token_new_embs = conv1d(token_embs) # [bs,e,t] --> [bs,2e,t]
    token_new_embs = torch.permute(token_new_embs, (0, 2, 1))  # [bs,e,t] --> [bs,t,e]
    print('token_new_embs.shape',token_new_embs.shape)

    # 6. 文本特征向量
    text_emb = torch.mean(token_new_embs, dim=1)
    print('text_emb.shape',text_emb.shape)

def t3_rnnv1():
    """
    基于RNN提取特征的过程拆解：
    :return:
    """
    bs, t, e = 20, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    ih_linear = nn.Linear(e, e * 2, dtype=dtype)
    hh_linear = nn.Linear(2 * e, e * 2, dtype=dtype)

    w_ih,b_ih = ih_linear.weight.T, ih_linear.bias
    w_hh,b_hh = hh_linear.weight.T, hh_linear.bias

    ## RNN特征向量提取过程拆解
    token_new_embs = []
    hi = torch.zeros((bs,2*e))
    for i in range(t):
        # for i in range(t - 1, -1, -1):
        # 当前时刻的输入特征向量
        xi = token_embs[:, i, :]  # [bs,e]
        # 当前时刻的输入特征向量进行提取转换
        xhi = torch.matmul(xi, w_ih)+b_ih # [bs,e] * [e,2e] + [2e] --> [bs,2e]
        # 上一个时刻特征的加权转换
        hhi = torch.matmul(hi, w_hh)+b_hh# [bs,2e] * [2e,2e] + [2e] --> [bs,2e]
        # 将当前输入特征和上一个时刻的特征进行合并
        hi = xhi + hhi
        hi = F.relu(hi)
        # 合并输出
        oi = hi[:, None]  # [bs,2e] --> [bs, 1, 2e]
        token_new_embs.append(oi)
    token_new_embs = torch.concat(token_new_embs, dim=1)
    print(token_new_embs.shape)

    # -------------------------------------------
    # for i in range(t):
    #     xi = token_embs[:, i, :]
    #     xhi = ih_linear(xi)
    #
    #     hhi = hh_linear(hi)
    #
    #     hi = xhi + hhi
    #
    #     hi = F.relu(hi)
    #
    #     oi = hi[:, None]
    #
    #     token_new_embs.append(oi)
    # token_new_embs = torch.concat(token_new_embs, dim=1)
    # print(token_new_embs.shape)

    # 6. 文本特征向量
    text_emb = torch.mean(token_new_embs, dim=1)
    print(text_emb.shape)

def t3_rnnv2():
    """
    基于RNN提取特征的过程拆解 + 双向
    :return:
    """
    bs, t, e = 20, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    ih_linear = nn.Linear(e, e * 2, dtype=dtype)
    hh_linear = nn.Linear(2 * e, e * 2, dtype=dtype)
    ih_linear_r = nn.Linear(e, e * 2, dtype=dtype)
    hh_linear_r = nn.Linear(2 * e, e * 2, dtype=dtype)

    w_ih, b_ih = ih_linear.weight.T, ih_linear.bias
    w_hh, b_hh = hh_linear.weight.T, hh_linear.bias
    w_ih_r, b_ih_r = ih_linear_r.weight.T, ih_linear_r.bias
    w_hh_r, b_hh_r = hh_linear_r.weight.T, hh_linear_r.bias

    ## RNN特征向量提取过程拆解
    token_new_embs = []
    hi = torch.zeros((bs, 2 * e))
    for i in range(t):
        # for i in range(t - 1, -1, -1):
        # 当前时刻的输入特征向量
        xi = token_embs[:, i, :]  # [bs,e]
        # 当前时刻的输入特征向量进行提取转换
        xhi = torch.matmul(xi, w_ih) + b_ih  # [bs,e] * [e,2e] + [2e] --> [bs,2e]
        # 上一个时刻特征的加权转换
        hhi = torch.matmul(hi, w_hh) + b_hh  # [bs,2e] * [2e,2e] + [2e] --> [bs,2e]
        # 将当前输入特征和上一个时刻的特征进行合并
        hi = xhi + hhi
        hi = F.relu(hi)
        # 合并输出
        oi = hi[:, None]  # [bs,2e] --> [bs, 1, 2e]
        token_new_embs.append(oi)
    token_new_embs = torch.concat(token_new_embs, dim=1)
    print(token_new_embs.shape)

    token_new_embs_r = []
    hi_r = torch.zeros((bs, 2 * e))
    for i in reversed(range(t)):
        # for i in range(t - 1, -1, -1):
        # 当前时刻的输入特征向量
        xi_r = token_embs[:, i, :]  # [bs,e]
        # 当前时刻的输入特征向量进行提取转换
        xhi_r = torch.matmul(xi_r, w_ih_r) + b_ih_r  # [bs,e] * [e,2e] + [2e] --> [bs,2e]
        # 上一个时刻特征的加权转换
        hhi_r = torch.matmul(hi_r, w_hh_r) + b_hh_r  # [bs,2e] * [2e,2e] + [2e] --> [bs,2e]
        # 将当前输入特征和上一个时刻的特征进行合并
        hi_r = xhi_r + hhi_r
        hi_r = F.relu(hi_r)
        # 合并输出
        oi_r = hi_r[:, None]  # [bs,2e] --> [bs, 1, 2e]
        token_new_embs_r.append(oi_r)

    token_new_embs_r = torch.concat(token_new_embs_r, dim=1)
    print(token_new_embs_r.shape)

    # 正向和反向的合并
    token_new_embs = torch.concat([token_new_embs, token_new_embs_r], dim=-1)
    print(token_new_embs.shape)

    # 6. 文本特征向量
    text_emb = torch.mean(token_new_embs, dim=1)
    print(text_emb.shape)

def t3_rnnv3():
    """
    RNN API使用
    :return:
    """
    bs, t, e = 20, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs,t,e,dtype=dtype)

    rnn = torch.nn.RNN(
        input_size=e,  # 每个时刻的输入特征向量维度
        hidden_size=2 * e,  # 每个时刻期望输出的特征向量维度大小，也就是细胞信息的维度大小
        num_layers=2,  # RNN的层数
        nonlinearity='tanh',  # 激活函数，当前API仅支持tanh和relu
        bias=True,  # 内部线性转换的时候是否有bias操作
        batch_first=True,  # 输入数据中bs这个维度是第一维还是第二维(是不是在最前面)，True表示shape为[bs,t,e]; False表示[t,bs,e]
        dropout=0.0,
        bidirectional=True,  # 是否是双向的RNN结构
        device=None,
        dtype=dtype
    )
    for name, param in rnn.named_parameters():
        print(name, param.shape)

    # 5. 提取特征向量
    # rnn_output: [bs,t,v] RNN输出的每个样本、每个token对应的v维的特征向量(如果是多层rnn结构，仅最后一层输出)
    # rnn_state:[num_layers*bidirectional,bs,v]  序列最后一个时刻/token对应的状态信息 --> 在RNN中，就是和最后一个时刻的输出值一样
    rnn_output, rnn_state = rnn(token_embs)
    print(rnn_output.shape)
    print(rnn_state.shape)
    print(torch.max(torch.abs(torch.stack([rnn_output[:,-1,:256],rnn_output[:,0,256:]],dim=0) - rnn_state[-2:,:,:])))


    # 6. 文本特征向量
    text_emb = torch.mean(rnn_output, dim=1)
    print('text_emb.shape',text_emb.shape)

def t3_rnnv4():
    """
    RNN API使用 和 RNN过程拆解进行比较
    :return:
    """
    bs, t, e = 20, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量
    rnn = nn.RNN(
        input_size=e,  # 每个时刻的输入特征向量维度
        hidden_size=2 * e,  # 每个时刻期望输出的特征向量维度大小，也就是细胞信息的维度大小
        num_layers=1,  # RNN的层数
        nonlinearity='tanh',  # 激活函数，当前API仅支持tanh和relu
        bias=True,  # 内部线性转换的时候是否有bias操作
        batch_first=True,  # 输入数据中bs这个维度是第一维还是第二维(是不是在最前面)，True表示shape为[bs,t,e]; False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False,  # 是否是双向的RNN结构
        device=None,
        dtype=dtype
    )
    for name, param in rnn.named_parameters():
        print(name, "---->", param.shape)

    # 5. 提取特征向量
    # rnn_output: [bs,t,v] RNN输出的每个样本、每个token对应的v维的特征向量(如果是多层rnn结构，仅最后一层输出)
    # rnn_state:[?,bs,v]  序列最后一个时刻/token对应的状态信息 --> 在RNN中，就是和最后一个时刻的输出值一样
    ##### ?表示的是 ?=(2 if bidirectional else 1) * num_layers
    rnn_output, rnn_state = rnn(token_embs)
    print(rnn_output.shape)
    print(rnn_state.shape)

    # 6. 文本特征向量
    text_emb = torch.mean(rnn_output, dim=1)
    print(text_emb.shape)

    ####
    uw, ub = rnn.weight_ih_l0.T, rnn.bias_ih_l0
    ww, wb = rnn.weight_hh_l0.T, rnn.bias_hh_l0

## RNN特征向量提取过程拆解
    token_new_embs = []
    hi = torch.zeros((bs, 2 * e))
    for i in range(t):
        # 当前时刻的输入特征向量
        xi = token_embs[:, i, :]  # [bs,e]
        # 当前时刻的输入特征向量进行提取转换
        xhi = torch.matmul(xi, uw) + ub  # [bs,e] * [e,2e] + [2e] --> [bs,2e]
        # 上一个时刻特征的加权转换
        # [bs,2e] * [2e,2e] + [2e] --> [bs,2e]
        hhi = torch.matmul(hi, ww) + wb
        # 将当前输入特征和上一个时刻的特征进行合并
        hi = xhi + hhi
        hi = F.tanh(hi)
        # 合并输出
        oi = hi[:, None]  # [bs,2e] --> [bs, 1, 2e]
        token_new_embs.append(oi)
    token_new_embs = torch.concat(token_new_embs, dim=1)
    print(token_new_embs.shape)
    print(torch.max(torch.abs(token_new_embs - rnn_output)))

def t3_rnnv5():
    """
    长时依赖问题
    :return:
    """
    e = 4
    v = 8
    rnn = nn.RNN(
        input_size=e,  # 每个时刻/每个token输入的特征向量维度大小
        hidden_size=v,  # 期望每个时刻/每个token输出的特征向量维度大小
        num_layers=1,  # 给定有多少层RNN结构
        nonlinearity='tanh',  # 给定激活函数
        bias=False,  # 内部线性转换是否添加bias
        batch_first=True,  # 输入数据的shape形状, true表示输入shape为[bs,t,e], false表示输入shape为[t,bs,e]
        bidirectional=False,  # 给定当前结构是否是双向RNN，False表示不是
    )
    print("RNN内部的参数shape:")
    for name, param in rnn.named_parameters():
        print(name, "--->", param.shape)

    # 4. token_ids哑编码 + 每个单词(哑编码的向量)进行全连接转换 每个token都得到一个稠密的项目
    # bs个样本，每个样本由t个token组成，每个token对应的稠密特征向量的维度大小为128
    # bs: batch size ; t: 时刻/序列长度; e: 每个token对应的向量维度大小
    token_embs = torch.rand(1, 10, e)  # [bs,t,e]
    token_embs = torch.tile(token_embs, dims=(2, 1, 1))  # 参数重复
    token_embs[0, 0] = torch.rand(e)
    print(token_embs)
    bs, t, e = token_embs.shape

    # 5. rnn的调用
    output, h_n = rnn(token_embs)
    print(output.shape)
    print(h_n.shape)
    print(f"最后一个时刻的输出（最后一层）:\n{output[:, -1, :]}")
    print(f"第一个时刻的输出（第一层）:\n{output[:, 0, :]}")
    print(f"状态信息输出:\n{h_n}")

if __name__ == '__main__':
    # t0()
    # t1_fc()
    # t2_conv1d()
    # t3_rnnv1()
    # t3_rnnv2()
    t3_rnnv3()
    # t3_rnnv4()
    # t3_rnnv5()
    pass











