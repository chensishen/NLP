import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CBOW(nn.Module):
    def __init__(self,vocab_size,dim_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.dim_size = dim_size
        # 这个w就是需要训练学习的参数列表，也是最终期望输出的词向量矩阵(包含了每个单词对应的词向量)
        self.w = nn.Parameter(torch.randn(vocab_size, dim_size))
        # 输出层使用不同参数，也可以直接使用w(相当于复用)
        self.v = nn.Parameter(torch.randn(dim_size, vocab_size))

        self.emb_layer = nn.Embedding(vocab_size, dim_size)

    def forward(self,input_tokens):
        """
        :param input_tokens: [bs,m] bs个样本，每个样本输入m个token id
        :return:
        """
        # 1 针对 input_tokens id做哑编码, [bs,m] -> [bs,m,v]
        x = F.one_hot(input_tokens, self.vocab_size)
        x = x.to(dtype=self.w.dtype)
        # 2 获取每个单词对应的词向量 [bs,m,v]*[v,e] -> [bs,m,e]
        input_token_embs = torch.matmul(x, self.w)
        # 3. 求和 将m个单词的特征向量合并到一起 [bs,m,e] -> [bs, e]
        input_ctx_embs = torch.sum(input_token_embs, dim=1)
        # 4. 基于上下文求解属于各个单词的置信度 [bs,e] * [e,v] -> [bs,v]
        # z = torch.matmul(input_ctx_embs, self.w.T)
        z = torch.matmul(input_ctx_embs, self.v)
        return z


if __name__ == '__main__':
    net = CBOW(vocab_size=100, dim_size=4)
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    opt = optim.SGD(net.parameters(), lr=0.01)

    y = torch.tensor([5,13])
    r = net(
        torch.tensor([
            [1,2,3,4],
            [5,6,7,8]
    ]))
    print(r.shape)
    loss = loss_fn(y, r)
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()




