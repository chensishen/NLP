import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SkipGram(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SkipGram, self).__init__()
        self.emb_table = nn.Embedding(vocab_size, hidden_size)
        self.fc_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, token_ids):
        """
        SkipGram
        :param token_ids: 中心词 [bs,1]
        :return:
        """
        # 1. 获取输入token的特征向量 [bs,1] -> [bs,1,hidden_size]
        x = self.emb_table(token_ids)
        # 2. 将输入token特征向量合并为上下文特征向量
        x = torch.mean(x, dim=1) # [bs,1,hidden_size]-> [bs,hidden_size]
        # 3. 基于合并的特征向量预测其它token属于各个类别的置信度
        score = self.fc_layer(x) # [bs,vocab_size]
        return score

if __name__ == '__main__':
    vocab_size = 100
    net = SkipGram(vocab_size, 4)
    # loss_fn = nn.CrossEntropyLoss()  # softmax交叉熵损失函数
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid交叉熵损失函数
    opt = optim.SGD(net.parameters(), lr=0.001)

    x = torch.tensor([[5],[13]])
    y = torch.tensor([
        [1,2,3,4],
        [5,6,7,8]
    ]) # [bs,t]
    y_one_hot = F.one_hot(y, vocab_size)
    y_one_hot = torch.max(y_one_hot, dim=1).values
    r = net(x)
    print(r.shape)
    loss = loss_fn(r, y_one_hot.to(dtype=r.dtype))
    print(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()













