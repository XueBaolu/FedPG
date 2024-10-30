import torch
# 按标签对齐，计算latent和prototype距离
def euclidean_dist(x, y):
    # x: N x D, zq特征向量数目为N(n_class*n_query)，维度为D(z_dim)
    # y: M x D, zs特征向量数目为M(n_class)，维度为D(z_dim)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)# 把x复制为N * M * D维
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
    # 计算每对样本之间的特征差的平方，然后在第 2 维上求和，得到平方差的总和。这是欧氏距离的一部分，即特征差的平方和。
    # 最后，函数返回每对样本之间的欧氏距离矩阵，形状为 (n, m)，其中每个元素表示一个样本到另一个样本的欧氏距离。

import torch.nn.functional as F
def proto_loss(index, dist):
    # nll_loss
    scale = 2
    log_p_y = F.log_softmax(-scale * dist, dim=1)

    loss = 0
    num = len(index)
    for i in range(num):
        loss += log_p_y[index[i], index[i]]
    loss = - loss / num

    return loss



