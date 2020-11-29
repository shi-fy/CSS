

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


# v和q连接，简单的生成attention
class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w:[k,1]
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        # num_objs为k
        num_objs = v.size(1)
        # q:[batch,1,k,qdim]
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        # vq:[batch,k,(vdim+qdim)]
        vq = torch.cat((v, q), 2)
        # joint_repr:[k,num_hid]
        joint_repr = self.nonlinear(vq)
        # logits:[k,1]
        logits = self.linear(joint_repr)
        return logits
# 对应位相乘，生成简单的attention
class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, 1)
        # return w
        return logits

    def logits(self, v, q):
        # v:[512,36,2048]
        # q:[512,1024]
        batch, k, _ = v.size()
        # v_proj:[512,36,1024]
        v_proj = self.v_proj(v)
        # q_proj:[512,36,1024]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        # joint_repr:[512,36,1024]
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        # logits:[512,36,1]
        return logits
