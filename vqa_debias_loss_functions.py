from collections import OrderedDict, defaultdict, Counter

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import inspect


def convert_sigmoid_logits_to_binary_logprobs(logits):
    """computes log(sigmoid(logits)), log(1-sigmoid(logits))"""
    log_prob = -F.softplus(-logits)
    log_one_minus_prob = -logits + log_prob
    return log_prob, log_one_minus_prob


def elementwise_logsumexp(a, b):
    """computes log(exp(x) + exp(b))"""
    return torch.max(a, b) + torch.log1p(torch.exp(-torch.abs(a - b)))


def renormalize_binary_logits(a, b):
    """Normalize so exp(a) + exp(b) == 1"""
    norm = elementwise_logsumexp(a, b)
    return a - norm, b - norm


class DebiasLossFn(nn.Module):
    """General API for our loss functions"""

    def forward(self, hidden, logits, bias, labels):
        """
        :param hidden: [batch, n_hidden] hidden features from the last layer in the model
        :param logits: [batch, n_answers_options] sigmoid logits for each answer option
        :param bias: [batch, n_answers_options]
          bias probabilities for each answer option between 0 and 1
        :param labels: [batch, n_answers_options]
          scores for each answer option, between 0 and 1
        :return: Scalar loss
        """
        raise NotImplementedError()

    def to_json(self):
        """Get a json representation of this loss function.

        We construct this by looking up the __init__ args
        """
        cls = self.__class__
        init = cls.__init__
        if init is object.__init__:
            return []  # No init args
        # 获取到类的参数
        init_signature = inspect.getargspec(init)
        if init_signature.varargs is not None:
            raise NotImplementedError("varags not supported")
        if init_signature.keywords is not None:
            raise NotImplementedError("keywords not supported")
        args = [x for x in init_signature.args if x != "self"]
        out = OrderedDict()
        out["name"] = cls.__name__
        for key in args:
            out[key] = getattr(self, key)
        return out

# 使用binary_cross_entropy_with_logits来计算多分类的交叉熵损失
class Plain(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss *= labels.size(1)
        return loss


class Focal(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        # import pdb;pdb.set_trace()
        focal_logits=torch.log(F.softmax(logits,dim=1)+1e-5) * ((1-F.softmax(bias,dim=1))*(1-F.softmax(bias,dim=1)))
        loss=F.binary_cross_entropy_with_logits(focal_logits,labels)
        loss*=labels.size(1)
        return loss

class ReweightByInvBias(DebiasLossFn):
    def forward(self, hidden, logits, bias, labels):
        # Manually compute the binary cross entropy since the old version of torch always aggregates
        # 手动计算二进制交叉熵，因为旧版本的割炬总是聚合
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob)
        weights = (1 - bias)
        loss *= weights  # Apply the weights
        return loss.sum() / weights.sum()


class BiasProduct(DebiasLossFn):
    def __init__(self, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it 将学习到的Sigmoid（a）因子添加到偏差以使其平滑
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it 常数，添加到偏差中，使其更平滑
        """
        super(BiasProduct, self).__init__()
        self.constant_smooth = constant_smooth
        self.smooth_init = smooth_init
        self.smooth = smooth
        if smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        smooth = self.constant_smooth
        if self.smooth:
            smooth += F.sigmoid(self.smooth_param)

        # Convert the bias into log-space, with a factor for both the
        # binary outputs for each answer option
        # 将偏差转换为对数空间，并为每个答案选项的两个二进制输出都加上一个因子
        bias_lp = torch.log(bias + smooth)
        bias_l_inv = torch.log1p(-bias + smooth)

        # Convert the the logits into log-space with the same format
        # 将logits转换为相同格式的log-space
        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        # import pdb;pdb.set_trace()

        # Add the bias
        log_prob += bias_lp
        log_one_minus_prob += bias_l_inv

        # Re-normalize the factors in logspace
        log_prob, log_one_minus_prob = renormalize_binary_logits(log_prob, log_one_minus_prob)

        # Compute the binary cross entropy
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)
        return loss


class LearnedMixin(DebiasLossFn):
    def __init__(self, w, smooth=True, smooth_init=-1, constant_smooth=0.0):
        """
        :param w: Weight of the entropy penalty
        :param smooth: Add a learned sigmoid(a) factor to the bias to smooth it
        :param smooth_init: How to initialize `a`
        :param constant_smooth: Constant to add to the bias to smooth it
        """
        super(LearnedMixin, self).__init__()
        self.w = w
        # self.w=0
        self.smooth_init = smooth_init
        self.constant_smooth = constant_smooth
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.smooth = smooth
        if self.smooth:
            self.smooth_param = torch.nn.Parameter(
              torch.from_numpy(np.full((1,), smooth_init, dtype=np.float32)))
        else:
            self.smooth_param = None

    def forward(self, hidden, logits, bias, labels):
        # hidden:[512,1024]
        # logits:[512,2274]
        # bias:[512,2274]
        factor = self.bias_lin.forward(hidden)  # [batch, 1]
        factor = F.softplus(factor)

        bias = torch.stack([bias, 1 - bias], 2)  # [batch, n_answers, 2]

        # Smooth
        bias += self.constant_smooth
        if self.smooth:
            soften_factor = F.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)

        bias = torch.log(bias)  # Convert to logspace 转换成对数向量

        # Scale by the factor
        # [batch, n_answers, 2] * [batch, 1, 1] -> [batch, n_answers, 2]
        bias = bias * factor.unsqueeze(1)
        # 此时bias相当于g(xi)*log(bi)

        log_prob, log_one_minus_prob = convert_sigmoid_logits_to_binary_logprobs(logits)
        # 转换成双向对数概率
        log_probs = torch.stack([log_prob, log_one_minus_prob], 2)

        # Add the bias in 添加偏差
        logits = bias + log_probs
        # 这就相当于Learned-Mixin的概率公式

        # Renormalize to get log probabilities 重新正则化得到log概率
        log_prob, log_one_minus_prob = renormalize_binary_logits(logits[:, :, 0], logits[:, :, 1])

        # Compute loss
        loss = -(log_prob * labels + (1 - labels) * log_one_minus_prob).sum(1).mean(0)

        # Re-normalized version of the bias
        bias_norm = elementwise_logsumexp(bias[:, :, 0], bias[:, :, 1])
        bias_logprob = bias - bias_norm.unsqueeze(2)

        # Compute and add the entropy penalty
        # entropy实际计算的是文章3.2.5节的公式H（x）
        entropy = -(torch.exp(bias_logprob) * bias_logprob).sum(2).mean()

        # 虽然函数名是Learned-Mixin，但是实际计算结果是Learned-Mixin +H
        return loss + self.w * entropy