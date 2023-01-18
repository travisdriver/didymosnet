"""DidymosNet descriptor class.

Adapted from https://github.com/yuruntian/HyNet/blob/master/model.py
"""
import math

import torch
import torch.nn as nn

from models.binary_conv import BinaryConv2d, BinaryQuantize

EPS_L2_NORM = 1e-10


def desc_l2norm(desc):
    """descriptors with shape NxC or NxCxHxW"""
    desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(EPS_L2_NORM).pow(0.5)
    return desc


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        FRN layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return "num_features={num_features}, eps={init_eps}".format(**self.__dict__)

    def forward(self, x):
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x


class TLU(nn.Module):
    def __init__(self, num_features):
        """
        TLU layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return "num_features={num_features}".format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


class DidymosNet(nn.Module):
    """DidymosNet model definition"""

    def __init__(
        self,
        is_bias=True,
        is_bias_FRN=True,
        is_scale_FRN=True,
        dim_desc=128,
        drop_rate=0.2,
    ):
        super().__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dim_desc = 256 if dim_desc == -1 else dim_desc

        # Layer 1
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        # Layer 2
        self.conv2 = BinaryConv2d(32, 32, kernel_size=3, padding=1, bias=is_bias)
        self.norm2 = FRN(32, is_bias=is_bias_FRN, is_scale=is_scale_FRN)
        self.actv2 = TLU(32)

        # Layer 3
        self.conv3 = BinaryConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias)
        self.norm3 = FRN(64, is_bias=is_bias_FRN, is_scale=is_scale_FRN)
        self.actv3 = TLU(64)

        # Layer 4
        self.conv4 = BinaryConv2d(64, 64, kernel_size=3, padding=1, bias=is_bias)
        self.norm4 = FRN(64, is_bias=is_bias_FRN, is_scale=is_scale_FRN)
        self.actv4 = TLU(64)

        # Layer 5
        self.conv5 = BinaryConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias)
        self.norm5 = FRN(128, is_bias=is_bias_FRN, is_scale=is_scale_FRN)
        self.actv5 = TLU(128)

        # Layer 6
        self.conv6 = BinaryConv2d(128, 128, kernel_size=3, padding=1, bias=is_bias)
        self.norm6 = FRN(128, is_bias=is_bias_FRN, is_scale=is_scale_FRN)
        self.actv6 = TLU(128)

        # Layer 7
        self.drop7 = nn.Dropout(self.drop_rate)
        self.conv7_0 = nn.Conv2d(128, 128, kernel_size=2, stride=2, bias=True)
        self.norm7_0 = nn.BatchNorm2d(128, affine=True)

        self.conv7_1 = nn.Conv2d(128, 128, kernel_size=2, stride=2, bias=True)
        self.norm7_1 = nn.BatchNorm2d(128, affine=True)

        self.conv7_2 = nn.Conv2d(128, dim_desc, kernel_size=2, stride=1, bias=True)
        self.norm7_2 = nn.BatchNorm2d(dim_desc, affine=False)

    def forward(self, x, mode="eval"):

        x = self.layer1(x)
        # print("Layer 1 shape:", x.shape)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.actv2(x)
        # print("Layer 2 shape:", x.shape)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.actv3(x)
        # print("Layer 3 shape:", x.shape)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.actv4(x)
        # print("Layer 4 shape:", x.shape)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.actv5(x)
        # print("Layer 5 shape:", x.shape)

        x = self.conv6(x)
        x = self.norm6(x)
        x = self.actv6(x)
        # print("Layer 6 shape:", x.shape)

        x = self.drop7(x)
        x = self.conv7_0(x)
        x = self.norm7_0(x)
        x = self.conv7_1(x)
        x = self.norm7_1(x)
        x = self.conv7_2(x)
        x = self.norm7_2(x)
        # print("Layer 7 shape:", x.shape)

        desc_raw = x.squeeze(-1).squeeze(-1)
        if self.dim_desc == -1:
            desc = BinaryQuantize().apply(desc_raw, self.conv6.k, self.conv6.t)
            desc /= 16
        else:
            desc = desc_l2norm(desc_raw)

        if mode == "train":
            return desc, desc_raw
        elif mode == "eval":
            return desc

    def adjust_kt(self, i, n, T_min=1e-1, T_max=1e1):
        """
        Ref:
        -https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w32a/trainer.py#L139
        """
        K_min, K_max = math.log(T_min) / math.log(10), math.log(T_max) / math.log(10)
        t = torch.tensor([math.pow(10, K_min + (K_max - K_min) / n * i)]).float().cuda()
        if t < 1:
            k = 1 / t
        else:
            k = torch.tensor([1]).float().cuda()

        self.conv2.k = k
        self.conv2.t = t
        self.conv3.k = k
        self.conv3.t = t
        self.conv4.k = k
        self.conv4.t = t
        self.conv5.k = k
        self.conv5.t = t
        self.conv6.k = k
        self.conv6.t = t
