import torch
import torch.nn as nn
from torch.autograd import Variable

from abc import ABCMeta, abstractmethod

class AbstractPrimitive(nn.Module, metaclass=ABCMeta):
    """
    Use this class when creating new operations for edges.
    """

    def __init__(self, *args, **kwargs):
        super(AbstractPrimitive, self).__init__(*args, **kwargs)
    
    @abstractmethod
    def forward(self, x, edge_data):
        """
        The forward processing of the operation.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_embedded_ops(self):
        """
        Return any embedded ops so that they can be
        analysed whether they contain a child graph, e.g.
        a 'motif' in the hierachical search space.

        If there are no embedded ops, then simply return
        `None`. Should return a list otherwise.
        """
        raise NotImplementedError()


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, edge_data):
        return x

    def get_embedded_ops(self):
        return None


class Zero(AbstractPrimitive):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride


    def forward(self, x, edge_data):
        if self.stride == 1:
            return x.mul(0.)
        else:
            x = x[:, :, ::self.stride, ::self.stride].mul(0.)
            return torch.cat([x, x], dim=1)   # double the channels TODO: ugly as hell
    
    def get_embedded_ops(self):
        return None


class ModuleWrapper(AbstractPrimitive):

    def __init__(self, module):
        super(ModuleWrapper, self).__init__()
        self.module = module

    def forward(self, x, edge_data):
        return self.module.forward(x)
    
    def get_embedded_ops(self):
        return None


class SepConv(AbstractPrimitive):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, edge_data):
        return self.op(x)
    
    def get_embedded_ops(self):
        return None


class DilConv(AbstractPrimitive):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, edge_data):
        return self.op(x)


    def get_embedded_ops(self):
        return None


class Stem(AbstractPrimitive):

    def __init__(self, C_curr):
        super(Stem, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))

    def forward(self, x, edge_data):
        return self.seq(x)
    
    def get_embedded_ops(self):
        return None




class Sequential(AbstractPrimitive):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.primitives = args
        self.op = nn.Sequential(*args)
    
    def forward(self, x, edge_data):
        return self.op(x)
    
    def get_embedded_ops(self):
        return list(self.primitives)


class FactorizedReduce(AbstractPrimitive):
    """
    Whatever this is, it replaces the identiy when stride=2
    """

    def __init__(self, C_in, C_out, affine=False):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, edge_data):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
    
    def get_embedded_ops(self):
        return None


class MaxPool1x1(AbstractPrimitive):

    def __init__(self, kernel_size, stride, C_in=None, C_out=None, affine=False):
        super(MaxPool1x1, self).__init__()
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=1)
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, edge_data):
        x = self.maxpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x

    def get_embedded_ops(self):
        return None


class AvgPool1x1(AbstractPrimitive):

    def __init__(self, kernel_size, stride, C_in=None, C_out=None, affine=False):
        super(AvgPool1x1, self).__init__()
        self.stride = stride
        self.avgpool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, edge_data):
        x = self.avgpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x

    def get_embedded_ops(self):
        return None


if __name__ == '__main__':
    i = Identity()
    print(issubclass(type(i), AbstractPrimitive))
    print(isinstance(i, AbstractPrimitive))


# Batch Normalization from nasbench
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x, *args, **kwargs):
        return self.op(x)


class ConvBnRelu(nn.Module):
    """
    Equivalent to conv_bn_relu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L32
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding=1, affine=True):
        super(ConvBnRelu, self).__init__()
        self.op = nn.Sequential(
            # Padding = 1 is for a 3x3 kernel equivalent to tensorflow padding
            # = same
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False),
            # affine is equivalent to scale in original tensorflow code
            nn.BatchNorm2d(C_out, affine=affine, momentum=BN_MOMENTUM,
                           eps=BN_EPSILON),
            nn.ReLU(inplace=False)
        )

    def forward(self, x, *args, **kwargs):
        return self.op(x)











class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super(NoiseOp, self).__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x, *args, **kwargs):
        if self.stride != 1:
            x_new = x[:, :, ::self.stride, ::self.stride]
        else:
            x_new = x
        noise = Variable(x_new.data.new(x_new.size()).normal_(self.mean, self.std))
        return noise
