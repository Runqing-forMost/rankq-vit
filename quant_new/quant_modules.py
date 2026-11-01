import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter

from quant_new.quant_utils import *


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()
    
    

class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    """
    def __init__(self,
                weight_bit,       
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.weight_bit = weight_bit
        self.quant = False
        self.weight_function = SymmetricQuantFunction.apply

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={})".format(self.weight_bit)
        return s

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        v = self.weight
        v = v.reshape(v.shape[0], -1)
        v_max = v.max(axis=1).values
        v_min = v.min(axis=1).values
        w = self.weight_function(self.weight, self.weight_bit, v_min, v_max)

        return F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 weight_bit,
                 in_features,
                 out_features,
                 bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.quant = False
        self.weight_function = SymmetricQuantFunction.apply

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={})".format(self.weight_bit)
        return s

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        if not self.quant:
            return F.linear(
                x, 
                self.weight, 
                self.bias
            )

        v = self.weight
        v = v.reshape(v.shape[0], -1)
        v_max = v.max(axis=1).values
        v_min = v.min(axis=1).values
        w = self.weight_function(self.weight, self.weight_bit, v_min, v_max)

        return F.linear(
            x, 
            weight=w, 
            bias=self.bias
        )


class QuantAct(nn.Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 running_stat=True):
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.n_levels = 2 ** self.activation_bit
        self.running_stat = running_stat
        self.quant = False
        self.act_function = AsymmetricQuantFunction.apply

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

    def __repr__(self):
        return "{0}(activation_bit={1}, running_stat={2}, Act_min: {3:.2f}, Act_max: {4:.2f})".format(
            self.__class__.__name__, self.activation_bit, self.running_stat, 
            self.x_min.item(), self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    
    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.activation_bit - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    
    
    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            # cur_max =  x.data.max()
            # cur_min =  x.data.min()
            # if self.x_max == 0:
            #     self.x_max = cur_max
            #     self.x_min = cur_min
            # else:
            #     self.x_max = torch.max(cur_max, self.x_max)
            #     self.x_min = torch.min(cur_min, self.x_min)
            
            # Add pct for scale selection
            x_clone = x.clone().detach()
            self.x_max = x_clone.max()
            self.x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                    
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    self.x_min = new_min
                    self.x_max = new_max
        if not self.quant:
            return x

        quant_act = self.act_function(x, self.activation_bit, self.x_min, self.x_max)

        return quant_act
