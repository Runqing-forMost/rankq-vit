from logging import raiseExceptions
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
from .quantizer import UniformQuantizer,DFQ, ShiftUniformQuantizer2


class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    """
    def __init__(self,   
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                input_quant_params={},
                weight_quant_params={}):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        # no need to quantize input image since image is already 8-bit
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_weight_quant = False
        self.ignore_reconstruction = True

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(False,self.use_weight_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_weight_quant = weight_quant

    def forward(self, x):
        """
        using quantized weights to forward input x
        """
        # no need to quantize input image since image is already 8-bit

        if self.use_weight_quant:
            w = self.weight_quantizer(self.weight)
        else:
            w = self.weight

        out = F.conv2d(
            x, 
            w, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

        return out


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 in_features,
                 out_features,
                 input_quant_params={},
                 weight_quant_params={},
                 use_adapter=False):
        super(QuantLinear, self).__init__(in_features, out_features)

        input_quant_params_linear = deepcopy(input_quant_params)
        input_quant_params_linear['is_act'] = True
        self.input_quantizer = UniformQuantizer(**input_quant_params_linear)

        self.weight_quantizer = UniformQuantizer(**weight_quant_params)

        self.use_input_quant = False
        self.use_weight_quant = False
        self.ignore_reconstruction = True

        #adapter
        r = 100
        self.use_adapter = use_adapter
        self.loraA = nn.Parameter(torch.rand(in_features, r))
        self.loraB = nn.Parameter(torch.zeros(r, out_features))
        self.lora_bias = nn.Parameter(torch.zeros(out_features))

        self.eye = torch.eye(in_features)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.min_dim = min([in_features, out_features])
        self.lora_diag = nn.Parameter(torch.zeros(self.min_dim))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + "input_quant={}, weight_quant={})".format(self.use_input_quant, self.use_weight_quant)
        return s
    

    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant
        self.use_weight_quant = weight_quant

    def weight_quantizer_refactor(self, n_bits):
        self.weight_quantizer.n_bits = n_bits
        self.weight_quantizer.n_levels = 2 ** n_bits
        # self.weight_quantizer.inited.fill_(0)

    def forward(self, x):
        """
        using quantized weights to forward input x
        """
        if self.use_input_quant:
            x = self.input_quantizer(x)

        if self.use_adapter:
            out = F.linear(x, weight=self.weight, bias=self.bias)
            weight_lora = torch.mm(self.loraA, self.loraB).T
            if self.use_weight_quant:
                weight_lora += self.weight
                weight_lora = self.weight_quantizer(weight_lora)
                out = F.linear(x, weight_lora, bias=self.bias)
                # w = self.weight_quantizer(self.weight)
                # out = F.linear(x, w, bias=self.bias)
            else:
                out_adp = F.linear(x, weight_lora, bias=self.lora_bias)
                out += out_adp
                # out /= 2
        else:
            if self.use_weight_quant:
                w = self.weight_quantizer(self.weight)
                out = F.linear(x, w, bias=self.bias)
            else:
                out = F.linear(x, weight=self.weight, bias=self.bias)
        return out 
   

class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self,
                 input_quant_params={}):
        super(QuantMatMul, self).__init__()
        input_quant_params_matmul = deepcopy(input_quant_params)
        input_quant_params_matmul['is_act'] = True
        if 'log_quant' in input_quant_params_matmul:
            input_quant_params_matmul.pop('log_quant')            
            self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)
            # self.quantizer_A = ShiftUniformQuantizer2(**input_quant_params_matmul)
        else:
            self.quantizer_A = UniformQuantizer(**input_quant_params_matmul)    
        self.quantizer_B = UniformQuantizer(**input_quant_params_matmul)
        self.use_input_quant = False

    def __repr__(self):
        s = super(QuantMatMul, self).__repr__()
        s = "(" + s + "input_quant={})".format(self.use_input_quant)
        return s
    
    def set_quant_state(self, input_quant=False, weight_quant=False):
        self.use_input_quant = input_quant

    def forward(self, A, B):
        if self.use_input_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        
        out = A @ B
        return out