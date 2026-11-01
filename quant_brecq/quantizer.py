import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from copy import deepcopy


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 16, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.channel_wise = channel_wise
        self.is_act = is_act
        self.register_buffer('inited', torch.zeros(1))
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={}, bit = {})".format(self.inited, self.channel_wise, self.n_bits)
        return s

    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
            self.m = Parameter(torch.Tensor([0.]).cuda())
            self.t = deepcopy(self.delta)
            self.inited.fill_(1)

        # start quantization
        x_int = round_ste(x / (self.delta)) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * (self.delta)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if self.is_act:
                if len(x.shape) == 3:
                    n_channels = x_clone.shape[-1]
                elif len(x.shape) == 4: 
                    n_channels = x_clone.shape[1] 
                elif len(x.shape) == 2: 
                    n_channels = x_clone.shape[1]
                else:
                    raise NotImplementedError

                if len(x.shape) == 4: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=0)[0]
                elif len(x.shape) == 3: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,...], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(1, -1, 1, 1)
                    zero_point = zero_point.view(1, -1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(1, -1)
                    zero_point = zero_point.view(1, -1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
                
            else: 
                n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)

                if len(x.shape) == 4:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.view(-1, 1)
                    zero_point = zero_point.view(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.view(1, 1, -1)
                    zero_point = zero_point.view(1, 1, -1)
                else:
                    raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            if self.is_act:
                search_range = [0.999, 0.9999, 0.99999]
            else:
                search_range = [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999]
            for pct in search_range:
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
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class ShiftUniformQuantizer2(nn.Module):
    def __init__(self, n_bits: int=8, channel_wise:bool=False, is_act: bool=False) -> None:
        super(ShiftUniformQuantizer2, self).__init__()
        self.low_org = 2e-4 # 1e-3
        self.mid = 0.2
        self.n_levels = 2 ** n_bits
        self.register_buffer('inited', torch.zeros(1))
    
    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            mid = self.init_quantization_scale(x)
            mid = mid.cuda()
            low = torch.tensor(5e-4).cuda()
            self.mid = Parameter(mid).contiguous()
            self.low = Parameter(low).contiguous()
            self.inited.fill_(1)
        if self.mid > self.low:        
            delta_low = (self.mid - self.low) / self.n_levels
        else:
            delta_low = (self.mid - self.low_org) / self.n_levels
            self.low = Parameter(torch.tensor(self.low_org))

        x_int_low = round_ste(x / delta_low) + self.low
        x_quant_low = torch.clamp(x_int_low, 0, self.n_levels - 1)
        x_dequant_low = (x_quant_low - self.low) * delta_low
        return x_dequant_low
    
    def init_quantization_scale(self, x:torch.Tensor):
        x_clone = x.clone().detach()
        cur_mid = -10
        best_score = 1e+10
        for mid in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.12, 0.2, 0.5, 1.0]:
            x_q = self.quantize(x_clone, mid)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                cur_mid = mid
        return torch.tensor(cur_mid)
    
    def quantize(self, x, mid):
        delta_low = (mid - self.low_org) / self.n_levels
        x_int_low = round_ste(x / delta_low) + self.low_org
        x_quant_low = torch.clamp(x_int_low, 0, self.n_levels - 1)
        x_dequant_low = (x_quant_low - self.low_org) * delta_low
        return x_dequant_low
    
class DFQ(nn.Module):
    def __init__(self, n_bits: int=8, channel_wise:bool=False, is_act: bool=False) -> None:
        super(DFQ, self).__init__()
        self.low_org = 2e-4
        self.mid = 0.2
        self.n_levels = 2 ** n_bits
        self.register_buffer('inited', torch.zeros(1))
        self.ema_alpha = 0.99
        self.slid_step = 1e-5
        self.is_train = True
    
    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            mid = self.init_quantization_scale(x)
            low = torch.tensor(5e-4) # 5e-4 for vit-b (3/3, 4/4), 1e-3 for others
            self.mid = nn.Parameter(mid).contiguous()
            self.low = nn.Parameter(low)
            self.ema_low = deepcopy(self.low)
            self.ema_mid = deepcopy(self.mid)
            self.inited.fill_(1)
        
        # data_low, data_mid = self.low.data, self.mid.data
        # self.low.data = (1 - self.ema_alpha) * self.ema_low.data + self.ema_alpha *  self.low.data
        # self.mid.data = (1 - self.ema_alpha) * self.ema_mid.data + self.ema_alpha *  self.mid.data
        # self.ema_low.data, self.ema_mid.data = data_low, data_mid
        
        # sliding region quantizer

        if self.mid > self.low and self.low > 0:        
            delta_low = (self.mid - self.low) / self.n_levels
        else:
            delta_low = (self.mid - self.low_org) / self.n_levels
            self.low = nn.Parameter(torch.tensor(self.low_org))

        # x_int_low = round_ste(x / delta_low) + self.low
        x_int_low = round_ste((x - self.low) / delta_low)
        x_int_low_pre = round_ste(x / delta_low) + self.low - self.slid_step
        x_int_low_next = round_ste(x / delta_low) + self.low + self.slid_step
        x_quant_low = torch.clamp(x_int_low, 0, self.n_levels - 1)
        x_quant_low_pre = torch.clamp(x_int_low_pre, 0, self.n_levels - 1)
        x_quant_low_next = torch.clamp(x_int_low_next, 0, self.n_levels - 1)
        
        x_dequant_low = (x_quant_low - self.low) * delta_low
        x_dequant_pre = (x_quant_low_pre - self.low + self.slid_step) * delta_low
        x_dequant_next = (x_quant_low_next - self.low - self.slid_step) * delta_low
        
        # return x
    
        if self.is_train:
            # return (x_dequant_low + x_dequant_pre + x_dequant_next) / 3.
            return x_dequant_low
        else:
            return x_dequant_low
    
    def init_quantization_scale(self, x:torch.Tensor):
        x_clone = x.clone().detach()
        cur_mid = -10
        best_score = 1e+10
        for mid in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.12, 0.2, 0.5, 1.0]:
            x_q = self.quantize(x_clone, mid)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                cur_mid = mid
        return torch.tensor(cur_mid)
    
    def quantize(self, x, mid):
        delta_low = (mid - self.low_org) / self.n_levels
        x_int_low = round_ste(x / delta_low) + self.low_org
        x_quant_low = torch.clamp(x_int_low, 0, self.n_levels - 1)
        x_dequant_low = (x_quant_low - self.low_org) * delta_low
        return x_dequant_low






