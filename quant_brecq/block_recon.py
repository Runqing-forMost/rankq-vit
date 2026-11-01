import torch
import os
from quant_brecq.data_utils import save_grad_data, save_inp_oup_data, save_inp_oup_data_hengshuang
from quant_brecq.quant_modules import *
from quant_brecq.quantizer import lp_loss
from quant_brecq.quant_model import set_quant_state

def block_reconstruction_pct(model, block, teacher_model, T_module, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, weight_quant: bool=False, lr: float = 3e-4, p: float = 2.0,
                         last_stage: bool = False):
    """
    Block reconstruction to optimize the output from each block.
    """
    set_quant_state(block, input_quant=act_quant, weight_quant=weight_quant)
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.weight.requires_grad = False
            module.bias.requires_grad = False
        if isinstance(module, UniformQuantizer):
            if hasattr(module, 'zero_point') and module.zero_point is not None:
                module.zero_point.requires_grad = False
            if hasattr(module, 'delta') and module.zero_point is not None:
                module.delta.requires_grad = False
        # if isinstance(module, QuantLinear):
        #     module.weight.requires_grad = False
            # module.bias.requires_grad = False
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, block.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    loss_mode = 'none'
    rec_loss = opt_mode
    loss_func = SelectLoss(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, use_grad=True)
    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data_hengshuang(model, block, teacher_model, T_module,
                                                 cali_data, asym, act_quant, weight_quant, batch_size, last_stage = last_stage)
    
    cached_inps_xyz, cached_inps_feat = cached_inps
    
    # 获取block的输出梯度
    # cached_grads = save_grad_data(model, block, teacher_model, cali_data, True, batch_size=batch_size)
    device = 'cuda'
    block.train()
    for i in range(iters):
        idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
        cur_inp_xyz = cached_inps_xyz[idx].to(device)
        cur_inp_feat = cached_inps_feat[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        # cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None
        optimizer.zero_grad()
        out_quant, _ = block(cur_inp_xyz, cur_inp_feat)
        err = loss_func(out_quant, cur_out)
        err.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
    block.eval()
    torch.cuda.empty_cache()

    
def block_reconstruction(model, block, teacher_model, T_module, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, weight_quant: bool=False, lr: float = 3e-4, p: float = 2.0,
                         last_stage: bool = False):
    """
    Block reconstruction to optimize the output from each block.
    """
    set_quant_state(block, input_quant=act_quant, weight_quant=weight_quant)
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.weight.requires_grad = False
            module.bias.requires_grad = False
        if isinstance(module, UniformQuantizer):
            if hasattr(module, 'delta') and module.delta is not None:
                module.delta.requires_grad = True
            if hasattr(module, 'zero_point') and module.zero_point is not None:
                module.zero_point.requires_grad = False
        if isinstance(module, (QuantLinear, QuantConv2d)):
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, block.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    loss_mode = 'none'
    rec_loss = opt_mode
    loss_func = SelectLoss(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p, use_grad=True)
    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(model, block, teacher_model, T_module,
                                                 cali_data, asym, act_quant, weight_quant, batch_size, last_stage = last_stage)
    
    # 获取block的输出梯度
    cached_grads = save_grad_data(model, block, teacher_model, cali_data, True, batch_size=batch_size)
    device = 'cuda'
    block.train()
    for i in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None
        optimizer.zero_grad()
        out_quant = block(cur_inp)
        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step()
    block.eval()
    torch.cuda.empty_cache()


class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2., 
                 use_grad: bool=False):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        
        self.use_grad = use_grad

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        total_loss = rec_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} \tcount={}'.format(
                  float(total_loss), self.count))
        return total_loss

class SelectLoss:
    """
    update model with samples with small-loss
    """
    def __init__(self, block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2., 
                 use_grad: bool=False,
                 ) -> None:
        
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0       
        self.use_grad = use_grad
        self.mse_none = torch.nn.MSELoss(reduction='none')
        self.drop_rate = 0.1
        self.rate_schedule = np.ones(max_count+1)*self.drop_rate
        self.rate_schedule[:10] = np.linspace(0, self.drop_rate, 10)
        

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            # rec_loss = lp_loss(pred, tgt, p=self.p)
            # rec_loss = self.mse_none(pred, tgt)
            rec_loss = self._batch_selection(pred, tgt)
            
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        total_loss = rec_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} \tcount={}'.format(
                  float(total_loss), self.count))
        return total_loss

    def _batch_selection(self, pred:torch.tensor, tgt:torch.tensor):
        """
        used for selecting samples with small-loss
        """
        loss_batch = torch.mean((pred - tgt).abs().pow(2.).sum(1), -1)
        n = len(loss_batch)
        # n_keep = int((1 - self.drop_rate)*n)
        n_keep = int((1 - self.rate_schedule[self.count])*n)
        # batch_loss_summed = torch.sum(loss_batch, dim=-1)
        sorted_batch, sorted_idx = torch.sort(loss_batch)
        batch_keep_idx = sorted_idx[:n_keep]
        loss_batch = loss_batch[batch_keep_idx]
        return loss_batch.mean()
    
class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))



