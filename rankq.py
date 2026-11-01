import argparse
from copy import deepcopy
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

from quant_brecq.block_recon import block_reconstruction
from quant_brecq.layer_recon import layer_reconstruction
from quant_brecq.quant_model import quant_model, set_quant_state
from quant_brecq.quant_modules import QuantConv2d, QuantLinear, QuantMatMul
from utils.build_model import build_model
from utils.data_utils import build_transform
from utils.meter import AverageMeter, accuracy
from utils.psaq_func import clip, lr_cosine_policy
from utils.schedule import prior_loss_schedule


MODEL_ZOO = {
    'deit_tiny': 'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base': 'deit_base_patch16_224',
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'vit_small': 'vit_small_patch16_224',
    'vit_base': 'vit_base_patch16_224',
}


class AttentionMap:
    def __init__(self, module):
        self.feature = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, output):
        self.feature = output

    def remove(self):
        self.hook.remove()
        self.feature = None


def attention_blocks(model):
    if hasattr(model, 'blocks'):
        return list(model.blocks)
    return [block for layer in model.layers for block in layer.blocks]


def contrastive_patch_loss_infoNCE(patch_embeddings, temperature=0.1):
    embeddings = F.normalize(patch_embeddings, p=2, dim=-1)
    similarities = torch.bmm(embeddings, embeddings.transpose(1, 2))
    return -F.log_softmax(similarities / temperature, dim=-1).diagonal(
        dim1=-2, dim2=-1
    ).mean()


@torch.no_grad()
def jsd_selection(model, data, label, k=32):
    """Preserve sim_con_loss.jsd_selection scoring and per-image rank tolerance.

    In particular, the legacy smoothed-label/log-softmax KL formula is retained
    for experiment comparability; this is not a redefinition of standard JSD.
    """
    if not 1 <= k <= len(data) or len(label) != len(data):
        raise ValueError('Selection requires 1 <= k <= sample count and aligned labels.')
    if k == len(data):
        return data
    if not hasattr(model, 'blocks'):
        raise ValueError('Swin subset selection is unsupported; set target_size=memory_bank.')
    hooks = []
    scores_jsd, ranks = torch.zeros(len(data)), torch.zeros(len(data))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for block in model.blocks:
            hooks.append(AttentionMap(block.attn.matmul2))
        for i in tqdm(range(len(data)), desc='JSD/rank selection'):
            output = model(data[i:i + 1].to(device)).cpu()
            target = label[i:i + 1].cpu()
            if target.ndim == 1:
                target = F.one_hot(target.long(), num_classes=output.size(1)).float()
            target = 0.9 * target + 0.1 / target.size(1)
            mixture = 0.5 * (target + F.softmax(output, dim=1))
            scores_jsd[i] = 0.5 * (
                F.kl_div(F.log_softmax(target, dim=1), mixture, reduction='batchmean')
                + F.kl_div(F.log_softmax(output, dim=1), mixture, reduction='batchmean')
            )
            rank_sum = torch.zeros((), device=device)
            for hook in hooks:
                patches = hook.feature.mean(dim=1)[:, 1:, :]
                similarities = F.cosine_similarity(
                    patches.unsqueeze(1), patches.unsqueeze(2), dim=3
                )
                rank_sum += torch.linalg.matrix_rank(similarities[0])
                hook.feature = None
            ranks[i] = (rank_sum / len(hooks)).cpu()
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    normalized_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-8)
    indices = torch.topk(-scores_jsd + 2 * normalized_ranks, k).indices
    return data[indices.to(data.device)]


class MainExp:
    def __init__(self, args):
        self.args = args
        self.val_loader = None
        self.validate_config()

    def validate_config(self):
        args = self.args
        if args.model not in MODEL_ZOO:
            raise ValueError(f'Unsupported model {args.model!r}; choose from {list(MODEL_ZOO)}')
        for name in ('batch_size', 'memory_bank', 'target_size', 'iter', 'print_freq',
                     'val_batchsize', 'calib_batchsize'):
            value = getattr(args, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        for name, default in (('gen_epochs', 2), ('gen_iterations', 500),
                              ('recon_batch_size', 24), ('eval_batch_size', 32)):
            value = getattr(args, name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        iterations = getattr(args, 'gen_iterations', 500)
        warmup = getattr(args, 'gen_warmup', min(100, iterations - 1))
        if not isinstance(warmup, int) or not 0 <= warmup < iterations:
            raise ValueError('gen_warmup must be an integer in [0, gen_iterations).')
        if not 1 <= args.target_size <= args.memory_bank:
            raise ValueError('target_size must be between 1 and memory_bank.')
        if args.model.startswith('swin') and args.target_size != args.memory_bank:
            raise ValueError('Swin subset selection is unsupported; set target_size=memory_bank.')
        if args.num_workers < 0:
            raise ValueError('num_workers must be nonnegative.')

    def run(self):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required by the existing quantization utilities.')
        seed = getattr(self.args, 'seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        model = build_model(MODEL_ZOO[self.args.model], Pretrained=True)
        print(f'Start quantizing {self.args.model}, w: {self.args.w_bits}-bit, '
              f'a: {self.args.a_bits}-bit')
        data, labels = self.load_or_generate(model)
        data = jsd_selection(model, data, labels, k=self.args.target_size)
        return self.block_wise_reconstruction(data, fp_model=model)

    def load_or_generate(self, model):
        cache_dir = Path(getattr(self.args, 'cache_dir', './generated_imgs/my_generated'))
        suffix = f'{self.args.model}_{self.args.memory_bank}.npy'
        image_path, label_path = cache_dir / f'gen_{suffix}', cache_dir / f'label_gen_{suffix}'
        if not getattr(self.args, 'regenerate', False) and image_path.is_file() and label_path.is_file():
            try:
                data = np.load(image_path, allow_pickle=False)
                labels = np.load(label_path, allow_pickle=False)
                if data.shape != (self.args.memory_bank, 3, 224, 224):
                    raise ValueError(f'Unexpected cached image shape: {data.shape}')
                if labels.shape != (len(data),) or not np.issubdtype(labels.dtype, np.integer):
                    raise ValueError('Cached labels must be a one-dimensional integer array.')
                if not np.issubdtype(data.dtype, np.floating):
                    raise ValueError('Cached images must have a floating-point dtype.')
                print(f'Using cached samples: {image_path}')
                return torch.from_numpy(data).float(), torch.from_numpy(labels).long()
            except (OSError, ValueError, EOFError) as error:
                print(f'Invalid cache; regenerating samples: {error}')
        data, labels = self.data_generation_v3(model)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Invalidate the pair first so an interrupted write cannot mix old labels
        # with new images. The next run will regenerate any incomplete pair.
        label_path.unlink(missing_ok=True)
        np.save(image_path, data.cpu().numpy(), allow_pickle=False)
        np.save(label_path, labels.cpu().numpy(), allow_pickle=False)
        return data, labels

    def data_generation_v3(self, model):
        args = self.args
        device = next(model.parameters()).device
        lr = float(getattr(args, 'gen_lr', 0.25 if args.model.startswith('swin') else 0.20))
        epochs = getattr(args, 'gen_epochs', 2)
        iterations = getattr(args, 'gen_iterations', 500)
        warmup = getattr(args, 'gen_warmup', min(100, iterations - 1))
        images, labels, hooks = [], [], []
        parameters = list(model.parameters())
        grad_flags = [parameter.requires_grad for parameter in parameters]
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            for block in attention_blocks(model):
                hooks.append(AttentionMap(block.attn.matmul2))
            for start in range(0, args.memory_bank, args.batch_size):
                batch_size = min(args.batch_size, args.memory_bank - start)
                image = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
                optimizer = optim.Adam([image], lr=lr, betas=(0.5, 0.9), eps=1e-8)
                targets = torch.tensor([random.randint(0, 999) for _ in range(batch_size)],
                                       device=device, dtype=torch.long)
                for epoch in range(epochs):
                    limit = 15 if epoch == 0 else 30
                    schedule = lr_cosine_policy(lr, warmup, iterations)
                    for iteration in tqdm(range(iterations), desc=f'Generate {start // args.batch_size + 1}, '
                                          f'epoch {epoch + 1}/{epochs}'):
                        schedule(optimizer, iteration, iteration)
                        offset = random.randint(-limit, limit)
                        jittered = torch.roll(image, shifts=(offset, offset), dims=(2, 3))
                        if random.random() > 0.5:
                            jittered = torch.flip(jittered, dims=(3,))
                        optimizer.zero_grad(set_to_none=True)
                        output = model(jittered)
                        loss = F.cross_entropy(output, targets)
                        if iteration > 50:
                            contrastive = 0
                            for index, hook in enumerate(hooks, start=1):
                                patches = hook.feature.mean(dim=1)[:, 1:, :]
                                contrastive = contrastive + len(hooks) / index * (
                                    contrastive_patch_loss_infoNCE(patches, temperature=index / 10.)
                                )
                            loss = loss + 0.05 * prior_loss_schedule(iteration) * contrastive
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            clip(image)
                        for hook in hooks:
                            hook.feature = None
                images.append(image.detach().cpu())
                labels.append(targets.cpu())
        finally:
            for hook in hooks:
                hook.remove()
            for parameter, requires_grad in zip(parameters, grad_flags):
                parameter.requires_grad_(requires_grad)
            model.train(was_training)
        return torch.cat(images), torch.cat(labels)

    @torch.no_grad()
    def test_gen_image_acc(self, model, data, label):
        if len(data) == 0 or len(data) != len(label):
            raise ValueError('Evaluation requires nonempty data and aligned labels.')
        batch_size = getattr(self.args, 'eval_batch_size', 32)
        device = next(model.parameters()).device
        top1 = AverageMeter()
        was_training = model.training
        model.eval()
        try:
            for start in range(0, len(data), batch_size):
                images = data[start:start + batch_size].to(device)
                targets = label[start:start + batch_size].to(device)
                output = model(images)
                precision, = accuracy(output, targets, topk=(1,))
                top1.update(precision.item(), len(images))
        finally:
            model.train(was_training)
        print(f'acc for generated images is {top1.avg:.2f}')
        return top1.avg

    def block_wise_reconstruction(self, calib_data, fp_model=None):
        if len(calib_data) == 0:
            raise ValueError('Calibration data must not be empty.')
        if fp_model is None:
            fp_model = build_model(MODEL_ZOO[self.args.model], Pretrained=True)
        fp_model.eval()
        q_model = getattr(self, 'q_model', None)
        if q_model is None:
            q_model = quant_model(
                deepcopy(fp_model),
                input_quant_params={'n_bits': self.args.a_bits, 'channel_wise': self.args.a_cw},
                weight_quant_params={'n_bits': self.args.w_bits, 'channel_wise': self.args.w_cw},
            ).cuda().eval()
            set_quant_state(q_model, input_quant=True, weight_quant=True)
            print('Start Calibration...')
            with torch.no_grad():
                q_model(calib_data[:32].cuda())
            print('Calibration done.')
        self.test(q_model)
        # Shared reconstruction helpers use floor division and drop remainders.
        # Select a divisor, including for calibration sets smaller than 24.
        requested = min(getattr(self.args, 'recon_batch_size', 24), len(calib_data))
        batch_size = next(size for size in range(requested, 0, -1) if len(calib_data) % size == 0)
        print(f'Reconstruction batch size: {batch_size} ({len(calib_data)} samples)')
        kwargs = dict(cali_data=calib_data, asym=True, warmup=self.args.warmup,
                      act_quant=True, weight_quant=True, opt_mode='mse',
                      batch_size=batch_size, iters=self.args.iter,
                      lr=float(getattr(self.args, 'recon_lr', 4e-5)))
        q_model.patch_embed.proj.ignore_reconstruction = False
        q_model.head.ignore_reconstruction = False
        block_type = type(attention_blocks(q_model)[0])

        def reconstruct(student, teacher):
            teacher_children = dict(teacher.named_children())
            student_children = dict(student.named_children())
            if student_children.keys() != teacher_children.keys():
                raise RuntimeError('Teacher/student module structure mismatch.')
            for name, module in student_children.items():
                teacher_module = teacher_children[name]
                if isinstance(module, (QuantConv2d, QuantLinear)):
                    if module.ignore_reconstruction:
                        print(f'Ignore reconstruction of layer {name}')
                    else:
                        print(f'Reconstruction for layer {name}')
                        layer_reconstruction(q_model, module, fp_model, teacher_module, **kwargs)
                elif isinstance(module, block_type):
                    print(f'Reconstruction for block {name}')
                    block_reconstruction(q_model, module, fp_model, teacher_module, **kwargs)
                elif isinstance(module, QuantMatMul):
                    continue
                else:
                    reconstruct(module, teacher_module)

        reconstruct(q_model, fp_model)
        self.test(q_model)
        self.q_model = q_model
        return q_model

    def get_val_loader(self):
        if self.val_loader is None:
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader
            family = self.args.model.split('_')[0]
            mean = (0.5, 0.5, 0.5) if family == 'vit' else (0.485, 0.456, 0.406)
            std = (0.5, 0.5, 0.5) if family == 'vit' else (0.229, 0.224, 0.225)
            transform = build_transform(mean=mean, std=std, crop_pct=0.875 if family == 'deit' else 0.9)
            dataset = ImageFolder(str(Path(self.args.dataset) / 'val'), transform)
            self.val_loader = DataLoader(dataset, batch_size=self.args.val_batchsize,
                                         shuffle=False, num_workers=self.args.num_workers,
                                         pin_memory=True)
        return self.val_loader

    @torch.no_grad()
    def test(self, model):
        loader = self.get_val_loader()
        device = next(model.parameters()).device
        losses, top1, top5, batch_time = (AverageMeter() for _ in range(4))
        was_training = model.training
        model.eval()
        start = end = time.time()
        try:
            for index, (data, target) in enumerate(loader):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                loss = F.cross_entropy(output, target)
                precision1, precision5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), len(data))
                top1.update(precision1.item(), len(data))
                top5.update(precision5.item(), len(data))
                batch_time.update(time.time() - end)
                end = time.time()
                if index % self.args.print_freq == 0:
                    print(f'Test: [{index}/{len(loader)}]\t'
                          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                          f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')
        finally:
            model.train(was_training)
        print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time.time() - start:.3f}')
        return {'top1': top1.avg, 'top5': top5.avg}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'configs/imageNet.yaml')
    parser.add_argument('--cache-dir', type=Path, help='Override the synthetic sample cache directory.')
    parser.add_argument('--regenerate', action='store_true', help='Ignore cached synthetic samples.')
    options = parser.parse_args()
    if not options.config.is_file():
        parser.error(f'Config file does not exist: {options.config}')
    config = OmegaConf.load(options.config)
    if options.cache_dir is not None:
        config.cache_dir = str(options.cache_dir)
    if options.regenerate:
        config.regenerate = True
    MainExp(config).run()


if __name__ == '__main__':
    main()
