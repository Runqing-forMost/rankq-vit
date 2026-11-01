"""CPU regression checks: python -m unittest discover -s tests -v."""

import ast
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

import rankq
from utils.build_model import MatMul


def config(**overrides):
    values = dict(model='deit_tiny', batch_size=2, memory_bank=3, target_size=3,
                  iter=1, print_freq=100, val_batchsize=2, calib_batchsize=2,
                  num_workers=0, w_bits=4, a_bits=4, w_cw=True, a_cw=False,
                  warmup=0.2, gen_epochs=1, gen_iterations=52)
    values.update(overrides)
    return SimpleNamespace(**values)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.matmul2 = MatMul()

    def forward(self, x):
        identity = torch.eye(x.size(-1), device=x.device)
        return self.attn.matmul2(x, identity)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 3, 1)
        self.blocks = nn.ModuleList([TinyBlock()])
        self.head = nn.Linear(3, 1000)
        self.batch_sizes = []
        self.fail = False

    def forward(self, images):
        self.batch_sizes.append(len(images))
        x = F.adaptive_avg_pool2d(images, (2, 2)).flatten(2).transpose(1, 2).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        if self.fail:
            raise RuntimeError('injected failure')
        return self.head(x.mean(dim=(1, 2)))


class RankQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_infonce_value_and_input_gradient_match_original(self):
        original = Path(rankq.__file__).with_name('main_train_v2.py')
        tree = ast.parse(original.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'contrastive_patch_loss_infoNCE')
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), 'exec'), namespace)
        for shape in ((1, 3, 4), (2, 7, 5)):
            before = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
            after = before.detach().clone().requires_grad_()
            old_loss = namespace[node.name](before, temperature=0.3)
            new_loss = rankq.contrastive_patch_loss_infoNCE(after, temperature=0.3)
            torch.testing.assert_close(new_loss, old_loss)
            old_loss.backward()
            new_loss.backward()
            torch.testing.assert_close(after.grad, before.grad)

    def test_generation_tail_batch_and_teacher_state(self):
        teacher = TinyTeacher().train()
        teacher.head.bias.requires_grad_(False)
        flags = [p.requires_grad for p in teacher.parameters()]
        images, labels = rankq.MainExp(config()).data_generation_v3(teacher)
        self.assertEqual(images.shape, (3, 3, 224, 224))
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(teacher.batch_sizes, [2] * 52 + [1] * 52)
        self.assertEqual(images.device.type, 'cpu')
        self.assertFalse(images.requires_grad)
        self.assertTrue(teacher.training)
        self.assertEqual(flags, [p.requires_grad for p in teacher.parameters()])
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertFalse(teacher.blocks[0].attn.matmul2._forward_hooks)

    def test_generation_failure_cleans_hooks_and_restores_flags(self):
        teacher = TinyTeacher().train()
        teacher.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            rankq.MainExp(config()).data_generation_v3(teacher)
        self.assertTrue(teacher.training)
        self.assertTrue(all(p.requires_grad for p in teacher.parameters()))
        self.assertFalse(teacher.blocks[0].attn.matmul2._forward_hooks)

    def test_cache_hit_skips_generation_and_incomplete_pair_regenerates(self):
        with TemporaryDirectory() as directory:
            experiment = rankq.MainExp(config(cache_dir=directory))
            expected = (torch.randn(3, 3, 224, 224), torch.tensor([1, 2, 3]))
            with patch.object(experiment, 'data_generation_v3', return_value=expected) as generate:
                experiment.load_or_generate(None)
                actual = experiment.load_or_generate(None)
                self.assertEqual(generate.call_count, 1)
                for before, after in zip(expected, actual):
                    torch.testing.assert_close(before, after)
                (Path(directory) / 'label_gen_deit_tiny_3.npy').unlink()
                experiment.load_or_generate(None)
                self.assertEqual(generate.call_count, 2)

    def test_legacy_selection_equivalence_and_no_grad_or_hook_leak(self):
        source = Path(rankq.__file__).with_name('sim_con_loss.py')
        names = {'AttentionMap', 'label_smoothing', 'js_divergence', 'jsd_selection'}
        nodes = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        namespace = dict(torch=torch, nn=nn, F=F)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        teacher = TinyTeacher().eval()
        data, labels = torch.randn(5, 3, 8, 8), torch.arange(5)
        baseline = deepcopy(teacher)
        with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            expected = namespace['jsd_selection'](baseline, data, labels, k=3)
        actual = rankq.jsd_selection(teacher, data, labels, k=3)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(teacher.blocks[0].attn.matmul2._forward_hooks)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertIs(rankq.jsd_selection(teacher, data, labels, k=5), data)
        teacher.train()
        teacher.fail = True
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            rankq.jsd_selection(teacher, data, labels, k=3)
        self.assertTrue(teacher.training)
        self.assertFalse(teacher.blocks[0].attn.matmul2._forward_hooks)

    def test_generated_accuracy_uses_nonoverlapping_weighted_batches(self):
        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.batch_sizes = []

            def forward(self, data):
                self.batch_sizes.append(len(data))
                return F.one_hot(data.long(), num_classes=6).float() * self.weight

        teacher = Classifier().train()
        experiment = rankq.MainExp(config(eval_batch_size=2))
        result = experiment.test_gen_image_acc(teacher, torch.tensor([0, 1, 2]),
                                               torch.tensor([0, 1, 3]))
        self.assertAlmostEqual(result, 200 / 3, places=5)
        self.assertEqual(teacher.batch_sizes, [2, 1])
        self.assertTrue(teacher.training)

    def test_reconstruction_traversal_and_all_samples_used(self):
        experiment = rankq.MainExp(config())
        teacher = TinyTeacher()
        experiment.q_model = rankq.quant_model(deepcopy(teacher))
        calls = []

        def record(*args, **kwargs):
            calls.append(kwargs['batch_size'])
            self.assertEqual(len(kwargs['cali_data']) % kwargs['batch_size'], 0)

        with patch.object(experiment, 'test'), patch.object(rankq, 'layer_reconstruction', record), \
                patch.object(rankq, 'block_reconstruction', record):
            experiment.block_wise_reconstruction(torch.randn(25, 3, 4, 4), fp_model=teacher)
        self.assertEqual(calls, [5, 5, 5])

    def test_config_rejects_invalid_counts_and_unsupported_swin_selection(self):
        for overrides in (dict(batch_size=0), dict(target_size=4), dict(memory_bank=-1),
                          dict(gen_warmup=52), dict(model='swin_tiny', target_size=2)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                rankq.MainExp(config(**overrides))

    def test_validation_requires_only_val_directory_and_reuses_loader(self):
        from PIL import Image

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, images):
                return self.weight * torch.arange(6).float().expand(len(images), -1)

        with TemporaryDirectory() as directory:
            for label in range(6):
                folder = Path(directory) / 'val' / str(label)
                folder.mkdir(parents=True)
                Image.new('RGB', (256, 256)).save(folder / 'image.png')
            experiment = rankq.MainExp(config(dataset=directory))
            loader = experiment.get_val_loader()
            self.assertIs(loader, experiment.get_val_loader())
            teacher = Classifier().train()
            result = experiment.test(teacher)
            self.assertAlmostEqual(result['top1'], 100 / 6, places=5)
            self.assertAlmostEqual(result['top5'], 500 / 6, places=5)
            self.assertTrue(teacher.training)


if __name__ == '__main__':
    unittest.main()
