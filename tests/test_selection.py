"""Controlled CPU checks for joint JSD and mean block rank selection."""

import unittest

import torch
from torch import nn
import torch.nn.functional as F

import rankq
from utils.build_model import MatMul


class SelectionTeacher(nn.Module):
    """Expose exact patch ranks and logits for each candidate sample."""

    def __init__(self, block_ranks, logits):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.blocks = nn.ModuleList()
        for _ in block_ranks[0]:
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.matmul2 = MatMul()
            self.blocks.append(block)
        features = []
        for ranks in block_ranks:
            sample = []
            for rank in ranks:
                patches = torch.eye(3)[torch.arange(3) % rank]
                sample.append(torch.cat((torch.zeros(1, 3), patches)))
            features.append(torch.stack(sample))
        self.register_buffer('features', torch.stack(features))
        self.register_buffer('logits', torch.tensor(logits, dtype=torch.float32))
        self.fail = False

    def forward(self, images):
        index = int(images[0, 0].item())
        for block, features in zip(self.blocks, self.features[index]):
            block.attn.matmul2(features[None, None], torch.eye(3))
        if self.fail:
            raise RuntimeError('injected failure')
        return self.logits[index:index + 1] + self.weight


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = torch.arange(3, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.zeros(3, dtype=torch.long)

    def test_uses_mean_rank_across_all_blocks(self):
        # First-block-only and last-block-only ranking both prefer sample 0.
        # Equal averaging prefers sample 1: 10/4 > 8/4 > 4/4.
        teacher = SelectionTeacher([[3, 1, 1, 3], [2, 3, 3, 2], [1, 1, 1, 1]],
                                   [[0, 0]] * 3)
        selected = rankq.jsd_rank_selection(teacher, self.data, self.labels, k=1)
        torch.testing.assert_close(selected, self.data[1:2])

    def test_lower_jsd_wins_when_mean_ranks_are_equal(self):
        teacher = SelectionTeacher([[2, 2]] * 3, [[0, 8], [1, 0], [0, 0]])
        selected = rankq.jsd_rank_selection(teacher, self.data, self.labels, k=1)
        torch.testing.assert_close(selected, self.data[1:2])

    def test_joint_selection_prefers_low_jsd_and_high_rank(self):
        teacher = SelectionTeacher([[1, 1], [3, 3], [3, 3]],
                                   [[8, 0], [8, 0], [0, 8]])
        for labels in (self.labels, F.one_hot(self.labels, num_classes=2).float()):
            with self.subTest(label_shape=labels.shape):
                selected = rankq.jsd_rank_selection(teacher, self.data, labels, k=1)
                torch.testing.assert_close(selected, self.data[1:2])

    def test_restores_model_state_and_cleans_hooks_on_success_and_failure(self):
        teacher = SelectionTeacher([[1, 2]] * 3, [[0, 0]] * 3)
        for training in (True, False):
            for fail in (False, True):
                with self.subTest(training=training, fail=fail):
                    teacher.train(training)
                    teacher.fail = fail
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                            rankq.jsd_rank_selection(teacher, self.data, self.labels, k=1)
                    else:
                        rankq.jsd_rank_selection(teacher, self.data, self.labels, k=1)
                    self.assertEqual(teacher.training, training)
                    self.assertIsNone(teacher.weight.grad)
                    self.assertTrue(all(not block.attn.matmul2._forward_hooks
                                        for block in teacher.blocks))

    def test_compatibility_entry_point_and_full_pool_shortcut(self):
        teacher = SelectionTeacher([[1, 1], [3, 3], [3, 3]],
                                   [[8, 0], [8, 0], [0, 8]])
        expected = rankq.jsd_rank_selection(teacher, self.data, self.labels, k=2)
        actual = rankq.jsd_selection(teacher, self.data, self.labels, k=2)
        torch.testing.assert_close(actual, expected)
        self.assertIs(rankq.jsd_rank_selection(teacher, self.data, self.labels, k=3),
                      self.data)

    def test_rejects_invalid_counts_and_misaligned_labels(self):
        for k, labels in ((0, self.labels), (4, self.labels), (1, self.labels[:2])):
            with self.subTest(k=k, label_count=len(labels)), self.assertRaises(ValueError):
                rankq.jsd_rank_selection(None, self.data, labels, k=k)


if __name__ == '__main__':
    unittest.main()
