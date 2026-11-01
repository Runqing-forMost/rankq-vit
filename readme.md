# RankQ-ViT

Pytorch implementation for **Beyond Similarity Entropy: A Rank-Guided Data-Free Quantization Framework for Vision Transformers**


## ImageNet results

Top-1 accuracy (%) of RankQ-ViT from the supplied experiment tables. All
RankQ-ViT entries use **32 synthetic calibration images**, with no real images
used for calibration. `Wn/Am` denotes `n`-bit weights and `m`-bit activations;
the full-precision reference uses W32/A32.

| Model | Full precision | W4/A4 | W5/A5 | W4/A8 | W8/A8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-S | 81.39 | 44.06 ± 0.18 | 68.07 ± 0.17 | 78.76 ± 0.12 | 81.19 ± 0.08 |
| ViT-B | 84.53 | 17.77 ± 0.22 | 50.19 ± 0.24 | 82.44 ± 0.14 | 84.25 ± 0.07 |
| DeiT-T | 72.21 | 44.35 ± 0.20 | 62.38 ± 0.14 | 68.55 ± 0.12 | 72.22 ± 0.06 |
| DeiT-S | 79.85 | 50.40 ± 0.27 | 64.13 ± 0.19 | 77.86 ± 0.15 | 79.59 ± 0.05 |
| DeiT-B | 81.80 | 64.34 ± 0.17 | 75.16 ± 0.21 | 80.63 ± 0.10 | 81.63 ± 0.04 |
| Swin-T | 81.35 | 71.57 ± 0.22 | 79.05 ± 0.20 | 79.24 ± 0.10 | 81.14 ± 0.05 |
| Swin-S | 83.20 | 77.78 ± 0.23 | 81.25 ± 0.26 | 82.36 ± 0.15 | 83.19 ± 0.09 |

The `±` values are preserved from the supplied tables, which do not define
their statistical meaning or the number of runs. These are reported results;
the example configuration retains 1,024 calibration images and is not the
complete reproduction configuration for the 32-image experiments above.

## Environment

The runtime imports these third-party packages:

- PyTorch and torchvision
- timm
- NumPy
- OmegaConf
- tqdm
- Pillow

Use a Python environment with a compatible PyTorch/torchvision pair and a CUDA
build suitable for your GPU. Install the remaining dependencies in that
environment:

```bash
python -m pip install "timm==0.9.2" numpy omegaconf tqdm Pillow
```

The local CPU checks used Python 3.8.17, PyTorch 2.4.1, timm 0.9.2, NumPy 1.23.5,
and OmegaConf 2.3.0. This is a record of the check environment, not a verified
full GPU training environment. The model builder replaces timm attention
forward methods and depends on their internal attributes; other timm versions
need compatibility checks.


Pretrained weights must be available through timm's cache or downloaded when
the model is first constructed.

## Prepare ImageNet validation data

Arrange validation images in torchvision `ImageFolder` format:

```text
/path/to/imagenet/
└── val/
    ├── n01440764/
    │   └── image.JPEG
    ├── n01443537/
    │   └── image.JPEG
    └── ...
```

Use the full ImageNet-1k class directory layout so the sorted folder indices
match the pretrained classifier's class indices. A flat validation directory
must be reorganized before use.

Set `dataset` in `configs/imageNet.yaml` to the directory containing `val/`.
The entry point does not require a `train/` directory. The example config
contains a machine-specific dataset path; replace it for your environment.

## Run an experiment

Run from the project root:

```bash
python rankq.py --config configs/imageNet.yaml
```

Omitting `--config` uses `configs/imageNet.yaml` beside the script.

To use a separate synthetic-image cache or force regeneration:

```bash
python rankq.py --config configs/imageNet.yaml --cache-dir ./generated_imgs/experiment_01
python rankq.py --config configs/imageNet.yaml --regenerate
```

### Model aliases

| Alias | timm model name |
| --- | --- |
| `deit_tiny` | `deit_tiny_patch16_224` |
| `deit_small` | `deit_small_patch16_224` |
| `deit_base` | `deit_base_patch16_224` |
| `vit_small` | `vit_small_patch16_224` |
| `vit_base` | `vit_base_patch16_224` |
| `swin_tiny` | `swin_tiny_patch4_window7_224` |
| `swin_small` | `swin_small_patch4_window7_224` |

These aliases are recognized by the entry point; complete GPU experiments
have not been validated for every model. Swin subset selection is unsupported:
set `target_size` equal to `memory_bank` for Swin. Synthesis assumes RGB images
of size 224 × 224 and a 1,000-class ImageNet output head.

## Cache and outputs

The default cache contains normalized floating-point image tensors and integer
pseudo-labels stored as NumPy arrays:

```text
generated_imgs/my_generated/
├── gen_deit_small_1024.npy        # Shape: [1024, 3, 224, 224]
└── label_gen_deit_small_1024.npy  # Shape: [1024]
```

