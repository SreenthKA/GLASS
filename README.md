# GLASS: Global Linear Attention for Efficient Single-Image Super-Resolution

> **GLASS** &nbsp;·&nbsp; Accepted at **IEEE TENCON 2026**
>
> Sreenath K A\*, Nishalini K\*, Jiji C V &nbsp;—&nbsp; Department of Computer Science, Shiv Nadar University Chennai
>
> <sub>\* Equal contribution</sub>

GLASS is a Mamba-inspired architecture for single-image super-resolution (SISR)
that replaces Mamba's **recurrent selective state-space scan** with
**parallelizable linear attention** augmented by positional encodings
(RoPE + LePE). This preserves a global receptive field and linear
computational complexity while eliminating sequential computation — giving
**24.3% fewer parameters** than MambaIR (20.42M → 15.44M) and **near-linear
memory scaling** with input resolution, at competitive reconstruction quality.

---

## Highlights

- **Linear attention inside a Mamba-style block.** GLASS keeps the benefits of
  Mamba's block design (forget-gate-like gating, modified block) but removes
  recurrence, enabling fully parallel global modeling.
- **24.3% parameter reduction** vs. MambaIR while staying within ~0.1–0.2 dB
  PSNR on most benchmarks.
- **Near-linear GPU memory scaling** with input size, versus the quadratic
  growth of full self-attention.

## Method

GLASS follows a standard three-stage SISR pipeline — shallow feature
extraction (a 3×3 conv), deep feature extraction (stacked Residual Linear
Attention Groups), and high-quality reconstruction (pixel-shuffle upsampling).
The novel components are:

| Module | Description | Code |
|---|---|---|
| **RLAB** — Residual Linear Attention Block | VMLA core + bottleneck conv with channel attention, each in a learnable-scaled residual | `VSSBlock` |
| **VMLA** — Vision Mamba Linear Attention | Dual-pathway block: a SiLU gating path modulates a depthwise-conv + linear-attention path, fused by a Hadamard product | `MLLACore` |
| **Linear Attention** | ELU feature-map linear attention with RoPE on Q/K and LePE (depthwise conv) on V | `LinearAttention`, `RoPE` |
| **RLAG** — Residual Linear Attention Group | A residual group of `L` RLABs followed by a conv | `ResidualGroup` |

The default classic-SR model stacks `N = 6` RLAGs of `L = 6` RLABs each, with
`embed_dim = 180` and `num_heads = 6`. The full architecture lives in
[`basicsr/archs/glass_arch.py`](basicsr/archs/glass_arch.py).

## Results

Quantitative comparison (PSNR / SSIM) on the Y channel for classic image SR.
GLASS is competitive with Transformer- and SSM-based methods while using far
fewer parameters.

| Scale | Method | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|---|---|---|---|---|---|---|
| ×2 | MambaIR | 38.57 / 0.9627 | 34.67 / 0.9261 | 32.58 / 0.9048 | 34.15 / 0.9446 | 40.28 / 0.9806 |
| ×2 | **GLASS (Ours)** | 38.41 / 0.9621 | 34.56 / 0.9254 | 32.48 / 0.9034 | 33.75 / 0.9416 | 39.99 / 0.9801 |
| ×3 | MambaIR | 35.13 / 0.9326 | 31.06 / 0.8549 | 29.53 / 0.8162 | 29.99 / 0.8888 | 35.30 / 0.9546 |
| ×3 | **GLASS (Ours)** | 34.92 / 0.9310 | 30.78 / 0.8495 | 29.40 / 0.8125 | 29.49 / 0.8764 | 35.01 / 0.9523 |
| ×4 | MambaIR | 33.09 / 0.9046 | 29.20 / 0.7961 | 27.98 / 0.7503 | 27.68 / 0.8327 | 32.32 / 0.9287 |
| ×4 | **GLASS (Ours)** | 32.77 / 0.9013 | 29.05 / 0.7912 | 27.86 / 0.7455 | 27.11 / 0.8150 | 31.86 / 0.9225 |

**Efficiency (parameters):**

| Method | Params (M) |
|---|---|
| MambaIR | 20.42 |
| MambaIRv2 | 22.90 |
| **GLASS (Ours)** | **15.44** |

## Installation

GLASS is built on the [BasicSR](https://github.com/xinntao/BasicSR) framework.
Unlike MambaIR, it has **no dependency on `mamba-ssm` or `causal-conv1d`** — the
selective scan is replaced entirely by linear attention.

```bash
# create an environment (example with conda)
conda create -n glass python=3.10 -y
conda activate glass

# install PyTorch matching your CUDA version, then:
pip install -r requirements.txt

# install this repo (BasicSR) in editable mode
python setup.py develop
```

## Datasets

Train on **DF2K** and evaluate on Set5, Set14, BSD100, Urban100 and Manga109.
See [`datasets/README.md`](datasets/README.md) for the expected directory
layout (the config paths assume `datasets/DF2K/...` and `datasets/test/...`).

## Training

```bash
# ×2 from scratch
python basicsr/train.py -opt options/train/glass/train_GLASS_SR_x2.yml

# ×3 / ×4 fine-tune from the ×2 checkpoint
# (set path.pretrain_network_g to your GLASS_SR_x2 weights first)
python basicsr/train.py -opt options/train/glass/train_GLASS_SR_x3.yml
python basicsr/train.py -opt options/train/glass/train_GLASS_SR_x4.yml
```

For multi-GPU distributed training, launch with
`torchrun --nproc_per_node=<N> basicsr/train.py -opt <config> --launcher pytorch`.

## Testing

Place the pretrained weights under `experiments/pretrained_models/` (matching
the `pretrain_network_g` paths in the test configs), then:

```bash
python basicsr/test.py -opt options/test/glass/test_GLASS_SR_x2.yml
python basicsr/test.py -opt options/test/glass/test_GLASS_SR_x3.yml
python basicsr/test.py -opt options/test/glass/test_GLASS_SR_x4.yml
```

## Pretrained Models

Pretrained GLASS checkpoints are released separately (see the repository
**Releases** page). Download them into `experiments/pretrained_models/`.

## Analysis scripts

- [`scripts/calc_macs.py`](scripts/calc_macs.py) — report parameter count and
  MACs (needs `thop`).
- [`scripts/measure_memory.py`](scripts/measure_memory.py) — reproduce the
  memory-scaling experiment (needs a CUDA device).

## Citation

If you find GLASS useful, please cite:

```bibtex
@inproceedings{glass2026,
  title     = {GLASS: Global Linear Attention for Efficient Single-Image Super-Resolution},
  author    = {K A, Sreenath and K, Nishalini and C V, Jiji},
  booktitle = {IEEE Region 10 Conference (TENCON)},
  year      = {2026}
}
```

## Acknowledgements

This codebase is built upon [MambaIR](https://github.com/csguoh/MambaIR) and the
[BasicSR](https://github.com/xinntao/BasicSR) toolbox. We thank the authors for
their excellent open-source work.

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).
