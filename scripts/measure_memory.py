"""Measure peak GPU memory of GLASS across increasing input resolutions.

Reproduces the memory-scaling experiment (paper Fig. 3), demonstrating
near-linear memory growth with input size. Requires a CUDA device.
Run from the repository root:
    python scripts/measure_memory.py
"""
import gc

import torch

from basicsr.archs.glass_arch import GLASS


def measure_memory(model, input_size, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    x = torch.randn(1, 3, input_size, input_size).to(device)
    with torch.no_grad():
        _ = model(x)

    return torch.cuda.max_memory_allocated() / (1024 ** 3)


if __name__ == "__main__":
    device = "cuda"

    # Configuration matching the classic-SR x2 training config.
    model = GLASS(
        upscale=2,
        in_chans=3,
        img_size=96,  # must be >= the largest tested input
        img_range=1.0,
        embed_dim=180,
        num_heads=6,
        depths=(6, 6, 6, 6, 6, 6),
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    ).to(device)
    model.eval()

    input_sizes = [48, 56, 64, 72, 80, 84]
    memory_usage = []

    for size in input_sizes:
        try:
            mem = measure_memory(model, size)
            memory_usage.append(mem)
            print(f"Input {size}x{size}: {mem:.2f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Input {size}x{size}: OOM")
                memory_usage.append(None)
                torch.cuda.empty_cache()
            else:
                raise

    print("\nFinal results:")
    print("Input sizes:", input_sizes)
    print("Memory (GB):", memory_usage)
