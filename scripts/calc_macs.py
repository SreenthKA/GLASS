"""Compute parameter count and MACs for GLASS.

Reports the numbers quoted in the paper (e.g. 15.44M parameters). MACs are
profiled for a 720p output (input 640x360, upscale x2).

Requires `thop` (pip install thop). Run from the repository root:
    python scripts/calc_macs.py
"""
import torch
from thop import profile

from basicsr.archs.glass_arch import GLASS


def calculate_macs():
    # Configuration matching the classic-SR x2 training config.
    model_config = {
        'img_size': 64,
        'patch_size': 1,
        'in_chans': 3,
        'embed_dim': 180,
        'num_heads': 6,
        'depths': (6, 6, 6, 6, 6, 6),
        'mlp_ratio': 2,
        'upscale': 2,
        'upsampler': 'pixelshuffle',
        'resi_connection': '1conv',
        'img_range': 1.,
    }

    model = GLASS(**model_config)
    model.eval()

    # 720p output (1280x720) at scale x2 -> input 640x360.
    input_h, input_w = 360, 640
    input_tensor = torch.randn(1, 3, input_h, input_w)

    print(f"Profiling MACs for 720p output (input {input_w}x{input_h})...")
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)

    print("\n" + "=" * 50)
    print(f"Model config: dim={model_config['embed_dim']}, depths={model_config['depths']}")
    print("-" * 50)
    print(f"Total parameters: {params / 1e6:.4f} M")
    print(f"Total MACs:       {macs / 1e9:.4f} G")
    print("=" * 50)


if __name__ == "__main__":
    calculate_macs()
