"""Convert IBM's checkpoint to an MLX build.

    python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-8bit --bits 8      # 4.1.1 (IBM's main since 27 Oct 2026)
    python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-8bit-41 --bits 8 --revision granite-switch-4.1-8b   # 4.1
    python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-4bit --bits 4
    python scripts/convert.py ibm-granite/granite-switch-4.1-8b out-bf16 --bits 16

Base weights are quantized (affine, group size 64); the two adapters stay in bf16, trimmed from IBM's
zero-padded rank-64 layout to their own ranks (guardian-core 16, decision 64). The speech encoder is not
part of these builds (text only).
"""
import argparse
import shutil
from pathlib import Path

import granite_switch_mlx

granite_switch_mlx.register()
from huggingface_hub import snapshot_download  # noqa: E402
from mlx_lm import convert  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--bits", type=int, choices=[4, 8, 16], default=8)
ap.add_argument("--group-size", type=int, default=64)
ap.add_argument("--revision", default=None, help="IBM repo revision (default: main = Granite Switch 4.1.1)")
a = ap.parse_args()
src = Path(a.src) if Path(a.src).exists() else Path(snapshot_download(a.src, revision=a.revision))
if a.bits == 16:
    convert(str(src), mlx_path=a.out, dtype="bfloat16")
else:
    convert(str(src), mlx_path=a.out, quantize=True, q_bits=a.bits, q_group_size=a.group_size)
for f in ("BUILD.md", "adapter_index.json"):
    if (src / f).exists():
        shutil.copy(src / f, Path(a.out) / f)
if (src / "io_configs").exists():
    shutil.copytree(src / "io_configs", Path(a.out) / "io_configs", dirs_exist_ok=True)
print("wrote", a.out)
