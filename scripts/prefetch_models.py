"""Optional: populate upstream caches ahead of the first inference."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download, hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--depth-size", choices=["s", "b", "l"], default="l")
args = p.parse_args()
snapshot_download(
    f"lpiccinelli/unidepth-v2-vit{args.depth_size}14",
    allow_patterns=["config.json", "model.safetensors"],
    max_workers=2,
)
for name in ["depth_anything_v2_vitb.pth", "prior_depth_anything_vitb.pth"]:
    hf_hub_download("Rain729/Prior-Depth-Anything", filename=name)
import torch

target = Path(torch.hub.get_dir()) / "geocalib/pinhole.tar"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    urlretrieve(
        "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
        target,
    )
print(
    "Depth and calibration weights cached. DROID-SLAM weights are fetched by ViPE on first run."
)
