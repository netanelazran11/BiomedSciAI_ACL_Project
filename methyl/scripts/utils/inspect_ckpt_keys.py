"""Quick script to print the layer-0 MLP keys from a checkpoint."""
import sys
import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location="cpu")
sd = ckpt["state_dict"]

print("=== ALL UNIQUE KEY PREFIXES (first 3 segments) ===")
prefixes = sorted(set(".".join(k.split(".")[:3]) for k in sd))
for p in prefixes:
    print(f"  {p}")

print("\n=== KEYS CONTAINING 'mlp' ===")
for k in sorted(sd.keys()):
    if "mlp" in k:
        print(f"  {k}  shape={list(sd[k].shape)}")

print("\n=== KEYS CONTAINING 'layers.0' ===")
for k in sorted(sd.keys()):
    if "layers.0" in k:
        print(f"  {k}  shape={list(sd[k].shape)}")
