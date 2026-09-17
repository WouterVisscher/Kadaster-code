"""Small sanity check: find the compute device available to PyTorch."""

import torch

print(f"torch version : {torch.__version__}")
print(f"ROCm/HIP ver  : {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    for i in range(torch.cuda.device_count()):
        print(f"  device {i}   : {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")

print(f"selected device: {device}")

# Tiny smoke test on the selected device
x = torch.rand(3, 3, device=device)
y = x @ x
print(f"matmul on {y.device} OK, result sum = {y.sum().item():.4f}")
