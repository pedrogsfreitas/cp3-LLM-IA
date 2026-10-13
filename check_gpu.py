import torch

print("PyTorch:", torch.__version__)
print("CUDA (PyTorch):", torch.version.cuda)
print("GPU disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
