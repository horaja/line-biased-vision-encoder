#!/usr/bin/env python3
"""Test which import is causing the hang."""
import sys
print("Python started", flush=True)

print("Importing torch...", flush=True)
import torch
print(f"Torch imported: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print("Initializing CUDA...", flush=True)
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

print("All imports successful!", flush=True)
