"""
nanoXPP Benchmark Script - Simplified for RTX 4080
"""

import os
import time
import torch
from model import GPTConfig, GPT

# Configuration for benchmarking
batch_size = 8
block_size = 1024
n_layer = 8
n_head = 8
n_embd = 768

device = 'cuda'
dtype = 'bfloat16'

# Build model
model = GPT(GPTConfig(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=True,
    vocab_size=10000,
    dropout=0.0
)).to(device)

if torch.cuda.is_available():
    model = torch.compile(model)

print(f"Model: {n_layer} layers, {n_embd} embd → ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"Device: {device} | dtype: {dtype}")

# Benchmark
x = torch.randint(0, 10000, (batch_size, block_size), device=device)

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(50):   # warm up + benchmark
        y = model(x)
    
    torch.cuda.synchronize()
    end = time.time()

print(f"Time: {(end-start)*1000/50:.2f} ms per forward pass")
print(f"Tokens/s: {batch_size * block_size * 50 / (end - start):.0f}")
