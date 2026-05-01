"""
nanoXPP Training Script - Stable & Detailed Version
With proper argparse for command line arguments
"""

import os
import sys
import time
import numpy as np
import argparse
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# ==================== Argument Parser (Fixed) ====================
parser = argparse.ArgumentParser(description='nanoXPP Training for X++')
parser.add_argument('config', nargs='?', default='config/train_xpp.py',
                    help='Path to config file')
parser.add_argument('--max_iters', type=int, default=None)
parser.add_argument('--eval_interval', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=None)

args = parser.parse_args()

# Load config
print(f"📄 Loading config: {args.config}")
exec(open(args.config).read())

# Override with command line arguments
if args.max_iters is not None:
    max_iters = args.max_iters
    print(f"🔧 Overriding max_iters = {max_iters}")
if args.eval_interval is not None:
    eval_interval = args.eval_interval
if args.log_interval is not None:
    log_interval = args.log_interval

# -----------------------------------------------------------------------------
# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    master_process = int(os.environ['RANK']) == 0
else:
    master_process = True
    device = 'cuda'

print("=" * 80)
print("🚀 Starting nanoXPP Training")
print(f"Model Size     : ~{n_layer * n_embd * 12 / 1e6:.1f}M parameters")
print(f"Context Length : {block_size}")
print(f"Dataset        : {dataset}")
print(f"Device         : {device} | dtype: {dtype}")
print(f"Output Dir     : {out_dir}")
print("=" * 80)

os.makedirs(out_dir, exist_ok=True)

# Model
model = GPT(GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=block_size, bias=bias,
                      vocab_size=vocab_size, dropout=dropout))
model.to(device)

if compile:
    print("🔥 Compiling model with torch.compile()...")
    model = torch.compile(model)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

# Data
data_dir = os.path.join('data', dataset)
train_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16).astype(np.int64))
val_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16).astype(np.int64))

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        loss_sum = 0.0
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            loss_sum += loss.item()
        losses[split] = loss_sum / eval_iters
    model.train()
    return losses

# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
t0 = time.time()

while True:
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model': model.state_dict(),
                'model_args': {'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd, 'block_size': block_size,
                               'bias': bias, 'vocab_size': vocab_size, 'dropout': dropout},
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }, os.path.join(out_dir, 'ckpt.pt'))
            print(f"✅ Saved checkpoint to {out_dir}/ckpt.pt")

    if iter_num >= max_iters:
        break

    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        loss.backward()

        if micro_step == gradient_accumulation_steps - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    iter_num += 1

    if iter_num % log_interval == 0 and master_process:
        dt = time.time() - t0
        t0 = time.time()
        print(f"step {iter_num:5d} | loss {loss.item():.4f} | time {dt*1000:.1f}ms")

print("🎉 Training finished!")
if ddp:
    destroy_process_group()
