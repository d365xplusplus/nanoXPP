"""
nanoXPP Configuration for X++ (Dynamics 365) Training
"""

# Model Architecture
n_layer = 8
n_head = 8
n_embd = 768
block_size = 1024
bias = True
dropout = 0.0

# Vocabulary (will be overridden by prepare script)
vocab_size = 10000   # Will be updated automatically by prepare_xpp.py

# Training Hyperparameters
learning_rate = 4e-4
max_iters = 20000          # You can increase this later
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# Batch size settings (good for RTX 4080)
batch_size = 8
gradient_accumulation_steps = 8   # effective batch size = 64

# Evaluation
eval_interval = 100
eval_iters = 100
log_interval = 10

# Data
dataset = 'xpp'
out_dir = 'out-xpp'

# System
device = 'cuda'
dtype = 'bfloat16'      # Best for RTX 4080
compile = True          # torch.compile for speed

# Wandb (optional)
wandb_log = False
wandb_project = 'nanoXPP'
wandb_run_name = 'xpp-124M'

# Learning rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-5import torch

# ==================== Model Configuration ====================
n_layer = 8
n_head = 8
n_embd = 768                    # ≈ 124M parameters
block_size = 1024               # Context length
dropout = 0.1

# ==================== Training Configuration ====================
batch_size = 8
gradient_accumulation_steps = 8   # Effective batch size = 64
learning_rate = 4e-4
max_iters = 30000                 # You can stop earlier if needed
eval_interval = 200
eval_iters = 100
warmup_iters = 500

# Optimizer
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

# ==================== System Configuration ====================
device = 'cuda'
dtype = 'bfloat16'                # Recommended for RTX 4080
compile = True

# ==================== Data & Output ====================
dataset = 'xpp'
data_dir = 'data/xpp'
out_dir = 'out-xpp'

# Logging
wandb_log = False
wandb_project = 'nanoXPP'
wandb_run_name = 'xpp-124M'

print("=== nanoXPP Training Config Loaded ===")
print(f"Model size ≈ 124M parameters (n_embd={n_embd})")
