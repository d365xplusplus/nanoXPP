import torch

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
