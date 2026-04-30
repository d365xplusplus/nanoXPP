import os
import numpy as np
from tokenizers import Tokenizer

# ==================== Configuration ====================
input_file = 'data/xpp/input.txt'
train_bin = 'data/xpp/train.bin'
val_bin = 'data/xpp/val.bin'
tokenizer_path = 'tokenizers/xpp_tokenizer/tokenizer.json'

val_ratio = 0.1

print("=== Starting data preparation for X++ ===")

# Load custom X++ Tokenizer
print(f"Loading tokenizer from: {tokenizer_path}")
tokenizer = Tokenizer.from_file(tokenizer_path)
print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")

# Read input.txt
print("Reading input.txt ...")
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters: {len(text):,}")

# Encode using custom tokenizer
print("Encoding text into tokens...")
encoded = tokenizer.encode(text)
data = np.array(encoded.ids, dtype=np.uint32)

# Split into train and validation
n = int(len(data) * (1 - val_ratio))
train_data = data[:n]
val_data = data[n:]

# Save binary files
train_data.tofile(train_bin)
val_data.tofile(val_bin)

print("=" * 60)
print("✅ Data preparation completed successfully!")
print(f"Total tokens     : {len(data):,}")
print(f"Train tokens     : {len(train_data):,} ({len(train_data)/len(data)*100:.1f}%)")
print(f"Validation tokens: {len(val_data):,} ({len(val_data)/len(data)*100:.1f}%)")
print(f"Tokenizer vocab  : {tokenizer.get_vocab_size()}")
print(f"train.bin saved  : {train_bin}")
print(f"val.bin saved    : {val_bin}")
print("=" * 60)
