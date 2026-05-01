"""
Generate X++ code using nanoXPP model with custom tokenizer.
"""
import torch
import argparse
from model import GPTConfig, GPT
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='out-xpp')
parser.add_argument('--start', default="class ")
parser.add_argument('--max_new_tokens', type=int, default=400)
parser.add_argument('--temperature', type=float, default=0.75)
parser.add_argument('--top_k', type=int, default=60)
args = parser.parse_args()

# Load model
ckpt = torch.load(f'{args.out_dir}/ckpt.pt', map_location='cuda')
model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
model.cuda()

# Load BPE Tokenizer
tokenizer = Tokenizer.from_file("tokenizers/xpp_tokenizer/tokenizer.json")
print(f"✅ Using BPE Tokenizer (vocab size: {tokenizer.get_vocab_size()})")

# Generation
print("\n=== Generating X++ Code ===")
print(args.start, end="", flush=True)

encoded = tokenizer.encode(args.start)
x = torch.tensor(encoded.ids, dtype=torch.long, device='cuda').unsqueeze(0)

with torch.no_grad():
    for _ in range(args.max_new_tokens):
        # Fix: handle possible tuple return from model
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[:, -1, :]
        
        logits = logits / args.temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        if args.top_k:
            v, _ = torch.topk(probs, args.top_k)
            probs[probs < v[:, [-1]]] = 0
        
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        
        print(tokenizer.decode([next_token.item()]), end="", flush=True)

print("\n\n=== Generation finished ===")


