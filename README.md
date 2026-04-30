# 🤖 nanoXPP

A NanoGPT-based language model trained on X++ (Dynamics 365 Finance & Operations) code.
Built for educational purposes to demonstrate how LLMs can learn domain-specific programming languages.

---

## ✨ Features

- **X++ Focused** — Trained specifically on Dynamics 365 F&O X++ code
- **D365 Metadata Support** — Direct training from `PackagesLocalDirectory` XML files
- **Complete Training Pipeline** — Data preparation → Tokenizer → Pre-training → Inference
- **Easy Customization** — Simple configuration for further fine-tuning (SFT, LoRA, etc.)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n nanoXPP python=3.11 -y
conda activate nanoXPP

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tokenizers numpy tiktoken tqdm
```

### 2. Prepare Data

Copy your X++ XML files (custom models) into `data/xpp/` folder first:

```bash
python merge_all_xml_to_input.py
python data/prepare_xpp.py
```

### 3. Train Tokenizer

```bash
python train_xpp_tokenizer.py
```

### 4. Train Model

```bash
python train.py config/train_xpp.py
```

### 5. Run Inference

```bash
python sample.py --out_dir=out-xpp --start="class MyExtension extends RunBase"
```

---

## 📊 Model Information

| Property | Value |
|----------|-------|
| Architecture | NanoGPT (Decoder-only Transformer) |
| Recommended Size | 124M parameters (n_embd=768) |
| Current Data | ~98MB X++ code |
| Tested Hardware | RTX 4080 |

---

## 📁 Project Structure

```
nanoXPP/
├── assets/              # Images and assets
├── config/              # Training configuration files
│   └── train_xpp.py     # X++ specific training config
├── data/                # Data preparation scripts
│   └── prepare_xpp.py   # X++ data processor
├── model.py             # NanoGPT model definition
├── train.py             # Main training script
├── sample.py            # Inference/generation script
├── train_xpp_tokenizer.py  # Custom X++ tokenizer trainer
├── merge_all_xml_to_input.py  # D365 XML merger
├── bench.py             # Benchmarking script
├── scaling_laws.ipynb   # Scaling laws experiments
└── transformer_sizing.ipynb  # Model sizing calculator
```

---

## 🛠️ Requirements

- Python 3.11+
- CUDA-capable GPU (tested on RTX 4080)
- Dynamics 365 F&O `PackagesLocalDirectory` XML files

---

## 📖 Background

This project is inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
and adapted specifically for the X++ programming language used in Microsoft Dynamics 365
Finance & Operations.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add more X++ training data
- Improve the tokenizer
- Experiment with different model sizes
- Share your results

---

## 📄 Disclaimer

This project is based on karpathy/nanoGPT and is licensed under the MIT License.
