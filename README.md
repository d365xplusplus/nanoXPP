# nanoXPP

**A NanoGPT-based specialized language model for X++ (Dynamics 365 Finance and Operations)**

An open-source project focused on training small but effective language models for **X++** code generation, completion, and understanding in Microsoft Dynamics 365 F&O.

---

### ✨ Project Goals

- Build a lightweight, specialized LLM for X++ development
- Support training from scratch and continued pre-training
- Emphasize real-world business logic and Dynamics 365 best practices
- Make it easy for individual developers and companies to train their own X++ models

---

### 📌 Key Features

- **Custom X++ Tokenizer** — Better handling of X++ keywords, table names, methods, and indentation
- **D365 Metadata Support** — Direct training from `PackagesLocalDirectory` XML files
- **Complete Training Pipeline** — Data preparation → Tokenizer → Pre-training → Inference
- **Easy Customization** — Simple configuration for further fine-tuning (SFT, LoRA, etc.)

---

### 🚀 Quick Start

#### 1. Environment Setup
```bash
conda create -n nanoXPP python=3.11 -y
conda activate nanoXPP

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tokenizers numpy tiktoken tqdm
