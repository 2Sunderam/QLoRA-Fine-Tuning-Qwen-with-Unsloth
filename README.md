# QLoRA Fine-Tuning Qwen2.5 with Unsloth

Fine-tune **Qwen2.5-1.5B-Instruct** for structured JSON / function-calling tasks using **QLoRA** and **Unsloth** — all inside a single Kaggle notebook running on dual GPUs.

---

## Overview

This project fine-tunes Alibaba's `Qwen2.5-1.5B-Instruct` model on the [xLAM function-calling dataset](https://huggingface.co/datasets/Beryex/xlam-function-calling-60k-sharegpt) to teach the model to reliably output structured, valid JSON for tool/function-calling scenarios. Training is made memory-efficient and fast via 4-bit NF4 quantization (QLoRA) and Unsloth's hand-written Triton kernels.

**Key goals:**
- Teach a small LLM to produce valid, schema-conformant JSON on demand
- Demonstrate multi-GPU distributed training with `accelerate` inside Kaggle
- Evaluate the fine-tuned model against both the training-domain dataset and an out-of-domain JSON benchmark

---

## What's Inside

```
.
└── unsloth-qwen-fine-tuning.ipynb   # All-in-one Kaggle notebook
```

The notebook is structured into four main sections:

| Section | Description |
|---|---|
| **Installation** | Installs Unsloth (Kaggle build), xformers, TRL, PEFT, bitsandbytes |
| **Training** | Writes and launches `train.py` with `accelerate` across 2 GPUs |
| **Evaluation — json-mode-eval** | Tests fine-tuned model on NousResearch's out-of-domain JSON benchmark |
| **Evaluation — xLAM** | Tests fine-tuned model on held-out samples from the training distribution |

---

## Model & Dataset

| | Details |
|---|---|
| **Base model** | `unsloth/qwen2.5-1.5b-instruct-bnb-4bit` |
| **Training dataset** | [`Beryex/xlam-function-calling-60k-sharegpt`](https://huggingface.co/datasets/Beryex/xlam-function-calling-60k-sharegpt) (~60k function-calling conversations in ShareGPT/ChatML format) |
| **Eval dataset 1** | [`NousResearch/json-mode-eval`](https://huggingface.co/datasets/NousResearch/json-mode-eval) (out-of-domain JSON structure benchmark) |
| **Eval dataset 2** | Last 100 rows of the xLAM training dataset (in-domain exact-match check) |

---

## Training Configuration

```python
# Model loading
model_name       = "unsloth/qwen2.5-1.5b-instruct-bnb-4bit"
max_seq_length   = 2048
load_in_4bit     = True   # 4-bit NF4 QLoRA

# LoRA adapter settings
r                = 16
lora_alpha       = 16
lora_dropout     = 0
target_modules   = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
use_gradient_checkpointing = "unsloth"

# TrainingArguments
per_device_train_batch_size  = 4
gradient_accumulation_steps  = 2    # Effective batch size = 16
num_train_epochs             = 2
warmup_steps                 = 150
learning_rate                = 2e-4
lr_scheduler_type            = "cosine"
optim                        = "adamw_8bit"
weight_decay                 = 0.01
```

Training is launched with `accelerate` across 2 GPUs:

```bash
accelerate launch --num_processes 2 train.py
```

A custom `MultiGpuSaveCallback` saves LoRA adapter checkpoints every 500 steps (Rank 0 only), avoiding the DDP pickling bug triggered by the native `save_strategy`. The final adapters are saved with:

```python
model.save_pretrained_merged("qwen2.5-xlam-lora", tokenizer, save_method="lora")
```

---

## Evaluation

### 1. NousResearch json-mode-eval (out-of-domain)

Tests whether the fine-tuned model generalizes to JSON formatting it wasn't explicitly trained on.

**Metrics scored:**
- **Valid JSON %** — Is the output parseable by `json.loads()`?
- **Schema Match %** — Do the root-level keys of the output match the ground-truth schema?

A strict system prompt (`"You must always output raw, valid JSON"`) is injected for samples missing one. Inference runs at `temperature=0.1` with Markdown code-block cleanup handled via hex escape sequences.

### 2. xLAM in-domain evaluation (last 100 samples)

Tests performance on the training distribution itself.

**Metrics scored:**
- **Valid JSON %**
- **Exact Content Match %** — Does `parsed_output == parsed_ground_truth`?

Includes 3 visual spot-checks printed before the full scoring loop for quick sanity verification.

---

## Setup & Usage

### Requirements

- Kaggle Notebook (or any environment with 2× NVIDIA GPUs)
- Python 3.12+
- CUDA 12.1+

### Installation

```bash
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

### Run Training

```bash
accelerate launch --num_processes 2 train.py
```

### Run Evaluation

All evaluation code is in the notebook cells following the training section. Simply run them after training completes (or after loading saved adapters from `qwen2.5-xlam-lora/`).

> **Note:** If `config.json` is missing from the saved adapter directory (a known edge case with 16-bit merges), the evaluation cell automatically fetches it from the base model on Hugging Face.

---

## Key Design Decisions

**Why QLoRA?** 4-bit quantization dramatically reduces VRAM — a 1.5B model in 16-bit needs ~3 GB for weights alone; QLoRA keeps the full training footprint well under a single GPU's budget.

**Why Unsloth?** Custom Triton kernels and a manual backprop engine deliver ~2× faster training and ~70% less VRAM than a standard transformers/PEFT setup, with zero accuracy degradation.

**Why `save_strategy="no"`?** Hugging Face's native DDP checkpointing triggers a pickling error when Unsloth models are serialized across processes. The custom callback sidesteps this by saving only on Rank 0 using `model.save_pretrained()` directly.

**Why ChatML?** The xLAM dataset uses the ShareGPT format (`from`/`value`). Unsloth's `get_chat_template` maps this to the native ChatML schema that Qwen2.5-Instruct expects.

---

## References

- [Unsloth](https://github.com/unslothai/unsloth)
- [Qwen2.5 on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [xLAM Dataset](https://huggingface.co/datasets/Beryex/xlam-function-calling-60k-sharegpt)
- [NousResearch json-mode-eval](https://huggingface.co/datasets/NousResearch/json-mode-eval)
- [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer)
