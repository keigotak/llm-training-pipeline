# LLM Training Pipeline

Research-oriented PyTorch implementation of LLM pretraining, SFT, preference optimization, and RLHF-style post-training.

This repository explores the major stages of contemporary language model development:

- Decoder-only language model pre-training
- Supervised fine-tuning (SFT)
- Preference optimization (DPO / IPO / SimPO)
- RLHF-style post-training (reward modeling / PPO / GRPO)
- Multimodal vision-language training prototypes

The goal is not to reproduce frontier-scale training, but to provide a readable and extensible implementation of the key algorithms, model components, and engineering patterns used in modern LLM training.

⸻

## Why this project exists

Most open-source examples focus on a single stage of language model training.

This project aims to show how the full workflow fits together:

```text
Pre-training
    ↓
Supervised Fine-Tuning (SFT)
    ↓
Preference Optimization / RLHF-style Post-Training
    ↓
Evaluation / Iteration
```

I built this repository as an independent research and engineering project to deepen my understanding of modern LLM training systems, especially the transition from base models to instruction-following and preference-optimized models.

My professional work includes biology-facing biomedical language models, scientific evidence synthesis workflows, evaluation pipelines, and agentic AI systems. This repository serves as a compact environment for exploring the algorithmic and engineering foundations behind modern LLM training and post-training systems.

⸻

## Pipeline Overview

```text
                                  ┌──→ Reward Model ──→ PPO
                                  │    reward_model.py   ppo.py
Pre-training ──→ SFT ──┬──────────┤
   train.py      sft.py│          ├──→ DPO / IPO / SimPO
                       │          │    dpo.py
                       │          └──→ GRPO
                       │               grpo.py
                       │
                       └──→ Multimodal Vision-Language Training
                            multimodal_train.py
```

⸻

## Design Principles

1. Readability over maximum optimization
2. Modular separation between model, data, training, and evaluation
3. Reproducible small-scale experiments
4. Minimal assumptions about infrastructure
5. Clear distinction between implemented, experimental, and future components

⸻

## Current Status

This is an experimental research codebase.

It is intended for small-scale experiments, algorithmic study, and engineering exploration. It has not been validated at frontier-model scale.

| Area | Status |
|---|---|
| Decoder-only pre-training | Implemented |
| Supervised fine-tuning (SFT) | Implemented |
| Preference optimization | Implemented |
| Reward model training | Implemented |
| PPO-style post-training | Implemented |
| GRPO-style post-training | Implemented |
| LoRA fine-tuning | Implemented |
| DeepSpeed training configs | Implemented |
| Vision-language training | Prototype |
| FP8 / Transformer Engine support | Experimental |
| Flash Attention 3 support | Experimental |
| Large-scale benchmarking | Future work |

⸻

## Features

### Core LLM Training

* Decoder-only Transformer architecture
* Rotary Positional Embeddings (RoPE)
* RMSNorm
* SwiGLU feed-forward layers
* Grouped-Query Attention (GQA)
* BF16 mixed precision
* DeepSpeed ZeRO-2 / ZeRO-3 configuration examples
* Hugging Face dataset preparation utilities

### Supervised Fine-Tuning

* Multi-turn conversation format
* Assistant-token-only loss masking
* Full-parameter fine-tuning
* LoRA fine-tuning
* Optional LoRA merge after training

### Preference Optimization

* DPO
* IPO
* SimPO
* Reference-model based preference optimization
* Reference-free preference optimization variants

### RLHF-Style Post-Training

* Reward model training
* PPO-style policy optimization
* GRPO-style group-relative optimization
* KL penalty support
* Rule-based reward examples for reasoning-style tasks

### Multimodal Prototypes

* Vision-language model wrapper
* ViT-style vision encoder
* CLIP / SigLIP-style encoder support
* Image-text alignment stage
* Multimodal SFT stage
* Multimodal DPO / GRPO prototypes
* Simple video-frame sampling utilities

⸻

## Architecture

| Component | Implementation |
|---|---|
| Attention | Flash Attention 2 / optional Flash Attention 3 |
| Position encoding | RoPE |
| Feed-forward network | SwiGLU |
| Normalization | RMSNorm |
| Mixed precision | BF16, experimental FP8 |
| Distributed training | DeepSpeed ZeRO-2 / ZeRO-3 |
| Fine-tuning | Full-parameter fine-tuning / LoRA |
| Preference optimization | DPO / IPO / SimPO |
| RLHF-style training | Reward model + PPO / GRPO |
| Vision encoder | ViT / CLIP / SigLIP-style components |
| Multimodal training | Alignment, MM-SFT, MM-DPO / MM-GRPO prototypes |

⸻

## Model Configurations

The repository includes several model-size presets for experimentation.

| Name | Layers | Heads | Hidden Size | Approx. Parameters |
|---|---:|---:|---:|---:|
| 125M | 12 | 12 | 768 | ~125M |
| 350M | 24 | 16 | 1024 | ~350M |
| 1.3B | 24 | 32 | 2048 | ~1.3B |
| 6.7B | 32 | 32 | 4096 | ~6.7B |
| 13B | 40 | 40 | 5120 | ~13B |

These configurations are provided for code structure and experimentation. Actual feasibility depends on hardware, dataset size, optimizer settings, sequence length, and distributed training setup.

⸻

## Quality Gate

This repository includes lightweight CPU checks for the core model, data, loss, and reward paths. These checks do not replace full CUDA/DeepSpeed training, but they make the implementation reviewable without requiring a GPU.

What is currently checked:

* GPT forward pass shape and scalar loss on a tiny CPU model
* SFT, preference, and GRPO JSONL data-format loading
* DPO, reward-model, and GRPO loss/reward behavior
* Black formatting, Ruff linting, Python syntax checks, and pytest in GitHub Actions
* Stage-level CPU smoke examples under `examples/`

Smoke examples:

```bash
bash examples/pretrain_smoke.sh
bash examples/sft_smoke.sh
bash examples/dpo_smoke.sh
bash examples/reward_model_smoke.sh
bash examples/ppo_smoke.sh
bash examples/grpo_smoke.sh
```

Full local quality gate:

```bash
black --check .
ruff check .
pytest -q
python -m py_compile train.py sft.py dpo.py reward_model.py ppo.py grpo.py multimodal_train.py
```

Limitations:

* Full training paths are intended for CUDA environments with DeepSpeed.
* Flash Attention 3 and Transformer Engine are optional GPU dependencies.
* Multimodal training includes dataset, model, and training structure, but meaningful results require real image/video datasets and GPU training.
* Synthetic datasets are format checks, not training-quality corpora.

⸻

## Installation

Python 3.11 is recommended. The GitHub Actions quality gate runs on Python 3.11.

```bash
pip install "torch>=2.1" deepspeed transformer-engine flash-attn \
    datasets tiktoken wandb tqdm
```

Optional Flash Attention 3 support for Hopper GPUs:

```bash
pip install kernels
```

If installation of flash-attn or transformer-engine fails, first verify CUDA, PyTorch, compiler, and GPU compatibility.

⸻

## Quick Start with Synthetic Data

Generate synthetic data for the major stages:

Run CPU quality checks:

```bash
pytest -q
bash examples/pretrain_smoke.sh
bash examples/sft_smoke.sh
bash examples/dpo_smoke.sh
bash examples/reward_model_smoke.sh
bash examples/ppo_smoke.sh
bash examples/grpo_smoke.sh
```

Generate sample data for all stages:

```bash
python generate_sample_data.py --output_dir data/
```

Run a small pre-training smoke test:

```bash
deepspeed --num_gpus=1 train.py \
    --model_size 125m \
    --deepspeed ds_config.json \
    --max_steps 100
```

Run a small SFT smoke test:

```bash
deepspeed --num_gpus=1 sft.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/final \
    --data_path data/sft_train.jsonl \
    --max_steps 100
```

⸻

## Usage

### 1. Pre-Training

Prepare tokenized data from a Hugging Face dataset:

```bash
python prepare_data.py \
    --dataset openwebtext \
    --tokenizer gpt2 \
    --output data/owt.pt
```

Run pre-training:

```bash
deepspeed --num_gpus=8 train.py \
    --model_size 350m \
    --deepspeed ds_config.json \
    --data_path data/owt.pt \
    --max_steps 50000
```

Run with optional Flash Attention 3 support:

```bash
deepspeed --num_gpus=8 train.py \
    --model_size 350m \
    --deepspeed ds_config.json \
    --data_path data/owt.pt \
    --use_flash_attn_3 \
    --max_steps 50000
```

⸻

### 2. Supervised Fine-Tuning

Full-parameter SFT:

```bash
deepspeed --num_gpus=8 sft.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/final \
    --data_path data/sft_train.jsonl \
    --max_steps 3000
```

LoRA fine-tuning:

```bash
deepspeed --num_gpus=8 sft.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/final \
    --data_path data/sft_train.jsonl \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --merge_lora_on_save \
    --max_steps 3000
```

⸻

### 3. Preference Optimization

#### DPO

```bash
deepspeed --num_gpus=8 dpo.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --loss_type dpo \
    --beta 0.1 \
    --max_steps 2000
```

#### IPO

```bash
deepspeed --num_gpus=8 dpo.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --loss_type ipo \
    --beta 0.1
```

#### SimPO

```bash
deepspeed --num_gpus=8 dpo.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --loss_type simpo \
    --beta 2.0
```

⸻

### 4. Reward Modeling and PPO

Train a reward model:

```bash
deepspeed --num_gpus=8 reward_model.py \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/sft_final \
    --data_path data/preference_train.jsonl \
    --max_steps 2000
```

Run PPO-style post-training:

```bash
deepspeed --num_gpus=8 ppo.py \
    --deepspeed ds_config_sft.json \
    --policy_model ./checkpoints/sft_final \
    --reward_model ./checkpoints/rm_final \
    --data_path data/prompts.jsonl \
    --max_steps 1000 \
    --ppo_epochs 4 \
    --kl_coef 0.05
```

⸻

### 5. GRPO-Style Training

Generate synthetic reasoning prompts:

```bash
python generate_grpo_data.py \
    --output data/grpo_prompts.jsonl \
    --num 5000
```

Run GRPO with rule-based rewards:

```bash
deepspeed --num_gpus=8 grpo.py \
    --deepspeed ds_config_sft.json \
    --policy_model ./checkpoints/sft_final \
    --data_path data/grpo_prompts.jsonl \
    --group_size 8 \
    --kl_coef 0.04 \
    --max_steps 1000
```

Run GRPO with a reward model:

```bash
deepspeed --num_gpus=8 grpo.py \
    --deepspeed ds_config_sft.json \
    --policy_model ./checkpoints/sft_final \
    --data_path data/grpo_prompts.jsonl \
    --reward_type model \
    --reward_model_path ./checkpoints/rm_final \
    --group_size 8
```

⸻

### 6. Multimodal Vision-Language Training

Generate synthetic multimodal data:

```bash
python multimodal_data.py
```

Stage 1: vision-language alignment.

```bash
deepspeed --num_gpus=8 multimodal_train.py \
    --stage 1 \
    --deepspeed ds_config_sft.json \
    --data_path data/mm_alignment.jsonl \
    --image_dir data/images/ \
    --max_steps 5000
```

Stage 2: multimodal instruction tuning.

```bash
deepspeed --num_gpus=8 multimodal_train.py \
    --stage 2 \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/mm_stage1_final \
    --data_path data/mm_sft.jsonl \
    --image_dir data/ \
    --max_steps 10000
```

Stage 3: multimodal DPO.

```bash
deepspeed --num_gpus=8 multimodal_train.py \
    --stage 3 \
    --rl_method dpo \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/mm_stage2_final \
    --data_path data/mm_dpo.jsonl \
    --image_dir data/ \
    --max_steps 2000
```

Stage 3: multimodal GRPO.

```bash
deepspeed --num_gpus=8 multimodal_train.py \
    --stage 3 \
    --rl_method grpo \
    --deepspeed ds_config_sft.json \
    --base_model ./checkpoints/mm_stage2_final \
    --data_path data/mm_grpo.jsonl \
    --image_dir data/ \
    --group_size 4 \
    --max_steps 1000
```

⸻

## Data Formats

### SFT

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Preference Data

```json
{
  "chosen": [
    {"role": "user", "content": "Explain AI."},
    {"role": "assistant", "content": "AI is a broad field..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain AI."},
    {"role": "assistant", "content": "idk"}
  ]
}
```

### PPO Prompts

```json
{
  "prompt": [
    {"role": "user", "content": "Write a poem about space."}
  ]
}
```

### GRPO Prompts

```json
{"prompt": "What is 25 × 4 + 3?", "answer": "103"}
{"prompt": "Solve for x: 3x + 1 = 10", "answer": "3"}
{"prompt": "Explain how gravity works."}
```

### Multimodal Alignment

```json
{
  "image": "path/to/image.jpg",
  "caption": "A cat sitting on a mat."
}
```

### Multimodal SFT

```json
{
  "image": "path.jpg",
  "messages": [
    {"role": "user", "content": "<image>\nDescribe this image."},
    {"role": "assistant", "content": "The image shows..."}
  ]
}
```

### Multimodal DPO

```json
{
  "image": "path.jpg",
  "chosen": [
    {"role": "user", "content": "<image>\nDescribe the image."},
    {"role": "assistant", "content": "Detailed description..."}
  ],
  "rejected": [
    {"role": "user", "content": "<image>\nDescribe the image."},
    {"role": "assistant", "content": "I see an image."}
  ]
}
```

⸻

## Alignment Methods

| Method | Reward Model | Reference Model | Online Generation | Relative Complexity | Typical Use |
|---|---|---|---|---|---|
| DPO | No | Yes | No | Low | General preference optimization |
| IPO | No | Yes | No | Low | Stable preference optimization |
| SimPO | No | No | No | Low | Reference-free preference optimization |
| PPO | Yes | Yes | Yes | High | Flexible RLHF-style optimization |
| GRPO | Optional | Yes | Yes | Medium | Group-relative reasoning-style optimization |

General recommendation:

* Start with SFT.
* Use DPO for simple preference optimization.
* Use reward modeling + PPO when explicit learned rewards are needed.
* Use GRPO-style training for rule-based or group-relative reasoning experiments.

⸻

## DeepSpeed Configurations

| Config | ZeRO Stage | Typical Use |
|---|---:|---|
| ds_config.json | 2 | Pre-training experiments |
| ds_config_zero3.json | 3 | Larger model experiments |
| ds_config_sft.json | 2 | SFT / DPO / reward model / PPO / GRPO |

⸻

## Project Structure

```text
.
├── train.py                 # Decoder-only LLM pre-training
├── prepare_data.py          # Hugging Face dataset tokenization
├── sft.py                   # Supervised fine-tuning and LoRA
├── dpo.py                   # DPO / IPO / SimPO
├── reward_model.py          # Reward model training
├── ppo.py                   # PPO (RLHF)
├── grpo.py                  # GRPO (DeepSeek-R1 style)
├── vision_encoder.py        # ViT encoder + pixel shuffle + projector
├── multimodal_model.py      # Vision-language model (LLaVA/InternVL style)
├── multimodal_train.py      # Multimodal training (all stages)
├── multimodal_data.py       # Multimodal datasets + synthetic data generator
├── generate_sample_data.py  # Synthetic SFT/DPO/PPO test data
├── generate_grpo_data.py    # Math/reasoning GRPO data generator
├── examples/                # CPU smoke checks for each stage
├── tests/                   # Pytest coverage for model/data/reward paths
├── .github/workflows/       # CI quality gate
├── ds_config.json           # DeepSpeed ZeRO-2 config (pre-training)
├── ds_config_zero3.json     # DeepSpeed ZeRO-3 config (large models)
├── ds_config_sft.json       # DeepSpeed config (post-training)
└── LICENSE
```

⸻

## Validation Status

This repository currently focuses on implementation clarity and small-scale experimentation.

Validated:

* Script-level training flows with synthetic data
* Small-scale pre-training and SFT smoke tests
* Preference-data format handling
* Modular training-stage structure

Not yet validated:

* Frontier-scale pre-training
* Large-scale distributed benchmarking
* Production-grade data curation
* Full reward-model evaluation suite
* Large-scale multimodal training

⸻

## Roadmap

Planned improvements:

- Add unit tests
- Add reproducible experiment configs
- Add small benchmark runs
- Add training curves and example logs
- Add more rigorous reward-model evaluation
- Improve documentation for each training stage
- Add agent-specific post-training examples

⸻

References

* FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
* FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
* DeepSpeed ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
* LoRA: Low-Rank Adaptation of Large Language Models
* DPO: Direct Preference Optimization
* IPO: A General Theoretical Paradigm to Understand Learning from Human Preferences
* SimPO: Simple Preference Optimization
* Training Language Models to Follow Instructions with Human Feedback
* GRPO / DeepSeek-R1-style group-relative optimization
* LLaVA: Large Language and Vision Assistant
* InternVL
* SigLIP

⸻

License

MIT
