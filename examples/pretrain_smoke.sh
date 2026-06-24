#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

$PYTHON - <<'PY'
import torch

from train import GPTModel, ModelConfig, SyntheticDataset


config = ModelConfig(
    vocab_size=64,
    max_seq_len=8,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    d_model=32,
    d_ff=64,
    use_flash_attn=False,
    use_flash_attn_3=False,
    use_te=False,
)

dataset = SyntheticDataset(config.vocab_size, config.max_seq_len, num_samples=2)
sample = dataset[0]
batch = {
    "input_ids": torch.stack([sample["input_ids"], dataset[1]["input_ids"]]),
    "labels": torch.stack([sample["labels"], dataset[1]["labels"]]),
}

model = GPTModel(config)
logits, loss = model(batch["input_ids"], labels=batch["labels"])

assert logits.shape == (2, config.max_seq_len, config.vocab_size)
assert loss is not None and loss.ndim == 0
print("pretrain smoke passed")
PY
