#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

$PYTHON generate_sample_data.py \
  --output_dir "$tmpdir" \
  --num_sft 2 \
  --num_preference 2 \
  --num_prompts 2

$PYTHON - "$tmpdir/preference_train.jsonl" <<'PY'
import sys

import torch

from reward_model import PreferenceCollator, PreferenceDataset, reward_loss
from sft import ChatTokenizer


data_path = sys.argv[1]
tokenizer = ChatTokenizer()
dataset = PreferenceDataset(data_path, tokenizer, max_seq_len=256)
collator = PreferenceCollator(tokenizer.pad_id, max_seq_len=256)
batch = collator([dataset[0], dataset[1]])

assert batch["input_ids"].shape[0] == 4
assert batch["attention_mask"].shape == batch["input_ids"].shape

loss, metrics = reward_loss(
    chosen_rewards=torch.tensor([2.0, 1.0]),
    rejected_rewards=torch.tensor([0.0, -1.0]),
)

assert loss.ndim == 0
assert metrics["accuracy"] == 1.0
print("reward model smoke passed")
PY
