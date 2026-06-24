#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

$PYTHON generate_grpo_data.py --output "$tmpdir/grpo_prompts.jsonl" --num 4

$PYTHON - "$tmpdir/grpo_prompts.jsonl" <<'PY'
import sys

import torch

from grpo import (
    GRPOCollator,
    GRPOPromptDataset,
    RuleBasedReward,
    grpo_advantages,
    grpo_loss,
)
from sft import ChatTokenizer


data_path = sys.argv[1]
tokenizer = ChatTokenizer()
dataset = GRPOPromptDataset(data_path, tokenizer, max_prompt_len=256)
collator = GRPOCollator(tokenizer.pad_id, max_len=256)
batch = collator([dataset[0], dataset[1]])

assert batch["input_ids"].shape[0] == 2
assert len(batch["answers"]) == 2

reward_fn = RuleBasedReward(length_penalty=0.0)
rewards = reward_fn(
    prompts=["2+3?", "2+3?"],
    completions=["The answer is 5.", "The answer is 6."],
    references=["5", "5"],
)
assert rewards[0] > rewards[1]

advantages = grpo_advantages(torch.tensor([1.0, 0.0, 2.0, 1.0]), group_size=2)
loss, metrics = grpo_loss(
    new_log_probs=torch.zeros(4, 3),
    old_log_probs=torch.zeros(4, 3),
    ref_log_probs=torch.zeros(4, 3),
    advantages=advantages,
    response_mask=torch.ones(4, 3),
)

assert torch.isfinite(loss)
assert "policy_loss" in metrics
print("grpo smoke passed")
PY
