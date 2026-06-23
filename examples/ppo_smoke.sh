#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

python generate_sample_data.py \
  --output_dir "$tmpdir" \
  --num_sft 2 \
  --num_preference 2 \
  --num_prompts 2

python - "$tmpdir/prompts.jsonl" <<'PY'
import sys

from ppo import PPOConfig, PromptCollator, PromptDataset
from sft import ChatTokenizer


data_path = sys.argv[1]
tokenizer = ChatTokenizer()
dataset = PromptDataset(data_path, tokenizer, max_prompt_len=64)
collator = PromptCollator(tokenizer.pad_id, max_len=64)
batch = collator([dataset[0], dataset[1]])
config = PPOConfig(max_new_tokens=4, rollout_batch_size=2, ppo_mini_batch_size=1)

assert batch["input_ids"].shape[0] == 2
assert config.max_new_tokens == 4
print("ppo smoke passed")
PY
