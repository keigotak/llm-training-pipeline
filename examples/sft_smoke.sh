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

python - "$tmpdir/sft_train.jsonl" <<'PY'
import sys

from sft import ChatTokenizer, SFTCollator, SFTDataset


data_path = sys.argv[1]
tokenizer = ChatTokenizer()
dataset = SFTDataset(data_path, tokenizer, max_seq_len=64)
collator = SFTCollator(tokenizer.pad_id, max_seq_len=64)
batch = collator([dataset[0], dataset[1]])

assert batch["input_ids"].shape == batch["labels"].shape
assert batch["attention_mask"].shape == batch["input_ids"].shape
assert (batch["labels"] != -100).any()
print("sft smoke passed")
PY
