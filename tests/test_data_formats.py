import json

from dpo import DPODataset
from grpo import GRPOPromptDataset
from reward_model import PreferenceDataset
from sft import SFTDataset


class TinyTokenizer:
    pad_id = 0

    def encode_chat(self, messages):
        input_ids = [1]
        loss_mask = [0]
        for message in messages:
            content = message["content"]
            tokens = [2 + (ord(char) % 20) for char in content]
            input_ids.extend(tokens)
            loss_mask.extend([1 if message["role"] == "assistant" else 0] * len(tokens))
        input_ids.append(2)
        loss_mask.append(1)
        return {"input_ids": input_ids, "loss_mask": loss_mask}


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_sft_jsonl_format_loads(tmp_path):
    path = tmp_path / "sft.jsonl"
    write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        ],
    )

    dataset = SFTDataset(str(path), TinyTokenizer(), max_seq_len=16)
    item = dataset[0]

    assert len(dataset) == 1
    assert item["input_ids"].shape == item["labels"].shape
    assert (item["labels"] != -100).any()


def test_preference_jsonl_formats_load(tmp_path):
    path = tmp_path / "prefs.jsonl"
    rows = [
        {
            "chosen": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "rejected": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "5"},
            ],
        }
    ]
    write_jsonl(path, rows)

    dpo_dataset = DPODataset(str(path), TinyTokenizer(), max_seq_len=16)
    reward_dataset = PreferenceDataset(str(path), TinyTokenizer(), max_seq_len=16)

    assert len(dpo_dataset) == 1
    assert len(reward_dataset) == 1
    assert dpo_dataset[0]["chosen_ids"].ndim == 1
    assert reward_dataset[0]["chosen_ids"].ndim == 1


def test_grpo_jsonl_format_loads(tmp_path):
    path = tmp_path / "grpo.jsonl"
    write_jsonl(path, [{"prompt": "What is 2+3?", "answer": "5"}])

    dataset = GRPOPromptDataset(str(path), TinyTokenizer(), max_prompt_len=16)
    item = dataset[0]

    assert len(dataset) == 1
    assert item["answer"] == "5"
    assert item["prompt_text"] == "What is 2+3?"
    assert item["input_ids"].ndim == 1
