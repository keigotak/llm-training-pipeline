import torch

from train import GPTModel, ModelConfig


def test_small_gpt_forward_returns_logits_and_loss():
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_model=16,
        d_ff=32,
        use_flash_attn=False,
        use_flash_attn_3=False,
        use_te=False,
    )
    model = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

    logits, loss = model(input_ids, labels=input_ids)

    assert logits.shape == (2, config.max_seq_len, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
