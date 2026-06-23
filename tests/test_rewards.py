import torch

from grpo import CompositeReward, RuleBasedReward
from reward_model import reward_loss


def test_reward_loss_prefers_higher_chosen_scores():
    chosen = torch.tensor([2.0, 1.0])
    rejected = torch.tensor([0.0, -1.0])

    loss, metrics = reward_loss(chosen, rejected)

    assert loss.item() > 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["reward_margin"] > 0.0


def test_rule_based_reward_scores_correct_answer_above_wrong_answer():
    reward_fn = RuleBasedReward(length_penalty=0.0)
    rewards = reward_fn(
        prompts=["2+3?", "2+3?"],
        completions=["The answer is 5.", "The answer is 6."],
        references=["5", "5"],
    )

    assert rewards[0] > rewards[1]


def test_composite_reward_applies_weights():
    base = RuleBasedReward(length_penalty=0.0)
    composite = CompositeReward([(base, 0.5), (base, 0.5)])

    single = base(["2+3?"], ["The answer is 5."], ["5"])
    combined = composite(["2+3?"], ["The answer is 5."], ["5"])

    assert torch.allclose(single, combined)
