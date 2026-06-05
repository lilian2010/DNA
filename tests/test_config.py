import pytest

from dna_llm.config import ExpressionModelConfig, ModelConfig, TrainingConfig


def test_model_config_requires_embedding_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=8, n_embd=10, n_head=3)


def test_training_config_rejects_negative_weight_decay():
    with pytest.raises(ValueError, match="negative"):
        TrainingConfig(weight_decay=-0.1)


def test_expression_model_config_requires_omics_dimension():
    with pytest.raises(ValueError, match="omics_dim"):
        ExpressionModelConfig(vocab_size=8, omics_dim=0)
