"""Configuration dataclasses for DNA language-model pretraining."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Transformer decoder architecture settings."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and data-loading settings."""

    batch_size: int = 32
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_batches: int = 50
    seed: int = 1337
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.warmup_steps < 0 or self.eval_interval <= 0 or self.eval_batches <= 0:
            raise ValueError("warmup_steps must be non-negative; eval settings must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


@dataclass(frozen=True)
class ExpressionModelConfig:
    """Multimodal architecture settings for expression prediction."""

    vocab_size: int
    omics_dim: int
    pad_token_id: int = 0
    block_size: int = 512
    sequence_embedding_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.omics_dim <= 0:
            raise ValueError("vocab_size and omics_dim must be positive")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id cannot be negative")
        if self.block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        if self.sequence_embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("sequence_embedding_dim and hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
