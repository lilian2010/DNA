"""A small GPT-style Transformer decoder for DNA token streams."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        mask = torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        output, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return output


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DNAGPT(nn.Module):
    """Causal language model for DNA sequences."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Cannot process sequence length {seq_len}; block_size is {self.config.block_size}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(batch_size * seq_len, -1), targets.reshape(batch_size * seq_len))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressively sample tokens from the model."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def cosine_lr(step: int, max_steps: int, warmup_steps: int, learning_rate: float) -> float:
    """Linear warmup followed by cosine decay."""

    if step < warmup_steps:
        return learning_rate * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
