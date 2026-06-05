"""Multimodal models for gene-expression prediction from DNA and omics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ExpressionModelConfig


class SequencePoolingEncoder(nn.Module):
    """Embed DNA tokens and mean-pool non-padding positions."""

    def __init__(self, config: ExpressionModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.sequence_embedding_dim, padding_idx=config.pad_token_id)
        self.projection = nn.Sequential(
            nn.LayerNorm(config.sequence_embedding_dim),
            nn.Linear(config.sequence_embedding_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.pad_token_id = config.pad_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.pad_token_id).unsqueeze(-1)
        embeddings = self.embedding(input_ids) * mask
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = embeddings.sum(dim=1) / lengths
        return self.projection(pooled)


class OmicsEncoder(nn.Module):
    """Project dense omics covariates into the shared fusion space."""

    def __init__(self, config: ExpressionModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(config.omics_dim),
            nn.Linear(config.omics_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

    def forward(self, omics: torch.Tensor) -> torch.Tensor:
        return self.network(omics)


class MultimodalExpressionPredictor(nn.Module):
    """Predict gene expression by fusing DNA sequence and omics features."""

    def __init__(self, config: ExpressionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.sequence_encoder = SequencePoolingEncoder(config)
        self.omics_encoder = OmicsEncoder(config)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, input_ids: torch.Tensor, omics: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence_features = self.sequence_encoder(input_ids)
        omics_features = self.omics_encoder(omics)
        predictions = self.fusion(torch.cat((sequence_features, omics_features), dim=-1)).squeeze(-1)
        loss = None
        if targets is not None:
            loss = F.mse_loss(predictions, targets)
        return predictions, loss

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
