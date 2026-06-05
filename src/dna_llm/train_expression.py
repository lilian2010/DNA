"""Command-line training loop for multimodal gene-expression prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from tqdm import trange

from .config import ExpressionModelConfig, TrainingConfig
from .expression_data import GeneExpressionExample, omics_dimension, read_expression_csv, split_examples
from .expression_model import MultimodalExpressionPredictor
from .tokenizer import DNATokenizer
from .model import cosine_lr
from .train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal DNA + omics expression predictor.")
    parser.add_argument("--csv", required=True, help="CSV with gene_id, sequence, target, and omics_* columns.")
    parser.add_argument("--out", default="checkpoints/expression_predictor.pt", help="Checkpoint output path.")
    parser.add_argument("--omics-prefix", default="omics_", help="Prefix used to discover omics feature columns.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--sequence-embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=25)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def encode_sequence(sequence: str, tokenizer: DNATokenizer, block_size: int) -> list[int]:
    """Encode, truncate, and pad one sequence to a fixed context length."""

    ids = tokenizer.encode(sequence, add_special_tokens=True)[:block_size]
    return ids + [tokenizer.pad_id] * (block_size - len(ids))


def sample_examples(examples: list[GeneExpressionExample], batch_size: int, rng: random.Random) -> list[GeneExpressionExample]:
    """Sample a mini-batch with replacement."""

    if not examples:
        raise ValueError("Cannot sample from an empty example set")
    return [examples[rng.randrange(len(examples))] for _ in range(batch_size)]


def make_tensor_batch(
    examples: list[GeneExpressionExample],
    tokenizer: DNATokenizer,
    config: ExpressionModelConfig,
    batch_size: int,
    rng: random.Random,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = sample_examples(examples, batch_size, rng)
    input_ids = torch.tensor([encode_sequence(example.sequence, tokenizer, config.block_size) for example in batch], dtype=torch.long, device=device)
    omics = torch.tensor([example.omics for example in batch], dtype=torch.float32, device=device)
    targets = torch.tensor([example.target for example in batch], dtype=torch.float32, device=device)
    return input_ids, omics, targets


@torch.no_grad()
def estimate_mse(
    model: MultimodalExpressionPredictor,
    examples: list[GeneExpressionExample],
    tokenizer: DNATokenizer,
    model_config: ExpressionModelConfig,
    training_config: TrainingConfig,
    rng: random.Random,
    device: str,
) -> float:
    """Estimate validation MSE from random mini-batches."""

    model.eval()
    losses: list[float] = []
    for _ in range(training_config.eval_batches):
        input_ids, omics, targets = make_tensor_batch(examples, tokenizer, model_config, training_config.batch_size, rng, device)
        _, loss = model(input_ids, omics, targets)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> Path:
    tokenizer = DNATokenizer()
    examples = list(read_expression_csv(args.csv, omics_prefix=args.omics_prefix))
    train_examples, validation_examples = split_examples(examples, validation_fraction=args.validation_fraction)
    model_config = ExpressionModelConfig(
        vocab_size=tokenizer.vocab_size,
        omics_dim=omics_dimension(examples),
        pad_token_id=tokenizer.pad_id,
        block_size=args.block_size,
        sequence_embedding_dim=args.sequence_embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
    )
    rng = random.Random(training_config.seed)
    torch.manual_seed(training_config.seed)
    device = choose_device(training_config.device)
    model = MultimodalExpressionPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

    progress = trange(training_config.max_steps, desc="expression-training")
    for step in progress:
        lr = cosine_lr(step, training_config.max_steps, training_config.warmup_steps, training_config.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        input_ids, omics, targets = make_tensor_batch(train_examples, tokenizer, model_config, training_config.batch_size, rng, device)
        _, loss = model(input_ids, omics, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        optimizer.step()
        progress.set_postfix(mse=f"{loss.item():.4f}", lr=f"{lr:.2e}")
        if (step + 1) % training_config.eval_interval == 0:
            val_mse = estimate_mse(model, validation_examples, tokenizer, model_config, training_config, rng, device)
            progress.write(f"step {step + 1}: train_mse={loss.item():.4f} val_mse={val_mse:.4f}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "tokenizer_tokens": tokenizer.tokens,
        },
        output_path,
    )
    return output_path


def main() -> None:
    output_path = train(parse_args())
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
