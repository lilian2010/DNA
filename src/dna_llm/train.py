"""Command-line training loop for DNA GPT pretraining."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tqdm import trange

from .config import ModelConfig, TrainingConfig
from .data import load_token_stream, sample_batch, split_stream
from .model import DNAGPT, cosine_lr
from .tokenizer import DNATokenizer


def choose_device(requested: str) -> str:
    """Resolve an automatic or explicit torch device string."""

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a GPT-style LLM on FASTA DNA sequences.")
    parser.add_argument("--fasta", nargs="+", required=True, help="One or more FASTA files.")
    parser.add_argument("--out", default="checkpoints/dna_gpt.pt", help="Checkpoint output path.")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_tensor_batch(tokens: list[int], config: ModelConfig, training: TrainingConfig, rng: random.Random, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = sample_batch(tokens, config.block_size, training.batch_size, rng)
    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_loss(model: DNAGPT, tokens: list[int], config: ModelConfig, training: TrainingConfig, rng: random.Random, device: str) -> float:
    """Average validation loss over several random batches."""

    model.eval()
    losses: list[float] = []
    for _ in range(training.eval_batches):
        x, y = make_tensor_batch(tokens, config, training, rng, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> Path:
    tokenizer = DNATokenizer()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
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

    stream = load_token_stream(args.fasta, tokenizer)
    train_tokens, val_tokens = split_stream(stream, args.validation_fraction)
    model = DNAGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

    progress = trange(training_config.max_steps, desc="pretraining")
    for step in progress:
        lr = cosine_lr(step, training_config.max_steps, training_config.warmup_steps, training_config.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = make_tensor_batch(train_tokens, model_config, training_config, rng, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        if (step + 1) % training_config.eval_interval == 0:
            val_loss = estimate_loss(model, val_tokens, model_config, training_config, rng, device)
            progress.write(f"step {step + 1}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

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
