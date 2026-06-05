"""FASTA parsing and batch sampling utilities."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import random

from .tokenizer import DNATokenizer


def read_fasta(path: str | Path) -> Iterator[str]:
    """Yield normalized sequences from a FASTA file."""

    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    yield "".join(chunks).upper()
                    chunks.clear()
                continue
            chunks.append(line)
    if chunks:
        yield "".join(chunks).upper()


def load_token_stream(paths: Sequence[str | Path], tokenizer: DNATokenizer) -> list[int]:
    """Concatenate tokenized FASTA records into one language-modeling stream."""

    stream: list[int] = []
    for path in paths:
        for sequence in read_fasta(path):
            stream.extend(tokenizer.encode(sequence, add_special_tokens=True))
    if len(stream) < 2:
        raise ValueError("At least two tokens are required to build next-token examples")
    return stream


def split_stream(tokens: Sequence[int], validation_fraction: float = 0.05) -> tuple[list[int], list[int]]:
    """Split a token stream into train and validation partitions."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    split_at = max(1, int(len(tokens) * (1.0 - validation_fraction)))
    split_at = min(split_at, len(tokens) - 1)
    return list(tokens[:split_at]), list(tokens[split_at:])


def sample_batch(tokens: Sequence[int], block_size: int, batch_size: int, rng: random.Random) -> tuple[list[list[int]], list[list[int]]]:
    """Sample input/target windows for causal next-token prediction."""

    if len(tokens) <= block_size:
        raise ValueError("Token stream must be longer than block_size")
    starts = [rng.randint(0, len(tokens) - block_size - 1) for _ in range(batch_size)]
    inputs = [list(tokens[start : start + block_size]) for start in starts]
    targets = [list(tokens[start + 1 : start + block_size + 1]) for start in starts]
    return inputs, targets
