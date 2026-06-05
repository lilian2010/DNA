#!/usr/bin/env python
"""Create a tiny synthetic FASTA file for smoke tests and demos."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy.fa")
    parser.add_argument("--records", type=int, default=32)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(args.records):
            handle.write(f">synthetic_{index}\n")
            sequence = "".join(rng.choice("ACGT") for _ in range(args.length))
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


if __name__ == "__main__":
    main()
