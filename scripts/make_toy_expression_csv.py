#!/usr/bin/env python
"""Create a tiny synthetic DNA + omics expression table for demos."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy_expression.csv")
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gene_id", "chrom", "start", "end", "sequence", "omics_atac", "omics_h3k27ac", "omics_rna_context", "target"],
        )
        writer.writeheader()
        for index in range(args.records):
            sequence = "".join(rng.choice("ACGT") for _ in range(args.length))
            gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
            atac = rng.random()
            h3k27ac = rng.random()
            rna_context = rng.random()
            target = 2.0 * gc_fraction + 1.5 * atac + 0.75 * h3k27ac + 0.25 * rna_context
            writer.writerow(
                {
                    "gene_id": f"gene_{index}",
                    "chrom": "chrToy",
                    "start": index * args.length,
                    "end": (index + 1) * args.length,
                    "sequence": sequence,
                    "omics_atac": f"{atac:.6f}",
                    "omics_h3k27ac": f"{h3k27ac:.6f}",
                    "omics_rna_context": f"{rna_context:.6f}",
                    "target": f"{target:.6f}",
                }
            )


if __name__ == "__main__":
    main()
