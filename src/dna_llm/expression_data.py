"""Utilities for multimodal gene-expression prediction datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class GeneExpressionExample:
    """One supervised example linking sequence, omics covariates, and expression."""

    gene_id: str
    sequence: str
    omics: tuple[float, ...]
    target: float
    chrom: str | None = None
    start: int | None = None
    end: int | None = None


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def read_expression_csv(path: str | Path, *, omics_prefix: str = "omics_") -> Iterator[GeneExpressionExample]:
    """Yield expression examples from a CSV table.

    Required columns are ``gene_id``, ``sequence``, and ``target``. Omics features
    are discovered from columns whose names start with ``omics_prefix`` and are
    emitted in the input column order so data preparation remains explicit and
    reproducible.
    """

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Expression CSV is missing a header row")
        required = {"gene_id", "sequence", "target"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Expression CSV is missing required columns: {sorted(missing)}")
        omics_columns = [name for name in reader.fieldnames if name.startswith(omics_prefix)]
        if not omics_columns:
            raise ValueError(f"Expression CSV must include at least one '{omics_prefix}*' omics feature column")

        for row in reader:
            yield GeneExpressionExample(
                gene_id=row["gene_id"],
                sequence=row["sequence"].upper(),
                omics=tuple(float(row[name]) for name in omics_columns),
                target=float(row["target"]),
                chrom=row.get("chrom") or None,
                start=_optional_int(row.get("start")),
                end=_optional_int(row.get("end")),
            )


def split_examples(
    examples: Sequence[GeneExpressionExample],
    *,
    validation_fraction: float = 0.1,
    seed: int = 1337,
) -> tuple[list[GeneExpressionExample], list[GeneExpressionExample]]:
    """Deterministically split examples by shuffled gene IDs.

    Grouping by gene ID prevents duplicated gene rows across train and validation
    when the same locus has multiple omics contexts, tissues, or perturbations.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(examples) < 2:
        raise ValueError("At least two examples are required for a validation split")

    gene_ids = sorted({example.gene_id for example in examples})
    if len(gene_ids) < 2:
        raise ValueError("At least two distinct gene IDs are required for a validation split")
    rng = random.Random(seed)
    rng.shuffle(gene_ids)
    validation_gene_count = max(1, int(round(len(gene_ids) * validation_fraction)))
    validation_gene_count = min(validation_gene_count, len(gene_ids) - 1)
    validation_genes = set(gene_ids[:validation_gene_count])

    train = [example for example in examples if example.gene_id not in validation_genes]
    validation = [example for example in examples if example.gene_id in validation_genes]
    return train, validation


def omics_dimension(examples: Iterable[GeneExpressionExample]) -> int:
    """Return and validate the shared omics feature dimension."""

    iterator = iter(examples)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot infer omics dimension from an empty dataset") from exc
    dimension = len(first.omics)
    for example in iterator:
        if len(example.omics) != dimension:
            raise ValueError("All examples must have the same omics feature dimension")
    return dimension
