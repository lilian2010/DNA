from pathlib import Path

import pytest

from dna_llm.expression_data import omics_dimension, read_expression_csv, split_examples


def test_read_expression_csv_discovers_omics_columns(tmp_path: Path):
    csv_path = tmp_path / "expression.csv"
    csv_path.write_text(
        "gene_id,chrom,start,end,sequence,omics_atac,omics_h3k27ac,target\n"
        "gene_a,chr1,10,14,acgt,0.5,1.5,2.0\n",
        encoding="utf-8",
    )

    examples = list(read_expression_csv(csv_path))

    assert len(examples) == 1
    assert examples[0].gene_id == "gene_a"
    assert examples[0].sequence == "ACGT"
    assert examples[0].omics == (0.5, 1.5)
    assert examples[0].target == 2.0
    assert examples[0].start == 10
    assert examples[0].end == 14


def test_read_expression_csv_requires_omics_columns(tmp_path: Path):
    csv_path = tmp_path / "expression.csv"
    csv_path.write_text("gene_id,sequence,target\ngene_a,ACGT,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="omics feature"):
        list(read_expression_csv(csv_path))


def test_split_examples_keeps_gene_ids_disjoint(tmp_path: Path):
    csv_path = tmp_path / "expression.csv"
    csv_path.write_text(
        "gene_id,sequence,omics_atac,target\n"
        "gene_a,AAAA,0.1,1.0\n"
        "gene_a,AAAT,0.2,1.1\n"
        "gene_b,CCCC,0.3,2.0\n"
        "gene_c,GGGG,0.4,3.0\n",
        encoding="utf-8",
    )
    examples = list(read_expression_csv(csv_path))

    train, validation = split_examples(examples, validation_fraction=0.34, seed=7)

    train_genes = {example.gene_id for example in train}
    validation_genes = {example.gene_id for example in validation}
    assert train_genes.isdisjoint(validation_genes)
    assert train
    assert validation


def test_omics_dimension_validates_consistent_dimensions(tmp_path: Path):
    csv_path = tmp_path / "expression.csv"
    csv_path.write_text(
        "gene_id,sequence,omics_atac,omics_h3k27ac,target\n"
        "gene_a,AAAA,0.1,0.2,1.0\n"
        "gene_b,CCCC,0.3,0.4,2.0\n",
        encoding="utf-8",
    )

    assert omics_dimension(read_expression_csv(csv_path)) == 2
