from pathlib import Path
import random

import pytest

from dna_llm.data import read_fasta, sample_batch, split_stream
from dna_llm.tokenizer import DNATokenizer


def test_read_fasta_concatenates_multiline_records(tmp_path: Path):
    fasta = tmp_path / "example.fa"
    fasta.write_text(">chr1\nacg\nta\n>chr2\nNN\n", encoding="utf-8")

    assert list(read_fasta(fasta)) == ["ACGTA", "NN"]


def test_split_stream_keeps_both_partitions_non_empty():
    train, val = split_stream(list(range(10)), validation_fraction=0.2)

    assert train == list(range(8))
    assert val == [8, 9]


def test_sample_batch_returns_shifted_targets():
    rng = random.Random(1)
    inputs, targets = sample_batch(list(range(20)), block_size=4, batch_size=2, rng=rng)

    assert len(inputs) == 2
    assert len(targets) == 2
    for x, y in zip(inputs, targets):
        assert x[1:] == y[:-1]


def test_sample_batch_requires_stream_longer_than_block_size():
    with pytest.raises(ValueError):
        sample_batch([1, 2, 3], block_size=3, batch_size=1, rng=random.Random(1))
