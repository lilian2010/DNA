from dna_llm import DNATokenizer


def test_encode_decode_round_trip_canonical_bases():
    tokenizer = DNATokenizer()

    ids = tokenizer.encode("ACGT")

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "ACGT"


def test_unknown_bases_map_to_n_and_whitespace_is_ignored():
    tokenizer = DNATokenizer()

    assert tokenizer.decode(tokenizer.encode("a c x t")) == "ACNT"
