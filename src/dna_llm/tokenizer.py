"""A compact tokenizer for DNA sequence language modeling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DNATokenizer:
    """Character-level tokenizer for canonical and ambiguous nucleotide symbols.

    The vocabulary includes PAD/BOS/EOS special tokens plus IUPAC nucleotide codes.
    Unknown letters are mapped to ``N`` so public FASTA files with noisy bases can
    still be streamed into a model without dropping whole sequences.
    """

    alphabet: str = "ACGTNRYKMSWBDHV"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __post_init__(self) -> None:
        tokens = [self.pad_token, self.bos_token, self.eos_token, *self.alphabet]
        if len(tokens) != len(set(tokens)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens")
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "token_to_id", {token: idx for idx, token in enumerate(tokens)})
        object.__setattr__(self, "id_to_token", {idx: token for idx, token in enumerate(tokens)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> list[int]:
        """Convert a DNA sequence to token IDs."""

        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for char in sequence.upper():
            if char.isspace():
                continue
            ids.append(self.token_to_id.get(char, self.token_to_id["N"]))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to a DNA string."""

        special = {self.pad_id, self.bos_id, self.eos_id}
        chars: list[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id in special:
                continue
            token = self.id_to_token[int(token_id)]
            if token.startswith("<") and token.endswith(">"):
                continue
            chars.append(token)
        return "".join(chars)
