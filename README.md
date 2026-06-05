# Pretraining a Large Language Model (LLM) from Scratch on DNA Sequences

This repository provides a compact, end-to-end scaffold for genome foundation-model research and downstream multimodal expression prediction. It supports pretraining a GPT-style causal language model on DNA FASTA files and training supervised models that integrate DNA sequence with omics covariates. It includes:

- A DNA tokenizer with IUPAC nucleotide support.
- FASTA parsing and random next-token batch sampling.
- A decoder-only Transformer implemented in PyTorch.
- A command-line pretraining loop with cosine learning-rate decay, validation loss estimation, gradient clipping, and checkpoint export.
- A supervised DNA + omics expression-prediction baseline.
- Unit tests for tokenizer, configuration, expression data, and data utilities.

## Why causal language modeling for DNA?

DNA sequences can be represented as token streams over nucleotide symbols. A causal model learns to predict the next token from preceding context, encouraging the Transformer to discover motifs, repeats, local syntax, and long-range dependencies directly from sequence data.

## Project layout

```text
src/dna_llm/
  config.py      Model and training dataclasses
  data.py        FASTA loading, stream splitting, and batch sampling
  model.py       GPT-style Transformer decoder
  tokenizer.py   DNA/IUPAC tokenizer
  train.py              CLI pretraining loop
  expression_data.py    CSV loading for DNA + omics expression tables
  expression_model.py   Multimodal expression predictor
  train_expression.py   CLI expression training loop
configs/
  small.yaml            Example small-run pretraining hyperparameters
  expression.yaml       Example supervised expression hyperparameters
scripts/
  make_toy_fasta.py           Synthetic FASTA generator
  make_toy_expression_csv.py  Synthetic DNA + omics table generator
tests/           Unit tests
```

## Installation

Clone the repository to your local machine and enter the project directory:

```bash
git clone <repository-url> DNA
cd DNA
```

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

PyTorch wheels are platform-specific. If the generic install does not select the best CUDA or CPU build for your machine, install the recommended wheel from the official PyTorch selector and then rerun the editable install.

## Quickstart

Create a small synthetic FASTA file:

```bash
python scripts/make_toy_fasta.py --out data/toy.fa --records 64 --length 512
```

Run a tiny smoke-training job:

```bash
dna-llm-pretrain \
  --fasta data/toy.fa \
  --out checkpoints/toy.pt \
  --block-size 64 \
  --n-layer 2 \
  --n-head 2 \
  --n-embd 64 \
  --batch-size 8 \
  --max-steps 100 \
  --eval-interval 50 \
  --eval-batches 4
```

For real pretraining, replace `data/toy.fa` with curated genome, contig, transcript, or metagenomic FASTA files and scale `block_size`, model depth, batch size, and training steps to your compute budget.


## Multimodal expression prediction

Chromatin and expression modeling often needs more than raw sequence. The expression CLI expects a CSV with required columns:

- `gene_id`
- `sequence`
- `target`
- one or more omics feature columns prefixed with `omics_` by default, such as `omics_atac`, `omics_h3k27ac`, or `omics_rna_context`

Optional genomic coordinate columns `chrom`, `start`, and `end` are preserved by the loader. The split utility groups by `gene_id` so repeated rows for the same locus do not leak across train and validation.

Create a synthetic multimodal expression table:

```bash
python scripts/make_toy_expression_csv.py --out data/toy_expression.csv --records 256 --length 256
```

Run a tiny supervised training job:

```bash
dna-llm-train-expression \
  --csv data/toy_expression.csv \
  --out checkpoints/toy_expression.pt \
  --block-size 256 \
  --sequence-embedding-dim 64 \
  --hidden-dim 128 \
  --batch-size 16 \
  --max-steps 200 \
  --eval-interval 50 \
  --eval-batches 4
```

The current supervised model is intentionally lightweight: it embeds DNA tokens, mean-pools non-padding positions, encodes dense omics vectors with an MLP, fuses both modalities, and predicts scalar expression with mean-squared error. This creates a clear baseline that can later be replaced with a pretrained sequence encoder, tissue-specific heads, cross-attention over assays, or contrastive pretraining objectives.

## Data recommendations

1. Deduplicate highly similar records to reduce memorization.
2. Track species, chromosome/contig, assembly, and train/validation split metadata.
3. Reserve whole chromosomes, contigs, taxa, or samples for validation when you need a stronger generalization estimate.
4. Keep ambiguous IUPAC bases when they matter biologically; otherwise normalize or filter records before pretraining.

## Checkpoints

The training CLI saves a PyTorch checkpoint containing:

- `model_state_dict`
- `model_config`
- `training_config`
- `tokenizer_tokens`

For expression checkpoints, the saved model state corresponds to `MultimodalExpressionPredictor`; for DNA pretraining checkpoints, it corresponds to `DNAGPT`.

This is intentionally simple so downstream fine-tuning, probing, and sequence generation scripts can load the exact architecture and tokenizer vocabulary used during pretraining.

## Testing

```bash
pytest
```
