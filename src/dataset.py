"""Reusable token-stream preparation for language-model notebooks."""

import random
from pathlib import Path

from src.tokenizer import bpe_encode, load_tokenizer


def load_tiny_shakespeare_tokens(data_directory):
    """Load Tiny Shakespeare and encode it with the saved BPE tokenizer."""
    corpus_path = Path(data_directory) / "tiny_shakespeare.txt"
    tokenizer_path = Path(data_directory) / "tiny_shakespeare_bpe.json"

    corpus = corpus_path.read_text(encoding="utf-8")
    vocab, merges = load_tokenizer(tokenizer_path)
    tokens = bpe_encode(corpus, vocab, merges)
    return tokens, vocab, merges


def split_token_stream(tokens, train_fraction=0.9):
    """Split a token stream into contiguous training and validation portions."""
    split_index = int(train_fraction * len(tokens))
    return tokens[:split_index], tokens[split_index:]


def get_batch(split, train_tokens, validation_tokens, block_size, batch_size):
    """Sample next-token prediction windows from a token stream."""
    data = train_tokens if split == "train" else validation_tokens
    starts = [random.randrange(len(data) - block_size) for _ in range(batch_size)]
    x = [data[index : index + block_size] for index in starts]
    y = [data[index + 1 : index + block_size + 1] for index in starts]
    return x, y
