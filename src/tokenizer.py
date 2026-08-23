"""A minimal byte-pair encoding tokenizer."""

import json
from pathlib import Path


def encode_unicode_string(unicode_string):
    return list(unicode_string.encode("utf-8"))


def decode_unicode_tokens(tokens):
    return bytes(tokens).decode("utf-8")


def merge_pair(tokens, pair, new_token):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def get_pair_frequencies(tokens):
    frequencies = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        frequencies[pair] = frequencies.get(pair, 0) + 1
    return frequencies


def perform_merges(tokens, target_vocab_size):
    starting_vocab_size = 256
    vocab = {i: bytes([i]) for i in range(starting_vocab_size)}
    merges = {}

    while target_vocab_size > starting_vocab_size:
        frequencies = get_pair_frequencies(tokens)
        if not frequencies:
            break

        best_pair = max(frequencies, key=frequencies.get)
        tokens = merge_pair(tokens, best_pair, starting_vocab_size)
        vocab[starting_vocab_size] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges[best_pair] = starting_vocab_size
        starting_vocab_size += 1

    return tokens, vocab, merges


def bpe_encode(text, vocab, merges):
    tokens = encode_unicode_string(text)
    for pair, new_token in merges.items():
        tokens = merge_pair(tokens, pair, new_token)
    return tokens


def bpe_decode(tokens, vocab):
    return b"".join(vocab[token] for token in tokens).decode("utf-8")


def save_tokenizer(path, vocab, merges):
    artifact = {
        "vocab": {str(token): value.hex() for token, value in vocab.items()},
        "merges": [[first, second, token] for (first, second), token in merges.items()],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact), encoding="utf-8")


def load_tokenizer(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {int(token): bytes.fromhex(value) for token, value in artifact["vocab"].items()}
    merges = {(first, second): token for first, second, token in artifact["merges"]}
    return vocab, merges
