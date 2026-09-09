"""Attention blocks and language models with selectable positional encodings."""

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Compute the fixed position vectors from Attention Is All You Need, section 3.5."""

    def __init__(self, d_model):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.d_model = d_model

        # Features 2i and 2i+1 share the denominator 10000 ** (2i / d_model).
        even_feature_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        wavelength_scales = 10000 ** (even_feature_indices / d_model)
        # A buffer follows device/dtype moves and checkpoints, but is not optimized.
        self.register_buffer("wavelength_scales", wavelength_scales)

    def forward(self, positions):
        """Map position ids [...shape] to vectors [...shape, d_model]."""
        angles = positions.unsqueeze(-1).to(self.wavelength_scales.dtype)
        angles = angles / self.wavelength_scales
        encodings = angles.new_empty(*positions.shape, self.d_model)
        encodings[..., 0::2] = torch.sin(angles)
        # Odd widths have one more sine feature than cosine features.
        cosine_feature_count = self.d_model // 2
        encodings[..., 1::2] = torch.cos(angles[..., :cosine_feature_count])
        return encodings


class NoPositionalEncoding(nn.Module):
    """Add zero position vectors for an explicit-position ablation."""

    def __init__(self, d_model):
        super().__init__()
        self.register_buffer("zero_vector", torch.zeros(d_model))

    def forward(self, positions):
        return self.zero_vector.expand(*positions.shape, -1)


def make_position_encoding(position_encoding, block_size, d_model):
    """Keep learned tables as the baseline; fixed sinusoids need no length-sized table."""
    if position_encoding == "learned":
        return nn.Embedding(block_size, d_model)
    if position_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model)
    if position_encoding == "none":
        return NoPositionalEncoding(d_model)
    raise ValueError("position_encoding must be 'learned', 'sinusoidal', or 'none'")


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.output_projection = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, query, key, value):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        B, len_q, _ = query.shape
        _, len_k, _ = key.shape
        _, len_v, _ = value.shape

        q = self.w_qs(query).view(B, len_q, n_head, d_k)
        k = self.w_ks(key).view(B, len_k, n_head, d_k)
        v = self.w_vs(value).view(B, len_v, n_head, d_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled dot-product attention
        # q @ k^T: (B, h, T, d_k) @ (B, h, d_k, T)
        #          -> (B, h, T, T)
        scores = q @ k.transpose(-2, -1)
        scores = scores / (d_k**0.5)

        # causal mask
        causal_mask = torch.triu(
            torch.ones(len_q, len_k, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # normalize, then retrieve values
        weights = torch.softmax(scores, dim=-1)  # (B, h, T, T)
        output = weights @ v  # (B, h, T, d_v)

        # put heads beside each other again
        output = output.transpose(1, 2).contiguous()  # (B, T, h, d_v)
        output = output.view(B, len_q, n_head * d_v)  # (B, T, h*d_v)
        output = self.output_projection(output)  # (B, T, C)

        return output, weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, x):
        # attention residual
        z = self.ln1(x)
        attn_out, weights = self.attn(z, z, z)
        x = x + attn_out

        # feed-forward residual
        x = x + self.ff(self.ln2(x))

        return x, weights


class TransformerStack(nn.Module):
    """Compose independent blocks; return embeddings and attention maps in layer order."""

    def __init__(self, n_layers, d_model, n_head, d_k, d_v, d_ff):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_k, d_v, d_ff) for _ in range(n_layers)]
        )

    def forward(self, x):
        # x stays [batch, tokens, d_model] throughout the stack.
        attention_weights = []
        for block in self.blocks:
            x, weights = block(x)
            attention_weights.append(weights)
        return x, attention_weights


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class SingleTransformerLanguageModel(nn.Module):
    """Add selectable positions, one pre-norm block, and final norm to the bigram path."""

    def __init__(
        self, vocab_size, n_embd=32, block_size=8, num_heads=2, *, position_encoding="learned"
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = make_position_encoding(
            position_encoding, block_size, n_embd
        )
        self.block = TransformerBlock(
            n_embd, num_heads, n_embd // num_heads, n_embd // num_heads, 4 * n_embd
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(positions)
        x, _ = self.block(x)
        return self.lm_head(self.final_norm(x))


class StackedTransformerLanguageModel(nn.Module):
    """Map token ids to vocabulary logits through a configurable block stack."""

    def __init__(
        self,
        vocab_size,
        n_layers=2,
        n_embd=32,
        block_size=8,
        num_heads=2,
        *,
        position_encoding="learned",
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = make_position_encoding(
            position_encoding, block_size, n_embd
        )
        self.stack = TransformerStack(
            n_layers=n_layers,
            d_model=n_embd,
            n_head=num_heads,
            d_k=n_embd // num_heads,
            d_v=n_embd // num_heads,
            d_ff=4 * n_embd,
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, token_ids):
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        embeddings = self.token_embedding_table(token_ids)
        embeddings = embeddings + self.position_embedding_table(positions)
        representations, _ = self.stack(embeddings)
        return self.lm_head(self.final_norm(representations))
