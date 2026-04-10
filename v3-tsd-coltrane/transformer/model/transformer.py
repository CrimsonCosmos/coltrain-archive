#!/usr/bin/env python3
"""
Music Transformer v2
- Learned positional encoding with Flash/Memory-Efficient attention (SDPA)
- Optimized for fast training on GPU
- Sliding window generation for unlimited length
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention using PyTorch's scaled_dot_product_attention.

    Uses Flash Attention (Ampere+) or Memory-Efficient Attention (Turing/T4)
    automatically — avoids materializing the full O(seq_len^2) attention matrix.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout_p = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)


class FeedForward(nn.Module):
    """Position-wise FFN with GELU activation (better than ReLU for music)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (more stable training than post-norm)."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        x = x + self.dropout1(self.attention(normed, mask))

        normed = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(normed))

        return x


class MusicTransformer(nn.Module):
    """
    Music Transformer v2 with:
    - Learned positional embeddings + SDPA (Flash/Memory-Efficient Attention)
    - Pre-norm transformer blocks (stable training)
    - GELU activation (smoother gradients)
    - Weight tying (embedding ↔ output)
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

        param_count = self.count_parameters()
        print(f"MusicTransformer v2:")
        print(f"  Vocab: {vocab_size}, Dim: {d_model}, Heads: {num_heads}")
        print(f"  Layers: {num_layers}, FFN: {d_ff}, Context: {max_seq_len}")
        print(f"  Parameters: {param_count:,}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits

    @torch.no_grad()
    def generate(self, start_tokens, max_len=2000, temperature=1.0,
                 top_k=None, top_p=None, device='cpu'):
        """
        Autoregressive generation with sliding context window.
        Every new token sees the previous max_seq_len tokens via attention.
        """
        self.eval()
        generated = list(start_tokens)

        for _ in range(max_len - len(generated)):
            # Sliding window: always use maximum available context
            window = generated[-self.max_seq_len:]
            x = torch.tensor([window], dtype=torch.long, device=device)

            logits = self(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = 0
                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b][remove[b]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == 2:  # END
                break

            generated.append(next_token)

        return generated


def test_model():
    print("Testing MusicTransformer v2")
    print("=" * 50)

    model = MusicTransformer(
        vocab_size=427,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )

    x = torch.randint(0, 427, (2, 64))
    print(f"\nInput: {x.shape}")
    logits = model(x)
    print(f"Output: {logits.shape}")

    generated = model.generate([1], max_len=50, temperature=1.0, top_k=50)
    print(f"Generated: {len(generated)} tokens")

    print("\nTest complete!")


if __name__ == '__main__':
    test_model()
