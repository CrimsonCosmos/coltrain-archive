#!/usr/bin/env python3
"""
Music Transformer v2
- Relative positional encoding (learns "4 beats apart" not "position 347")
- Optimized for compound tokens
- Sliding window generation for unlimited length
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with RELATIVE positional encoding.

    From Music Transformer (Huang et al., 2018):
    Instead of absolute position ("I'm at position 347"), the model learns
    relative distances ("this note is 10 tokens after that chord").

    This is critical for music because musical relationships are about
    relative timing, not absolute position.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Relative position embeddings
        # E_rel[i] = embedding for relative distance i
        # Range: [-max_seq_len+1, max_seq_len-1] but we only need [0, max_seq_len-1]
        # because of causal masking (can only look backward)
        self.rel_pos_embedding = nn.Embedding(max_seq_len, self.d_k)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Standard content-based attention scores
        content_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Relative position scores
        # For each query position i, compute score with relative positions 0..seq_len-1
        positions = torch.arange(seq_len, device=x.device)
        # Relative distance matrix: rel_dist[i,j] = i - j (clamped to valid range)
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        rel_dist = rel_dist.clamp(0, self.max_seq_len - 1)

        rel_embeddings = self.rel_pos_embedding(rel_dist)  # (seq_len, seq_len, d_k)

        # Compute relative attention: Q @ rel_embeddings
        # Q: (batch, heads, seq_len, d_k)
        # rel_embeddings: (seq_len, seq_len, d_k)
        rel_scores = torch.einsum('bhid,ijd->bhij', Q, rel_embeddings)

        # Combine content + relative scores
        scores = (content_scores + rel_scores) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


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

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm: normalize BEFORE attention (more stable)
        normed = self.norm1(x)
        x = x + self.dropout1(self.attention(normed, mask))

        normed = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(normed))

        return x


class MusicTransformer(nn.Module):
    """
    Music Transformer v2 with:
    - Relative positional encoding (musical distances, not absolute positions)
    - Pre-norm transformer blocks (stable training)
    - GELU activation (smoother gradients)
    - Compound token vocabulary
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding (no positional encoding - using relative attention instead!)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks with relative attention
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Weight tying: share embedding weights with output projection
        # This improves quality and reduces parameters
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

    def generate_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape

        if mask is None:
            mask = self.generate_causal_mask(seq_len, x.device)

        # Embed tokens (NO positional encoding - relative attention handles position)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm + project to vocab
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
        vocab_size=4739,  # Compound token vocab
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )

    x = torch.randint(0, 4739, (2, 64))
    print(f"\nInput: {x.shape}")
    logits = model(x)
    print(f"Output: {logits.shape}")

    generated = model.generate([1], max_len=50, temperature=1.0, top_k=50)
    print(f"Generated: {len(generated)} tokens")

    print("\nTest complete!")


if __name__ == '__main__':
    test_model()
