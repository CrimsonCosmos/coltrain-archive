#!/usr/bin/env python3
"""
Enhanced generation with:
  1. Melody forcing (high floor for M tokens)
  2. Repetition penalty (break out of loops)
  3. Longer multi-track seed (500 tokens)
  4. Track-aware sampling (prevents single-track collapse)
"""
import torch
import torch.nn.functional as F
import sys
import os
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent / 'transformer'))

from model.transformer import MusicTransformer
from data.tokenizer import MIDITokenizer


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_seed_tokens(tokenizer, jazz_dir, min_tracks=3, seed_len=500, rng_seed=42):
    """Get seed tokens from a real multi-track jazz file."""
    files = [f for f in os.listdir(jazz_dir) if f.endswith('.mid')]
    random.seed(rng_seed)
    random.shuffle(files)

    for f in files[:300]:
        try:
            tokens = tokenizer.tokenize_midi(os.path.join(jazz_dir, f))
            types = set()
            for t in tokens:
                prefix = tokenizer.id_to_token[t].split('_')[0]
                if prefix in ('M', 'P', 'B', 'D'):
                    types.add(prefix)
            if len(types) >= min_tracks and len(tokens) > seed_len:
                print(f"Seed: {f} ({len(tokens)} tokens, tracks: {types})")
                return tokens[:seed_len], f
        except Exception:
            pass
    return None, None


def build_track_masks(tokenizer, device='cpu'):
    """Pre-compute boolean masks for each track type."""
    masks = {}
    for track in ['M', 'P', 'B', 'D', 'T']:
        mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
        for tid in range(tokenizer.vocab_size):
            name = tokenizer.id_to_token[tid]
            if name.split('_')[0] == track:
                mask[tid] = True
        masks[track] = mask
    return masks


def generate_v2(model, tokenizer, start_tokens, max_len=2000,
                temperature=0.95, top_k=80, top_p=0.95,
                track_floors=None, rep_penalty=1.2, rep_window=64,
                device='cpu'):
    """
    Enhanced generation with melody forcing, repetition penalty,
    and track-aware rebalancing.

    track_floors: dict of minimum probability mass per track type.
        e.g. {'M': 0.08, 'P': 0.10, 'B': 0.08, 'D': 0.05}
    rep_penalty: multiplicative penalty for tokens seen in last rep_window steps.
    rep_window: how many recent tokens to penalize.
    """
    if track_floors is None:
        track_floors = {'M': 0.08, 'P': 0.10, 'B': 0.08, 'D': 0.05}

    model.eval()
    generated = list(start_tokens)
    masks = build_track_masks(tokenizer, device=device)

    for step in range(max_len - len(generated)):
        window = generated[-model.max_seq_len:]
        x = torch.tensor([window], dtype=torch.long, device=device)

        logits = model(x)
        logits = logits[:, -1, :]  # (1, vocab)

        # --- Repetition penalty ---
        if rep_penalty > 1.0:
            recent = set(generated[-rep_window:])
            for tid in recent:
                if logits[0, tid] > 0:
                    logits[0, tid] /= rep_penalty
                else:
                    logits[0, tid] *= rep_penalty

        # Apply temperature
        logits = logits / temperature

        # --- Track-aware rebalancing ---
        probs = F.softmax(logits, dim=-1).squeeze()
        track_probs = {}
        for track in ['M', 'P', 'B', 'D']:
            track_probs[track] = probs[masks[track]].sum().item()

        # Boost any track below its floor
        needs_boost = False
        for track, floor in track_floors.items():
            if track_probs[track] < floor:
                needs_boost = True
                break

        if needs_boost:
            for track, floor in track_floors.items():
                if track_probs[track] < floor:
                    deficit = floor - track_probs[track]
                    bonus = 2.0 + deficit * 30
                    logits[0, masks[track]] += bonus

        # --- Top-k filtering ---
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        # --- Top-p filtering ---
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = 0
        logits[0, sorted_indices[0][remove[0]]] = float('-inf')

        probs_final = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs_final, num_samples=1).item()

        if next_token == 2:  # END
            break

        generated.append(next_token)

    return generated


def _report(tokenizer, tokens, seed_len=0):
    """Print track distribution for generated portion only."""
    gen_tokens = tokens[seed_len:]
    types = Counter()
    for t in gen_tokens:
        prefix = tokenizer.id_to_token[t].split('_')[0]
        types[prefix] += 1
    total = len(gen_tokens)
    parts = [f"{k}={v}({100*v/max(total,1):.0f}%)" for k, v in types.most_common()]
    print(f"    Generated {total} tokens: {', '.join(parts)}", flush=True)


def main():
    device = get_device()
    print(f"Device: {device}", flush=True)
    tokenizer = MIDITokenizer()
    jazz_dir = str(Path(__file__).parent / '..' / 'augmented_dataset_v2')

    # Longer seed: 500 tokens (~25 seconds of multi-track context)
    seed_tokens, seed_file = get_seed_tokens(tokenizer, jazz_dir, min_tracks=3, seed_len=500)
    if seed_tokens is None:
        print("ERROR: Could not find a suitable multi-track seed file")
        return

    ckpt_path = str(Path(__file__).parent / 'checkpoints_finetune' / 'finetune_epoch1.pt')
    print(f"\nLoading finetune_epoch1...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.get('d_model', 384),
        num_heads=cfg.get('num_heads', 8),
        num_layers=cfg.get('num_layers', 4),
        d_ff=cfg.get('d_ff', 1536),
        max_seq_len=cfg.get('max_seq_len', 1024),
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}", flush=True)

    # --- V3: Longer seed only, same track-aware as the previous best ---
    out1 = "gen_best_v3.mid"
    print(f"\nGenerating {out1} (500-token seed + track-aware, no rep penalty)...", flush=True)
    tokens1 = generate_v2(
        model, tokenizer,
        start_tokens=seed_tokens,
        max_len=2500,
        temperature=0.95,
        top_k=80,
        top_p=0.95,
        track_floors={'M': 0.0, 'P': 0.08, 'B': 0.05, 'D': 0.03},
        rep_penalty=1.0,  # disabled
        rep_window=64,
        device=device,
    )
    tokenizer.detokenize_to_midi(tokens1, out1, tempo=140)
    _report(tokenizer, tokens1, seed_len=len(seed_tokens))

    # --- V3b: Gentle rep penalty (1.05) to reduce worst loops without killing structure ---
    out2 = "gen_best_v3b.mid"
    print(f"\nGenerating {out2} (500-token seed + track-aware + gentle rep penalty 1.05)...", flush=True)
    tokens2 = generate_v2(
        model, tokenizer,
        start_tokens=seed_tokens,
        max_len=2500,
        temperature=0.95,
        top_k=80,
        top_p=0.95,
        track_floors={'M': 0.0, 'P': 0.08, 'B': 0.05, 'D': 0.03},
        rep_penalty=1.05,
        rep_window=32,
        device=device,
    )
    tokenizer.detokenize_to_midi(tokens2, out2, tempo=140)
    _report(tokenizer, tokens2, seed_len=len(seed_tokens))

    print(f"\nDone!", flush=True)


if __name__ == '__main__':
    main()
