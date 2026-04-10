#!/usr/bin/env python3
"""
Enhanced generation v4:
  1. Lower temperature (0.80) — more coherent phrase continuation
  2. Pitch-pattern bonus — boost recently-heard pitches to maintain motifs
  3. Recency-weighted attention — model pays more attention to current phrase
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


def get_seed_tokens(tokenizer, jazz_dir, min_tracks=3, seed_len=200, rng_seed=42):
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
                print(f"Seed: {f} ({len(tokens)} tokens, tracks: {types})", flush=True)
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
            if tokenizer.id_to_token[tid].split('_')[0] == track:
                mask[tid] = True
        masks[track] = mask
    return masks


def build_pitch_map(tokenizer):
    """Map each NOTE_ON token ID to its pitch, for pitch-pattern bonus."""
    pitch_map = {}  # token_id -> pitch
    pitch_to_ons = {}  # pitch -> list of NOTE_ON token IDs (across all tracks)
    for tid in range(tokenizer.vocab_size):
        name = tokenizer.id_to_token[tid]
        parts = name.split('_')
        if len(parts) >= 4 and parts[1] == 'ON':
            pitch = int(parts[2])
            pitch_map[tid] = pitch
            if pitch not in pitch_to_ons:
                pitch_to_ons[pitch] = []
            pitch_to_ons[pitch].append(tid)
    return pitch_map, pitch_to_ons


def generate_v4(model, tokenizer, start_tokens, max_len=2000,
                temperature=0.80, top_k=80, top_p=0.95,
                track_floor=0.05, pitch_bonus=1.5, pitch_window=100,
                recency_bias=0.0, recency_decay=50.0,
                device='cpu'):
    """
    Generation with all three improvements:
      - track-aware sampling (prevents collapse)
      - pitch-pattern bonus (maintains motifs)
      - recency-weighted attention (focus on current phrase)
    """
    model.eval()
    generated = list(start_tokens)
    masks = build_track_masks(tokenizer, device=device)
    pitch_map, pitch_to_ons = build_pitch_map(tokenizer)

    for step in range(max_len - len(generated)):
        window = generated[-model.max_seq_len:]
        x = torch.tensor([window], dtype=torch.long, device=device)

        logits = model(x, recency_bias=recency_bias, recency_decay=recency_decay)
        logits = logits[:, -1, :]  # (1, vocab)

        # --- Pitch-pattern bonus ---
        # Boost NOTE_ON tokens whose pitch appeared recently
        if pitch_bonus > 0:
            recent = generated[-pitch_window:]
            recent_pitches = Counter()
            for tid in recent:
                if tid in pitch_map:
                    recent_pitches[pitch_map[tid]] += 1
            # Boost pitches proportional to how often they appeared
            for pitch, count in recent_pitches.items():
                if pitch in pitch_to_ons:
                    # Gentle boost: scale by log(count+1) to avoid runaway
                    import math
                    bonus = pitch_bonus * math.log1p(count)
                    for on_tid in pitch_to_ons[pitch]:
                        logits[0, on_tid] += bonus

        # Apply temperature
        logits = logits / temperature

        # --- Track-aware rebalancing ---
        probs = F.softmax(logits, dim=-1).squeeze()
        track_probs = {t: probs[masks[t]].sum().item() for t in ['M', 'P', 'B', 'D']}
        max_track = max(track_probs, key=track_probs.get)
        if track_probs[max_track] > 0.80:
            for track in ['M', 'P', 'B', 'D']:
                if track != max_track and track_probs[track] < track_floor:
                    deficit = track_floor - track_probs[track]
                    logits[0, masks[track]] += 2.0 + deficit * 20

        # Top-k
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        remove = cum > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = 0
        logits[0, sorted_i[0][remove[0]]] = float('-inf')

        probs_final = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs_final, num_samples=1).item()

        if next_token == 2:  # END
            break
        generated.append(next_token)

    return generated


def _report(tokenizer, tokens, seed_len=0):
    gen_tokens = tokens[seed_len:]
    types = Counter()
    for t in gen_tokens:
        types[tokenizer.id_to_token[t].split('_')[0]] += 1
    total = len(gen_tokens)
    parts = [f"{k}={v}({100*v/max(total,1):.0f}%)" for k, v in types.most_common()]
    print(f"    Generated {total} tokens: {', '.join(parts)}", flush=True)


def main():
    device = get_device()
    print(f"Device: {device}", flush=True)
    tokenizer = MIDITokenizer()
    jazz_dir = str(Path(__file__).parent / '..' / 'augmented_dataset_v2')

    # Same seed as the previous best (gen_finetune_ep1_trackaware.mid)
    seed_tokens, seed_file = get_seed_tokens(tokenizer, jazz_dir, min_tracks=3, seed_len=200)
    if seed_tokens is None:
        print("ERROR: No suitable seed file found")
        return

    # Load the 9M pretrained+finetuned ep1 model (produced the best output)
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

    # --- A: Baseline (reproduce previous best settings for comparison) ---
    print(f"\n[A] Baseline (temp=0.95, no pitch bonus, no recency bias)...", flush=True)
    tA = generate_v4(model, tokenizer, seed_tokens, max_len=2000,
                     temperature=0.95, top_k=80, top_p=0.95,
                     track_floor=0.05, pitch_bonus=0.0,
                     recency_bias=0.0,
                     device=device)
    tokenizer.detokenize_to_midi(tA, 'gen_v4_baseline.mid', tempo=140)
    _report(tokenizer, tA, len(seed_tokens))

    # --- B: Lower temperature only ---
    print(f"\n[B] Lower temp (0.80)...", flush=True)
    tB = generate_v4(model, tokenizer, seed_tokens, max_len=2000,
                     temperature=0.80, top_k=80, top_p=0.95,
                     track_floor=0.05, pitch_bonus=0.0,
                     recency_bias=0.0,
                     device=device)
    tokenizer.detokenize_to_midi(tB, 'gen_v4_lowtemp.mid', tempo=140)
    _report(tokenizer, tB, len(seed_tokens))

    # --- C: Lower temp + pitch bonus ---
    print(f"\n[C] Lower temp + pitch bonus...", flush=True)
    tC = generate_v4(model, tokenizer, seed_tokens, max_len=2000,
                     temperature=0.80, top_k=80, top_p=0.95,
                     track_floor=0.05, pitch_bonus=1.5, pitch_window=100,
                     recency_bias=0.0,
                     device=device)
    tokenizer.detokenize_to_midi(tC, 'gen_v4_pitch.mid', tempo=140)
    _report(tokenizer, tC, len(seed_tokens))

    # --- D: All three (lower temp + pitch bonus + recency attention) ---
    print(f"\n[D] All three (temp=0.80, pitch bonus, recency bias=1.0)...", flush=True)
    tD = generate_v4(model, tokenizer, seed_tokens, max_len=2000,
                     temperature=0.80, top_k=80, top_p=0.95,
                     track_floor=0.05, pitch_bonus=1.5, pitch_window=100,
                     recency_bias=1.0, recency_decay=50.0,
                     device=device)
    tokenizer.detokenize_to_midi(tD, 'gen_v4_all.mid', tempo=140)
    _report(tokenizer, tD, len(seed_tokens))

    print(f"\nDone! 4 files generated.", flush=True)


if __name__ == '__main__':
    main()
