#!/usr/bin/env python3
"""
Generate music using trained Music Transformer.
Autoregressive generation with temperature sampling.
"""
import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model.transformer import MusicTransformer
from data.tokenizer import MIDITokenizer


def load_trained_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a trained model from checkpoint.
    """
    print(f"📥 Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create tokenizer
    tokenizer = MIDITokenizer()

    # Read model config from checkpoint (falls back to old defaults)
    cfg = checkpoint.get('config', {})
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.get('d_model', 512),
        num_heads=cfg.get('num_heads', 8),
        num_layers=cfg.get('num_layers', 6),
        d_ff=cfg.get('d_ff', 2048),
        max_seq_len=cfg.get('seq_len', cfg.get('max_seq_len', 1024)),
        dropout=0.0,  # No dropout during generation
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")

    return model, tokenizer


def generate_music(
    model: MusicTransformer,
    tokenizer: MIDITokenizer,
    output_path: str,
    max_len: int = 2000,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed_tokens: list = None,
    device: str = 'cpu',
    tempo: int = 120,
):
    """
    Generate a MIDI file using the trained transformer.

    Args:
        model: Trained MusicTransformer
        tokenizer: MIDITokenizer
        output_path: Where to save generated MIDI
        max_len: Maximum sequence length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens by probability
        top_p: Nucleus sampling threshold
        seed_tokens: Optional starting tokens
        device: torch device
        tempo: Tempo in BPM
    """
    print(f"\n🎵 Generating music...")
    print(f"   Max length: {max_len} tokens")
    print(f"   Temperature: {temperature}")
    print(f"   Top-k: {top_k}")
    print(f"   Top-p: {top_p}")

    # Start with START token or provided seed
    if seed_tokens is None:
        seed_tokens = [tokenizer.token_to_id['START']]

    # Generate token sequence
    generated_tokens = model.generate(
        start_tokens=seed_tokens,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )

    print(f"✅ Generated {len(generated_tokens)} tokens")

    # Convert tokens back to MIDI
    print(f"\n💾 Converting to MIDI...")
    tokenizer.detokenize_to_midi(
        tokens=generated_tokens,
        output_path=output_path,
        tempo=tempo
    )

    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate music with trained Music Transformer',
        epilog="""
This uses a REAL TRANSFORMER with:
  - Multi-head self-attention (learns long-range dependencies)
  - Trained with backpropagation (not just pattern matching)
  - Can generate coherent multi-track compositions

Examples:
  # Generate with default settings
  python generate_transformer.py --checkpoint checkpoints/checkpoint_latest.pt

  # Higher temperature (more creative)
  python generate_transformer.py --checkpoint checkpoints/checkpoint_latest.pt --temperature 1.2

  # Longer piece
  python generate_transformer.py --checkpoint checkpoints/checkpoint_latest.pt --max-len 3000

  # Ballad tempo
  python generate_transformer.py --checkpoint checkpoints/checkpoint_latest.pt --tempo 80
        """
    )

    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output', '-o',
        default='generated.mid',
        help='Output MIDI file path'
    )
    parser.add_argument(
        '--max-len',
        type=int,
        default=2000,
        help='Maximum length in tokens (default: 2000)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0, higher = more random)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling threshold (default: 0.95)'
    )
    parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device (cpu/mps/cuda/auto)'
    )

    args = parser.parse_args()

    print("🎸 Music Transformer Generator")
    print("=" * 70)

    # Determine device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")

    # Load model
    model, tokenizer = load_trained_model(args.checkpoint, device=device)

    # Generate music
    generate_music(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        tempo=args.tempo,
    )

    print("\n🎉 Generation complete!")
    print("\nThis music was generated by a REAL TRANSFORMER:")
    print("  ✅ Trained with backpropagation")
    print("  ✅ Self-attention learns long-range dependencies")
    print("  ✅ Multi-track coordination")
    print("  ✅ Understands musical context across time")
    print("\nNot just Markov chains - actual deep learning! 🚀")


if __name__ == '__main__':
    main()
