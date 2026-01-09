#!/usr/bin/env python3
"""
BFF Checkpoint Analysis Tools

Analyze checkpoints from BFF primordial soup simulations to extract
and examine emergent replicators.

Usage:
    python bff_analysis.py <checkpoint_file> [--top 10]
"""

import argparse

# BFF command set (must match bff_soup.py)
COMMANDS = {ord('['), ord(']'), ord('+'), ord('-'), ord('.'), ord(','),
            ord('<'), ord('>'), ord('{'), ord('}')}

TAPE_SIZE = 64


def extract_programs(soup):
    """
    Extract and count unique program patterns from a soup.

    Args:
        soup: List of bytearrays (programs)

    Returns:
        Dictionary mapping BFF command strings to counts
    """
    counts = {}
    for prog in soup:
        # Filter to just BFF commands
        cmds = ''.join(chr(b) for b in prog if b in COMMANDS)
        counts[cmds] = counts.get(cmds, 0) + 1
    return counts


def print_top_programs(soup, top_n=10):
    """Print the most common programs in the soup."""
    counts = extract_programs(soup)
    sorted_progs = sorted(counts.items(), key=lambda x: -x[1])[:top_n]

    num_programs = len(soup)
    print(f"\nTop {top_n} programs:")
    for prog, count in sorted_progs:
        pct = 100 * count / num_programs
        display = prog[:50] + "..." if len(prog) > 50 else prog if prog else "(empty)"
        print(f"  {count:5d} ({pct:5.1f}%)  {display}")


def load_checkpoint(checkpoint_path):
    """
    Load a checkpoint file and return the soup.

    Args:
        checkpoint_path: Path to .dat checkpoint file

    Returns:
        Tuple of (soup, metadata) where metadata is a dict with
        'num_programs', 'tape_size', 'epoch'
    """
    with open(checkpoint_path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'BFFS':
            raise ValueError(f"Invalid checkpoint file (magic: {magic})")

        num_programs = int.from_bytes(f.read(4), 'little')
        tape_size = int.from_bytes(f.read(4), 'little')
        epoch = int.from_bytes(f.read(4), 'little')
        f.read(8)  # Skip padding

        # Read programs
        soup = []
        for _ in range(num_programs):
            prog = bytearray(f.read(tape_size))
            soup.append(prog)

    metadata = {
        'num_programs': num_programs,
        'tape_size': tape_size,
        'epoch': epoch,
    }
    return soup, metadata


def analyze_checkpoint(checkpoint_path, top_n=10):
    """
    Analyze a checkpoint file and print the most common programs.

    Args:
        checkpoint_path: Path to checkpoint file
        top_n: Number of top programs to display
    """
    soup, meta = load_checkpoint(checkpoint_path)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {meta['epoch']}")
    print(f"Programs: {meta['num_programs']}")
    print(f"Tape size: {meta['tape_size']}")

    print_top_programs(soup, top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BFF checkpoint files")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--top", type=int, default=10, help="Number of top programs to show")

    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint, args.top)
