#!/usr/bin/env python3
"""
Pure Python BFF Primordial Soup Simulation

A simple, educational implementation of the BFF (Brainfuck variant) primordial soup
experiment from "Computational Life" by Blaise Aguera y Arcas et al.

This demonstrates how self-replicating programs can emerge spontaneously from
random programs through self-modification.

Usage:
    python bff_soup.py [--num 1024] [--epochs 10000] [--seed 42]
"""

import random
import zlib
import time
import argparse
from typing import List, Tuple

# ============================================================================
# BFF Interpreter
# ============================================================================

# BFF instruction set (ASCII codes)
LOOP_START = ord('[')  # Jump to ] if cell at head0 is 0
LOOP_END = ord(']')    # Jump back to [ if cell at head0 is not 0
PLUS = ord('+')        # Increment cell at head0
MINUS = ord('-')       # Decrement cell at head0
COPY_TO_HEAD1 = ord('.')   # tape[head1] = tape[head0]
COPY_TO_HEAD0 = ord(',')   # tape[head0] = tape[head1]
DEC_HEAD0 = ord('<')   # head0 -= 1
INC_HEAD0 = ord('>')   # head0 += 1
DEC_HEAD1 = ord('{')   # head1 -= 1
INC_HEAD1 = ord('}')   # head1 += 1

# Valid BFF commands
COMMANDS = {LOOP_START, LOOP_END, PLUS, MINUS, COPY_TO_HEAD1, COPY_TO_HEAD0,
            DEC_HEAD0, INC_HEAD0, DEC_HEAD1, INC_HEAD1}

TAPE_SIZE = 64  # Size of each program
COMBINED_SIZE = 128  # Two programs concatenated


def evaluate(tape: bytearray, max_steps: int = 32768) -> int:
    """
    Execute a BFF program on a tape.

    Args:
        tape: 128-byte tape (two 64-byte programs concatenated)
        max_steps: Maximum execution steps

    Returns:
        Number of actual operations executed
    """
    head0 = 0  # Data pointer
    head1 = 0  # Secondary pointer (for copying)
    pc = 0     # Program counter
    ops = 0    # Operations counter

    for _ in range(max_steps):
        if pc < 0 or pc >= COMBINED_SIZE:
            break

        # Wrap head positions
        head0 &= (COMBINED_SIZE - 1)
        head1 &= (COMBINED_SIZE - 1)

        cmd = tape[pc]

        if cmd == DEC_HEAD0:
            head0 -= 1
            ops += 1
        elif cmd == INC_HEAD0:
            head0 += 1
            ops += 1
        elif cmd == DEC_HEAD1:
            head1 -= 1
            ops += 1
        elif cmd == INC_HEAD1:
            head1 += 1
            ops += 1
        elif cmd == PLUS:
            tape[head0] = (tape[head0] + 1) & 0xFF
            ops += 1
        elif cmd == MINUS:
            tape[head0] = (tape[head0] - 1) & 0xFF
            ops += 1
        elif cmd == COPY_TO_HEAD1:
            tape[head1] = tape[head0]
            ops += 1
        elif cmd == COPY_TO_HEAD0:
            tape[head0] = tape[head1]
            ops += 1
        elif cmd == LOOP_START:
            ops += 1
            if tape[head0] == 0:
                # Skip to matching ]
                depth = 1
                pc += 1
                while pc < COMBINED_SIZE and depth > 0:
                    if tape[pc] == LOOP_END:
                        depth -= 1
                    elif tape[pc] == LOOP_START:
                        depth += 1
                    pc += 1
                pc -= 1
                if depth != 0:
                    break  # Unmatched bracket, halt
        elif cmd == LOOP_END:
            ops += 1
            if tape[head0] != 0:
                # Jump back to matching [
                depth = 1
                pc -= 1
                while pc >= 0 and depth > 0:
                    if tape[pc] == LOOP_START:
                        depth -= 1
                    elif tape[pc] == LOOP_END:
                        depth += 1
                    pc -= 1
                pc += 1
                if depth != 0:
                    break  # Unmatched bracket, halt
        # else: NOOP (non-command bytes are ignored)

        pc += 1

    return ops


# ============================================================================
# Primordial Soup Simulation
# ============================================================================

def random_program(size: int = TAPE_SIZE) -> bytearray:
    """Generate a random program of given size."""
    return bytearray(random.randint(0, 255) for _ in range(size))


def compute_entropy(soup: List[bytearray]) -> float:
    """
    Compute higher-order entropy using compression.

    Higher values indicate more structure/patterns (like replicators).
    Near-zero values indicate random noise.
    """
    # Concatenate all programs
    data = b''.join(soup)

    # Compress with zlib (similar to brotli, widely available)
    compressed = zlib.compress(data, level=9)

    # Shannon entropy (bits per byte) for random data is ~8
    # Compression ratio gives us an approximation of Kolmogorov complexity
    raw_size = len(data)
    compressed_size = len(compressed)

    # Higher-order entropy = Shannon entropy - normalized Kolmogorov complexity
    # For random data: ~8 - ~8 = ~0
    # For structured data: ~8 - ~2 = ~6
    bits_per_byte = (compressed_size * 8) / raw_size
    higher_entropy = 8.0 - bits_per_byte

    return higher_entropy


def extract_programs(soup: List[bytearray]) -> dict:
    """Extract and count unique program patterns."""
    # Convert programs to strings of BFF commands only
    counts = {}
    for prog in soup:
        # Filter to just BFF commands
        cmds = ''.join(chr(b) for b in prog if b in COMMANDS)
        counts[cmds] = counts.get(cmds, 0) + 1
    return counts


def print_top_programs(soup: List[bytearray], top_n: int = 5):
    """Print the most common programs in the soup."""
    counts = extract_programs(soup)
    sorted_progs = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    print("\nTop programs:")
    for prog, count in sorted_progs:
        display = prog[:40] + "..." if len(prog) > 40 else prog
        print(f"  {count:5d}x  {display}")


def run_simulation(
    num_programs: int = 1024,
    max_epochs: int = 10000,
    seed: int = 42,
    print_interval: int = 100,
    mutation_rate: float = 0.0,
    log_file: str = None,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 256,
):
    """
    Run the primordial soup simulation.

    Args:
        num_programs: Number of programs in the soup (must be even)
        max_epochs: Maximum number of epochs to run
        seed: Random seed for reproducibility
        print_interval: How often to print progress
        mutation_rate: Probability of mutating each byte (0 = no mutation)
        log_file: Path to CSV log file for visualization (optional)
        checkpoint_dir: Directory to save soup checkpoints (optional)
        checkpoint_interval: Save checkpoint every N epochs (default 256)
    """
    import os
    random.seed(seed)

    # Initialize soup with random programs
    soup = [random_program() for _ in range(num_programs)]

    # Open log file if specified
    log_handle = None
    if log_file:
        log_handle = open(log_file, 'w')
        log_handle.write("epoch,brotli_size,soup_size,higher_entropy\n")

    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"BFF Primordial Soup Simulation")
    print(f"=" * 50)
    print(f"Programs: {num_programs}")
    print(f"Max epochs: {max_epochs}")
    print(f"Seed: {seed}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"=" * 50)
    print()
    print(f"{'Epoch':>8} {'Entropy':>10} {'Ops/Pair':>10} {'Status'}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*20}")

    start_time = time.time()
    max_entropy = 0.0
    transition_epoch = None

    for epoch in range(max_epochs):
        # Shuffle programs for random pairing
        random.shuffle(soup)

        total_ops = 0

        # Process pairs
        for i in range(0, num_programs, 2):
            # Concatenate two programs into a tape
            tape = bytearray(soup[i] + soup[i + 1])

            # Execute
            ops = evaluate(tape)
            total_ops += ops

            # Apply mutation if enabled
            if mutation_rate > 0:
                for j in range(len(tape)):
                    if random.random() < mutation_rate:
                        tape[j] = random.randint(0, 255)

            # Split back into two programs
            soup[i] = tape[:TAPE_SIZE]
            soup[i + 1] = tape[TAPE_SIZE:]

        # Compute metrics
        entropy = compute_entropy(soup)
        avg_ops = total_ops / (num_programs // 2)

        # Track maximum entropy
        if entropy > max_entropy:
            max_entropy = entropy

        # Detect transition
        if entropy > 3.0 and transition_epoch is None:
            transition_epoch = epoch
            print(f"\n*** PHASE TRANSITION DETECTED at epoch {epoch}! ***\n")

        # Write to log file (compute compressed size for compatibility)
        if log_handle:
            data = b''.join(soup)
            compressed_size = len(zlib.compress(data, level=9))
            log_handle.write(f"{epoch},{compressed_size},{num_programs},{entropy:.6f}\n")
            log_handle.flush()

        # Save checkpoint if specified
        if checkpoint_dir and epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch:010d}.dat")
            with open(checkpoint_path, 'wb') as f:
                # Write simple header (24 bytes to match C++ format)
                f.write(b'BFFS')  # Magic
                f.write(num_programs.to_bytes(4, 'little'))
                f.write(TAPE_SIZE.to_bytes(4, 'little'))
                f.write(epoch.to_bytes(4, 'little'))
                f.write(b'\x00' * 8)  # Padding
                # Write all programs
                for prog in soup:
                    f.write(prog)

        # Print progress
        if epoch % print_interval == 0 or epoch == max_epochs - 1:
            if entropy > 3.0:
                status = "LIFE!"
            elif entropy > 1.0:
                status = "Evolving..."
            else:
                status = "Pre-life"

            print(f"{epoch:8d} {entropy:10.4f} {avg_ops:10.1f} {status}")

    # Final summary
    elapsed = time.time() - start_time
    print()
    print(f"=" * 50)
    print(f"Simulation complete!")
    print(f"Time: {elapsed:.1f}s ({max_epochs/elapsed:.0f} epochs/sec)")
    print(f"Max entropy: {max_entropy:.4f}")

    if transition_epoch:
        print(f"Transition at epoch: {transition_epoch}")
        print_top_programs(soup)
    else:
        print("No transition occurred (entropy never exceeded 3.0)")

    # Close log file
    if log_handle:
        log_handle.close()
        print(f"Log saved to: {log_file}")

    return soup, max_entropy, transition_epoch


def analyze_checkpoint(checkpoint_path: str, top_n: int = 10):
    """
    Analyze a checkpoint file and print the most common programs.

    Usage: python bff_soup.py --analyze <checkpoint_file>
    """
    with open(checkpoint_path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'BFFS':
            print(f"Invalid checkpoint file (magic: {magic})")
            return
        num_programs = int.from_bytes(f.read(4), 'little')
        tape_size = int.from_bytes(f.read(4), 'little')
        epoch = int.from_bytes(f.read(4), 'little')
        f.read(8)  # Skip padding

        # Read programs
        soup = []
        for _ in range(num_programs):
            prog = bytearray(f.read(tape_size))
            soup.append(prog)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {epoch}")
    print(f"Programs: {num_programs}")
    print(f"Tape size: {tape_size}")
    print()

    # Count unique programs (BFF commands only)
    counts = extract_programs(soup)
    sorted_progs = sorted(counts.items(), key=lambda x: -x[1])[:top_n]

    print(f"Top {top_n} programs:")
    for prog, count in sorted_progs:
        pct = 100 * count / num_programs
        display = prog[:50] + "..." if len(prog) > 50 else prog if prog else "(empty)"
        print(f"  {count:5d} ({pct:5.1f}%)  {display}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFF Primordial Soup Simulation")
    parser.add_argument("--num", type=int, default=1024, help="Number of programs")
    parser.add_argument("--epochs", type=int, default=10000, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interval", type=int, default=100, help="Print interval")
    parser.add_argument("--mutation", type=float, default=0.0, help="Mutation rate")
    parser.add_argument("--log", type=str, default=None, help="Log file for visualization")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=256, help="Checkpoint every N epochs")
    parser.add_argument("--analyze", type=str, default=None, help="Analyze a checkpoint file")

    args = parser.parse_args()

    # If analyzing a checkpoint, do that and exit
    if args.analyze:
        analyze_checkpoint(args.analyze)
        exit(0)

    if args.num % 2 != 0:
        print("Error: Number of programs must be even")
        exit(1)

    run_simulation(
        num_programs=args.num,
        max_epochs=args.epochs,
        seed=args.seed,
        print_interval=args.interval,
        mutation_rate=args.mutation,
        log_file=args.log,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
