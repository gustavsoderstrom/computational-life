#!/usr/bin/env python3
"""
Numba-accelerated BFF Primordial Soup Simulation

A fast implementation of the BFF (Brainfuck variant) primordial soup
experiment from "Computational Life" by Blaise Aguera y Arcas et al.
Requires: pip install numba numpy

Usage:
    python bff_soup.py [--num 1024] [--epochs 10000] [--seed 42]
"""

import numpy as np
from numba import jit
import time
import argparse
import os

from bff_analysis import save_checkpoint

# ============================================================================
# BFF Interpreter (Numba-compatible)
# ============================================================================

TAPE_SIZE = 64
COMBINED_SIZE = 128

# BFF instruction set (pre-computed as ints for Numba)
LOOP_START = 91      # ord('[')
LOOP_END = 93        # ord(']')
PLUS = 43            # ord('+')
MINUS = 45           # ord('-')
COPY_TO_HEAD1 = 46   # ord('.')
COPY_TO_HEAD0 = 44   # ord(',')
DEC_HEAD0 = 60       # ord('<')
INC_HEAD0 = 62       # ord('>')
DEC_HEAD1 = 123      # ord('{')
INC_HEAD1 = 125      # ord('}')


@jit(nopython=True)
def evaluate(tape, max_steps=32768):
    """
    Execute a BFF program on a tape (Numba JIT compiled).

    The tape is 128 bytes: two 64-byte programs concatenated together.
    After execution, the tape may be modified (this is how replication works).

    Returns the number of operations executed.
    """
    head0 = 0
    head1 = 0
    pc = 0
    ops = 0

    for _ in range(max_steps):
        if pc < 0 or pc >= COMBINED_SIZE:
            break

        # Wrap head positions (0-127)
        head0 = head0 & (COMBINED_SIZE - 1)
        head1 = head1 & (COMBINED_SIZE - 1)

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
                    break
        elif cmd == LOOP_END:
            ops += 1
            if tape[head0] != 0:
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
                    break

        pc += 1

    return ops


# ============================================================================
# Primordial Soup
# ============================================================================

def run_soup(num_programs=1024, max_epochs=10000, seed=42, log_file=None,
             checkpoint_dir="checkpoints", checkpoint_interval=256, resume_path=None):
    """
    Run the primordial soup simulation with Numba acceleration.
    """
    np.random.seed(seed)

    # Initialize soup (from checkpoint or random)
    if resume_path:
        from bff_analysis import load_checkpoint
        soup_bytes, meta = load_checkpoint(resume_path)
        # Convert bytearrays to numpy arrays
        soup = [np.array(prog, dtype=np.uint8) for prog in soup_bytes]
        start_epoch = meta['epoch'] + 1
        num_programs = meta['num_programs']
        print(f"Resuming from {resume_path} at epoch {start_epoch}")
    else:
        soup = [np.random.randint(0, 256, TAPE_SIZE, dtype=np.uint8)
                for _ in range(num_programs)]
        start_epoch = 0

    # Setup logging
    if log_file:
        if resume_path:
            log = open(log_file, 'a')
        else:
            log = open(log_file, 'w')
            log.write("epoch,brotli_size,num_programs,higher_entropy\n")
    else:
        log = None

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"BFF Primordial Soup (Numba): {num_programs} programs, seed {seed}")
    print(f"{'Epoch':>8} {'Entropy':>10} {'Ops/Pair':>10}")
    print("-" * 32)

    # Warm up Numba JIT
    dummy = np.zeros(COMBINED_SIZE, dtype=np.uint8)
    evaluate(dummy, 100)

    start = time.time()

    for epoch in range(start_epoch, max_epochs):
        np.random.shuffle(soup)
        total_ops = 0

        # === THE CORE LOOP ===
        for i in range(0, num_programs, 2):
            # Concatenate two programs
            tape = np.concatenate((soup[i], soup[i + 1]))

            # Execute BFF
            ops = evaluate(tape)
            total_ops += ops

            # Split back (tape may have been modified!)
            soup[i] = tape[:TAPE_SIZE].copy()
            soup[i + 1] = tape[TAPE_SIZE:].copy()
        # === END CORE LOOP ===

        # Measure complexity via compression
        import zlib
        data = b''.join(bytes(s) for s in soup)
        compressed = zlib.compress(data, level=9)
        entropy = 8.0 - (len(compressed) * 8 / len(data))

        # Log progress
        if log:
            log.write(f"{epoch},{len(compressed)},{num_programs},{entropy:.6f}\n")
            log.flush()

        # Save checkpoint (convert to bytearrays for compatibility)
        if checkpoint_dir and epoch % checkpoint_interval == 0:
            path = os.path.join(checkpoint_dir, f"{epoch:010d}.dat")
            soup_bytes = [bytearray(s) for s in soup]
            save_checkpoint(soup_bytes, epoch, path)

        # Print progress
        if epoch % 100 == 0:
            avg_ops = total_ops / (num_programs // 2)
            print(f"{epoch:8d} {entropy:10.4f} {avg_ops:10.1f}")

        # Detect transition (log but don't stop)
        if entropy > 3.0 and epoch % 100 == 0:
            print(f"*** TRANSITION at epoch {epoch}! Entropy: {entropy:.2f} ***")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({epoch/elapsed:.0f} epochs/sec)")

    if log:
        log.close()

    return soup


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFF Primordial Soup (Numba)")
    parser.add_argument("--num", type=int, default=1024, help="Number of programs")
    parser.add_argument("--epochs", type=int, default=10000, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log", type=str, default="bff_soup.log",
                        help="Log file (default: bff_soup.log, use '' to disable)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints, use '' to disable)")
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint file")

    args = parser.parse_args()

    run_soup(
        num_programs=args.num,
        max_epochs=args.epochs,
        seed=args.seed,
        log_file=args.log if args.log else None,
        checkpoint_dir=args.checkpoint_dir if args.checkpoint_dir else None,
        checkpoint_interval=args.checkpoint_interval,
        resume_path=args.resume,
    )
