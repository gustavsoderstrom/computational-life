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
import os

from bff_analysis import save_checkpoint, load_checkpoint

# ============================================================================
# BFF Interpreter
# ============================================================================

TAPE_SIZE = 64       # Size of each program
COMBINED_SIZE = 128  # Two programs concatenated

# BFF instruction set (ASCII codes)
LOOP_START = ord('[')      # Jump to ] if cell at head0 is 0
LOOP_END = ord(']')        # Jump back to [ if cell at head0 is not 0
PLUS = ord('+')            # Increment cell at head0
MINUS = ord('-')           # Decrement cell at head0
COPY_TO_HEAD1 = ord('.')   # tape[head1] = tape[head0]
COPY_TO_HEAD0 = ord(',')   # tape[head0] = tape[head1]  <-- KEY for replication!
DEC_HEAD0 = ord('<')       # head0 -= 1
INC_HEAD0 = ord('>')       # head0 += 1
DEC_HEAD1 = ord('{')       # head1 -= 1
INC_HEAD1 = ord('}')       # head1 += 1


def evaluate(tape: bytearray, max_steps: int = 32768) -> int:
    """
    Execute a BFF program on a tape.

    The tape is 128 bytes: two 64-byte programs concatenated together.
    After execution, the tape may be modified (this is how replication works).

    Returns the number of operations executed.
    """
    head0 = 0  # Data pointer
    head1 = 0  # Secondary pointer (for copying)
    pc = 0     # Program counter
    ops = 0    # Operations counter

    for _ in range(max_steps):
        if pc < 0 or pc >= COMBINED_SIZE:
            break

        # Wrap head positions (0-127)
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
            tape[head0] = tape[head1]  # This is how programs copy themselves!
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
                    break
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
                    break
        # else: non-command bytes are ignored (NOOPs)

        pc += 1

    return ops


# ============================================================================
# Primordial Soup
# ============================================================================

def run_soup(num_programs=1024, max_epochs=10000, seed=42, log_file=None,
             checkpoint_dir="checkpoints", checkpoint_interval=256, resume_path=None):
    """
    Run the primordial soup simulation.

    The soup is a collection of random 64-byte programs. Each epoch:
    1. Shuffle the programs randomly
    2. Pair them up and concatenate each pair into a 128-byte tape
    3. Execute the BFF interpreter on each tape
    4. Split the (now modified) tapes back into two programs

    If a program happens to copy itself onto its partner, it replicates!
    Over time, successful replicators take over the soup.
    """
    random.seed(seed)

    # Initialize soup (from checkpoint or random)
    if resume_path:
        soup, meta = load_checkpoint(resume_path)
        start_epoch = meta['epoch'] + 1
        num_programs = meta['num_programs']
        print(f"Resuming from {resume_path} at epoch {start_epoch}")
    else:
        soup = [bytearray(random.randint(0, 255) for _ in range(TAPE_SIZE))
                for _ in range(num_programs)]
        start_epoch = 0

    # Setup logging
    if log_file:
        if resume_path:
            log = open(log_file, 'a')  # Append when resuming
        else:
            log = open(log_file, 'w')
            log.write("epoch,brotli_size,num_programs,higher_entropy\n")
    else:
        log = None

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"BFF Primordial Soup: {num_programs} programs, seed {seed}")
    print(f"{'Epoch':>8} {'Entropy':>10} {'Ops/Pair':>10}")
    print("-" * 32)

    start = time.time()

    for epoch in range(start_epoch, max_epochs):
        random.shuffle(soup)
        total_ops = 0

        # === THE CORE LOOP ===
        for i in range(0, num_programs, 2):
            # Concatenate two programs
            tape = bytearray(soup[i] + soup[i + 1])

            # Execute BFF
            ops = evaluate(tape)
            total_ops += ops

            # Split back (tape may have been modified!)
            soup[i] = tape[:TAPE_SIZE]
            soup[i + 1] = tape[TAPE_SIZE:]
        # === END CORE LOOP ===

        # Measure complexity via compression
        data = b''.join(soup)
        compressed = zlib.compress(data, level=9)
        entropy = 8.0 - (len(compressed) * 8 / len(data))

        # Log progress
        if log:
            log.write(f"{epoch},{len(compressed)},{num_programs},{entropy:.6f}\n")
            log.flush()

        # Save checkpoint
        if checkpoint_dir and epoch % checkpoint_interval == 0:
            path = os.path.join(checkpoint_dir, f"{epoch:010d}.dat")
            save_checkpoint(soup, epoch, path)

        # Print progress
        if epoch % 100 == 0:
            avg_ops = total_ops / (num_programs // 2)
            print(f"{epoch:8d} {entropy:10.4f} {avg_ops:10.1f}")

        # Detect transition
        if entropy > 3.0:
            print(f"\n*** TRANSITION at epoch {epoch}! Entropy: {entropy:.2f} ***")
            break

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({epoch/elapsed:.0f} epochs/sec)")

    if log:
        log.close()

    return soup


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFF Primordial Soup")
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
