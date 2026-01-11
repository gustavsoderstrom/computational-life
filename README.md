# BFF Primordial Soup

A simple and basic (Numba-accelerated) Python implementation of the BFF (Brainfuck variant) primordial soup experiment from ["Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction"](https://arxiv.org/abs/2406.19108) by Blaise Agüera y Arcas et al.

This demonstrates how **self-replicating programs can emerge spontaneously** from random programs through self-modification — no fitness function, no selection pressure, just random interactions.

![Computational Life - BFF Primordial Soup experiment showing phase transition](computational_life.jpg)

## How It Works

1. Start with a "soup" of random 64-byte programs with completely randomized byte values (only 10 of which correspond to actual brainfuck instructions, meaning only about ~4% of cells will actually have any type of instruction at all in them and, the rest are just no-ops)
2. Each epoch: shuffle the programs, pair them up, concatenate into 128-byte tapes
3. Execute the BFF interpreter on each tape (programs can modify themselves and each other)
4. Split tapes back into programs
5. Repeat — eventually, self-replicators emerge and take over the soup

The **phase transition** is detected when higher-order entropy spikes above 3.0, indicating structured replicators have emerged from random noise.

**No mutation:** This Python implementation deliberately uses no external mutation. Programs only change through self-modification during BFF execution. This demonstrates the paper's key insight — self-replicators can emerge purely from program interactions without any external randomness or mutation pressure.

## BFF Instruction Set

BFF (Brainfuck variant) modifies standard Brainfuck for self-modification instead of I/O. It uses two head pointers on a shared tape:

| Command | Description |
|---------|-------------|
| `>` `<` | Move head0 right/left |
| `}` `{` | Move head1 right/left |
| `+` `-` | Increment/decrement byte at head0 |
| `.` | Copy byte from head0 position to head1 position |
| `,` | Copy byte from head1 position to head0 position |
| `[` | Jump past matching `]` if byte at head0 is 0 |
| `]` | Jump back to matching `[` if byte at head0 is not 0 |

**Note:** Standard Brainfuck's I/O commands (`.` and `,`) are repurposed for copying between heads.

### The Emergent Palindrome Replicator

A replicator that emerges may look like the following and often has a palindrome-like pattern:

```
{[<},]],}<[{
```

**How it works:**
1. `{` — Move head1 left (wraps to position 127)
2. `[<},]` — Copy loop:
   - `[` — Start loop (exits when byte at head0 is 0)
   - `<` — Move head0 (destination) left
   - `}` — Move head1 (source) right
   - `,` — Copy byte from head1 to head0
   - `]` — Jump back to `[` if byte at head0 ≠ 0
3. `]` — End outer structure
4. `}<[{` — Near-mirror of the start

**Why palindrome structure?**

When two programs meet on a 128-byte tape:
```
|  Program A (0-63)  |  Program B (64-127)  |
     ↑ head1 (source)     ↑ head0 (destination)
```

The replicator **copies itself backwards** into the other program's space. The palindrome structure ensures that when concatenated with another copy, the combined tape still contains a valid copy loop — making it robust to being "cut" at different points.

## Files

| File | Description |
|------|-------------|
| `bff_soup.py` | Numba-accelerated simulation (~7 epochs/sec at 131k programs on M2 MacBook Air) |
| `bff_analysis.py` | Checkpoint save/load and replicator extraction tools |
| `visualize_bff.py` | Real-time ASCII visualization of entropy and compression |

## Quick Start

### Running the Simulation

```bash
# Install dependencies
pip install numba numpy

# Run simulation with 131k programs (as used in the paper)
# ~7 epochs/sec on M2 MacBook Air, transition typically around epoch 12-16k
# Expect ~30-40 minutes to transition
python3 bff_soup.py --num 131072 --epochs 20000

# In a separate terminal, watch the progress
python3 visualize_bff.py bff_soup.log

# Zoom in on last 500 epochs to see transition detail E.g.
python3 visualize_bff.py bff_soup.log --last 500
```

### Command-line Options

```
--num N              Number of programs in soup (default: 1024)
--epochs N           Maximum epochs to run (default: 10000)
--seed N             Random seed (default: 42)
--log FILE           Log file path (default: bff_soup.log, use '' to disable)
--checkpoint-dir DIR Checkpoint directory (default: checkpoints/, use '' to disable)
--checkpoint-interval N  Save checkpoint every N epochs (default: 256)
--resume FILE        Resume simulation from checkpoint file
```

### Resuming from Checkpoint

If a run is interrupted or you want to extend it, resume from the latest checkpoint:

```bash
# Find the latest checkpoint (using the .dat file for that specific run)
ls -t checkpoints/*.dat | head -1

# Resume and run to 50000 epochs (using the .dat file for that specific run)
python3 bff_soup.py --resume checkpoints/0000010240.dat --epochs 50000
```

### Analyzing Results

# After a run (or during), examine emergent replicators: (with the .dat file for that specific run)

```bash
python3 bff_analysis.py checkpoints/0000001024.dat --top 10
```

## Running with cubff (C++ Implementation)

For maximum speed, the [cubff](https://github.com/paradigms-of-intelligence/cubff) C++ implementation is ~4x faster than this Python version:

```bash
# Clone and build
git clone https://github.com/paradigms-of-intelligence/cubff.git
cd cubff && mkdir build && cd build
cmake .. -DCUDA=OFF
make -j$(nproc)

# Run (outputs to stdout in CSV format)
./main --lang bff_noheads --soup-size 1024 --print-interval 64 > ../bff_run.log &

# Use the same visualizer
cd ..
python3 visualize_bff.py bff_run.log
```

## Metrics Explained

The simulation tracks two metrics based on compression (using zlib in the Python version, vs Brotli in the cubff version):

**Higher-Order Entropy** (complexity metric, as used in the paper):
```
entropy = H0 - bpb
```
Where H0 is Shannon entropy and bpb is bits-per-byte after compression.
- Measures "bits saved per byte" through compression beyond simple character frequencies
- **Random soup ≈ 0**: Incompressible noise, no patterns
- **Structured soup > 3**: Repetitive patterns (replicators) compress well
- A sudden spike indicates phase transition — replicators have taken over

**Bits per Byte** (compression ratio):
```
bpb = compressed_size × 8 / original_size
```
- Inverse of entropy: how many bits needed per byte after compression
- **Random soup ≈ 8 bpb**: No compression possible
- **Structured soup < 5 bpb**: Significant compression = replicators present

## What to Look For

In the visualizer:
- **🔴 Pre-life**: Entropy near 0, ~8 bpb, random noise
- **🟡 Evolving**: Entropy 1-3, structure forming
- **🟢 TRANSITION**: Entropy spikes to 4-6, bpb drops below 4, replicators have emerged!

A successful transition typically shows:
- Sudden entropy spike (0 → 4+)
- Bits per byte drops (8 → 4 or lower)
- Operations per pair jumps from hundreds to thousands

## References

- Paper: [arXiv:2406.19108](https://arxiv.org/abs/2406.19108)
- Original implementation: [github.com/paradigms-of-intelligence/cubff](https://github.com/paradigms-of-intelligence/cubff)
- Sean Carroll interview: [Mindscape Podcast](https://www.preposterousuniverse.com/podcast/2024/07/22/283-blaise-aguera-y-arcas-on-the-emergence-of-replication-and-computation/)
