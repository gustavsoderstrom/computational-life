# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pure Python implementation of the BFF (Brainfuck variant) primordial soup experiment from "Computational Life" by Blaise Agüera y Arcas et al. Demonstrates how self-replicating programs can emerge spontaneously from random programs through self-modification.

## Commands

### Run simulation (use PyPy for ~24x speedup)
```bash
pypy3 bff_soup.py --num 1024 --epochs 10000 --seed 42
```

### Run with standard Python
```bash
python3 bff_soup.py --num 1024 --epochs 10000
```

### Real-time visualization (run in separate terminal)
```bash
python3 visualize_bff.py bff_soup.log
```

### Analyze checkpoint for emergent replicators
```bash
python3 bff_analysis.py checkpoints/0000001024.dat --top 10
```

## Architecture

- **bff_soup.py** - Core simulation: BFF interpreter + primordial soup loop. Programs are 64-byte tapes paired and executed together. Self-modification via `,` instruction enables replication.
- **bff_analysis.py** - Checkpoint save/load and replicator extraction tools.
- **visualize_bff.py** - ASCII real-time entropy/compression graphs.
- **cubff/** - External C++ implementation (not part of this project, ~100x faster).

## Key Concepts

- **Phase transition**: Entropy jumps from ~0 to 4-6 when replicators emerge (threshold: >3.0)
- **Checkpoint format**: Binary files with `BFFS` magic header, stores full soup state
- **Log format**: CSV with epoch, compressed_size, num_programs, entropy

## Defaults

- Logging enabled by default → `bff_soup.log`
- Checkpoints enabled by default → `checkpoints/` every 256 epochs
- Disable with `--log ''` or `--checkpoint-dir ''`
