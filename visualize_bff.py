#!/usr/bin/env python3
"""
Real-time visualization of BFF experiment progress.
Displays entropy and compression metrics as ASCII graphs in the terminal.
"""

import sys
import time
import os
from collections import deque

def read_log(filepath):
    """Read the CSV log file and return data."""
    data = {'epoch': [], 'brotli_size': [], 'num_programs': [], 'higher_entropy': []}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return data
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    data['epoch'].append(int(parts[0]))
                    data['brotli_size'].append(int(parts[1]))
                    data['num_programs'].append(int(parts[2]))
                    data['higher_entropy'].append(float(parts[3]))
    except (FileNotFoundError, ValueError):
        pass
    return data

def ascii_graph(values, width=60, height=15, title="", y_label="", min_epoch=0, max_epoch=None):
    """Create an ASCII graph of values."""
    if not values:
        return f"{title}\n  No data yet..."

    if max_epoch is None:
        max_epoch = len(values)

    # Get min/max for scaling
    min_val = min(values)
    max_val = max(values)

    # Handle edge case where all values are the same
    if max_val == min_val:
        max_val = min_val + 1

    # Sample values if we have too many
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
        width = len(sampled)

    # Build the graph
    lines = []
    lines.append(f"  {title}")
    lines.append(f"  {max_val:8.3f} ┤")

    for row in range(height - 2, -1, -1):
        threshold = min_val + (max_val - min_val) * (row / (height - 1))
        line = "           │"
        for val in sampled:
            normalized = (val - min_val) / (max_val - min_val) * (height - 1)
            if int(normalized) >= row:
                # Check if this is a peak (transition indicator)
                if val > 1.0:  # High entropy indicates transition
                    line += "█"
                else:
                    line += "▄"
            else:
                line += " "
        lines.append(line)

    lines.append(f"  {min_val:8.3f} ┤" + "─" * width)
    lines.append(f"           └{'─' * (width//2)}┬{'─' * (width//2)}")
    start_label = str(min_epoch)
    lines.append(f"            {start_label}{' ' * (width//2 - len(start_label))}epochs{' ' * (width//2 - 6)}{max_epoch}")

    return "\n".join(lines)

def status_line(data):
    """Create a status line with current metrics."""
    if not data['epoch']:
        return "Waiting for data..."

    epoch = data['epoch'][-1]
    entropy = data['higher_entropy'][-1]
    brotli = data['brotli_size'][-1]

    # Detect if we might have a transition
    max_entropy = max(data['higher_entropy']) if data['higher_entropy'] else 0
    status = "🔴 Pre-life" if max_entropy < 1.0 else "🟢 TRANSITION DETECTED!" if max_entropy > 3.0 else "🟡 Evolving..."

    return f"Epoch: {epoch:,} | Entropy: {entropy:.4f} | Max Entropy: {max_entropy:.4f} | Compressed: {brotli:,} | {status}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_bff.py <log_file> [--last N]")
        print("Example: python visualize_bff.py bff_run.log")
        print("         python visualize_bff.py bff_run.log --last 500")
        sys.exit(1)

    log_file = sys.argv[1]

    # Parse optional --last argument for zooming
    last_n = None
    if len(sys.argv) >= 4 and sys.argv[2] == "--last":
        try:
            last_n = int(sys.argv[3])
        except ValueError:
            print("Error: --last requires a number")
            sys.exit(1)

    zoom_msg = f" (last {last_n} epochs)" if last_n else ""
    print(f"\n🧬 BFF Experiment Visualizer{zoom_msg}")
    print(f"📁 Monitoring: {log_file}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            # Read current data
            data = read_log(log_file)

            # Apply zoom if --last N specified
            if last_n and data['epoch']:
                for key in data:
                    data[key] = data[key][-last_n:]

            print("=" * 75)
            title_suffix = f" (last {last_n} epochs)" if last_n else ""
            print(f"  BFF PRIMORDIAL SOUP EXPERIMENT - REAL-TIME MONITOR{title_suffix}")
            print("=" * 75)
            print()

            # Show entropy graph
            min_ep = data['epoch'][0] if data['epoch'] else 0
            max_ep = data['epoch'][-1] if data['epoch'] else 0
            print(ascii_graph(
                data['higher_entropy'],
                width=60,
                height=12,
                title="Higher-Order Entropy (complexity metric)",
                min_epoch=min_ep,
                max_epoch=max_ep
            ))
            print()

            # Show compression ratio
            if data['brotli_size'] and data['num_programs']:
                # Convert to bits per byte (use actual num_programs from log)
                bpb = [size * 8 / (num_prog * 64)
                       for size, num_prog in zip(data['brotli_size'], data['num_programs'])]
                print(ascii_graph(
                    bpb,
                    width=60,
                    height=8,
                    title="Bits per Byte (compression - lower = more structure)",
                    min_epoch=min_ep,
                    max_epoch=max_ep
                ))
            print()

            # Status line
            print("-" * 75)
            print(status_line(data))
            print("-" * 75)

            # Legend
            print("\n📊 What to look for:")
            print("   • Entropy spike to 4-6 = Phase transition (life emerges!)")
            print("   • Bits per byte drop = Structure forming (replicators taking over)")
            print(f"\n⏱️  Last update: {time.strftime('%H:%M:%S')}")

            time.sleep(2)  # Update every 2 seconds

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")

        # Final summary
        data = read_log(log_file)
        if data['higher_entropy']:
            max_ent = max(data['higher_entropy'])
            print(f"\n📈 Final Summary:")
            print(f"   Total epochs: {data['epoch'][-1] if data['epoch'] else 0:,}")
            print(f"   Max entropy reached: {max_ent:.4f}")
            print(f"   Transition occurred: {'Yes! 🎉' if max_ent > 3.0 else 'No'}")

if __name__ == "__main__":
    main()
