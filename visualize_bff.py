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
    data = {'epoch': [], 'brotli_size': [], 'higher_entropy': []}
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
                    data['higher_entropy'].append(float(parts[3]))
    except (FileNotFoundError, ValueError):
        pass
    return data

def ascii_graph(values, width=60, height=15, title="", y_label=""):
    """Create an ASCII graph of values."""
    if not values:
        return f"{title}\n  No data yet..."

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
    lines.append(f"            0{' ' * (width//2 - 2)}epochs{' ' * (width//2 - 6)}{len(values) * 256 if values else 0}")

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

    return f"Epoch: {epoch:,} | Entropy: {entropy:.4f} | Max Entropy: {max_entropy:.4f} | Brotli: {brotli:,} | {status}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_bff.py <log_file>")
        print("Example: python visualize_bff.py bff_run.log")
        sys.exit(1)

    log_file = sys.argv[1]

    print(f"\n🧬 BFF Experiment Visualizer")
    print(f"📁 Monitoring: {log_file}")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            # Read current data
            data = read_log(log_file)

            print("=" * 75)
            print("  BFF PRIMORDIAL SOUP EXPERIMENT - REAL-TIME MONITOR")
            print("=" * 75)
            print()

            # Show entropy graph
            print(ascii_graph(
                data['higher_entropy'],
                width=60,
                height=12,
                title="Higher-Order Entropy (complexity metric)"
            ))
            print()

            # Show compression ratio
            if data['brotli_size']:
                # Convert to bits per byte
                bpb = [size * 8 / (1024 * 64) for size in data['brotli_size']]
                print(ascii_graph(
                    bpb,
                    width=60,
                    height=8,
                    title="Bits per Byte (compression - lower = more structure)"
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
