#!/usr/bin/env python3
"""
Demo script showing the Polymarket AI Trading Bot CLI in action
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a CLI command and show the output."""
    print(f"\n{'='*60}")
    print(f"🎯 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()

    try:
        result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent,
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            # Show only the first 20 lines to keep demo concise
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines[:25]):
                if line.strip():
                    print(line)
            if len(lines) > 25:
                print(f"... ({len(lines) - 25} more lines)")
        else:
            print(f"❌ Command failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run CLI demo."""
    print("🌙 Polymarket AI Trading Bot CLI Demo")
    print("=" * 50)

    # Set up environment
    os.chdir(Path(__file__).parent)
    venv_python = "./venv/bin/python"

    # Demo commands
    commands = [
        ("python main.py --help", "Show CLI help and available commands"),
        ("python main.py status", "Show current bot status"),
        ("python main.py markets --limit 3", "List top 3 markets"),
        ("python main.py scan --top 2", "Scan for top 2 opportunities"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

    print(f"\n{'='*60}")
    print("🎉 CLI Demo Complete!")
    print("Available commands:")
    print("  python main.py markets     # List markets")
    print("  python main.py analyze <slug>  # Analyze specific market")
    print("  python main.py scan        # Scan for opportunities")
    print("  python main.py trade       # Start trading bot")
    print("  python main.py status      # Show bot status")
    print("  python main.py backtest    # Backtest strategy")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
