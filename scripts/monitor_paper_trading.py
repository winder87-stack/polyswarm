#!/usr/bin/env python3
"""
Paper Trading Monitor

Quick monitoring commands for running paper trading sessions.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd: str, description: str = "") -> tuple[str, str]:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", f"Error running command: {e}"

def check_state() -> None:
    """Check current paper trading state."""
    state_file = Path("data/paper_trading_state.json")
    if not state_file.exists():
        print("❌ No active paper trading session found")
        return

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)

        print("📊 CURRENT SESSION STATE:")
        print("=" * 40)

        # Basic info
        start_time = datetime.fromisoformat(state['start_time'])
        runtime = datetime.now() - start_time
        hours = int(runtime.total_seconds() // 3600)
        minutes = int((runtime.total_seconds() % 3600) // 60)

        print(f"⏱️  Runtime: {hours}h {minutes}m")
        print(f"💰 Bankroll: ${state['bankroll']:,.2f}")
        print(f"📈 Open Positions: {len(state['positions'])}")
        print(f"📉 Closed Trades: {len(state['closed_trades'])}")
        print(f"🎯 Total Trades: {state['total_trades']}")
        print(f"💸 Total P&L: ${state['total_pnl']:+,.2f}")
        print(f"🤖 AI Queries: {state['ai_queries']}")

        if state['positions']:
            print("\n📋 OPEN POSITIONS:")
            for pos in state['positions'][:5]:  # Show first 5
                market = pos.get('market_question', 'Unknown')[:40]
                direction = pos.get('direction', '?')
                entry = pos.get('entry_price', 0)
                size = pos.get('size_usd', 0)
                print(f"  • {market}... | {direction} | ${size:.0f} @ {entry:.3f}")

        win_rate = (state['winning_trades'] / max(state['total_trades'], 1)) * 100
        if state['total_trades'] > 0:
            print(f"🎯 Win Rate: {win_rate:.1f}%")
    except Exception as e:
        print(f"❌ Error reading state file: {e}")

def check_recent_reports():
    """Check recent hourly reports."""
    print("\n📈 RECENT HOURLY REPORTS:")
    print("=" * 40)

    stdout, stderr = run_command("grep 'HOURLY REPORT' logs/paper_trading_*.log 2>/dev/null | tail -5")
    if stdout:
        for line in stdout.split('\n'):
            if line.strip():
                # Extract just the report content
                parts = line.split(' - ')
                if len(parts) >= 2:
                    print(parts[1])
    else:
        print("No hourly reports found (session may not have run for an hour yet)")

def check_errors():
    """Check for recent errors."""
    print("\n⚠️  RECENT ERRORS:")
    print("=" * 40)

    stdout, stderr = run_command("grep -i error logs/paper_trading_*.log 2>/dev/null | tail -5")
    if stdout:
        for line in stdout.split('\n'):
            if line.strip():
                print(f"❌ {line}")
    else:
        print("✅ No recent errors found")

def main() -> None:
    """Main monitoring function."""
    print("🌙 Polymarket AI - Paper Trading Monitor")
    print("=" * 50)

    # Check if any log files exist
    log_files = list(Path("logs").glob("paper_trading_*.log"))
    if not log_files:
        print("❌ No paper trading log files found")
        print("💡 Run 'python scripts/paper_trading_24h.py' first")
        return

    # Show current state
    check_state()

    # Show recent reports
    check_recent_reports()

    # Check for errors
    check_errors()

    print("\n🔍 MONITORING COMMANDS:")
    print("=" * 50)
    print("• Watch live: tail -f logs/paper_trading_*.log")
    print("• State check: python scripts/monitor_paper_trading.py")
    print("• Stop trading: Press Ctrl+C in the trading terminal")

if __name__ == "__main__":
    main()
