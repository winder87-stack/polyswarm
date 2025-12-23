#!/usr/bin/env python3
"""
24-Hour Paper Trading Test for Polymarket AI Bot

Runs a comprehensive paper trading simulation for exactly 24 hours with:
- Market scanning every 30 minutes
- Position tracking and P&L calculation
- Comprehensive logging and statistics
- Graceful shutdown with final report
- Resume capability with state persistence

Usage:
    python scripts/paper_trading_24h.py                    # Run 24 hours
    python scripts/paper_trading_24h.py --hours 48         # Run 48 hours
    python scripts/paper_trading_24h.py --interval 15      # Scan every 15 min
    python scripts/paper_trading_24h.py --resume           # Resume previous session

Stop with Ctrl+C for immediate report generation.
"""

import os
import sys
import json
import time
import signal
import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from termcolor import colored, cprint
import logging

# Load environment variables
load_dotenv()

# Force paper trading mode
os.environ['PAPER_TRADING'] = 'true'

from src.agents.trading_swarm import TradingSwarm
from src.connectors import polymarket
from src.strategies.risk_manager import RiskManager


# Configuration
CONFIG = {
    # Duration
    "duration_hours": 24,
    "scan_interval_minutes": 30,

    # STRICT VOLUME REQUIREMENTS (anti-slippage)
    "min_volume": 50000,           # $50k minimum 24h volume
    "min_liquidity": 25000.0,      # $25k minimum liquidity (anti-slippage)
    "max_order_pct_of_liquidity": 0.02,  # Max 2% of liquidity per order
    "max_slippage_pct": 0.02,      # Max 2% slippage

    # Edge requirements
    "min_edge": 0.08,              # 8% minimum edge
    "min_confidence": 0.6,         # 60% model agreement

    # Position sizing
    "max_position_size": 50,       # $50 max per trade (conservative)
    "starting_bankroll": 1000,
    "max_daily_trades": 20,
    "kelly_multiplier": 0.25,      # Quarter Kelly

    # Risk limits
    "max_daily_loss": 100,
    "max_drawdown_pct": 0.15,
}


@dataclass
class PaperPosition:
    """Represents a paper trading position."""
    market_slug: str
    market_question: str
    direction: str  # "YES" or "NO"
    entry_price: float
    size_usd: float
    timestamp: datetime
    market_data: Dict[str, Any]  # Store market info at entry

    def current_value(self, current_price: float) -> float:
        """Calculate current position value."""
        if self.direction == "YES":
            return self.size_usd * (current_price / self.entry_price)
        else:  # NO position
            no_entry_price = 1 - self.entry_price
            no_current_price = 1 - current_price
            return self.size_usd * (no_current_price / no_entry_price)

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return self.current_value(current_price) - self.size_usd

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TradingSession:
    """Tracks the entire trading session."""
    start_time: datetime
    end_time: Optional[datetime] = None
    config: Dict[str, Any] = None
    positions: List[PaperPosition] = None
    closed_trades: List[Dict[str, Any]] = None
    bankroll: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    scans_performed: int = 0
    markets_scanned: int = 0
    signals_generated: int = 0
    signals_skipped: int = 0
    ai_queries: int = 0

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = CONFIG.copy()
        if self.positions is None:
            self.positions = []
        if self.closed_trades is None:
            self.closed_trades = []
        if self.bankroll == 0.0:
            self.bankroll = self.config["starting_bankroll"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        data['positions'] = [pos.to_dict() for pos in self.positions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSession':
        """Create from dictionary (for resuming sessions)."""
        # Convert ISO strings back to datetime
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])

        # Convert positions back to PaperPosition objects
        positions = []
        for pos_data in data.get('positions', []):
            pos_data['timestamp'] = datetime.fromisoformat(pos_data['timestamp'])
            positions.append(PaperPosition(**pos_data))
        data['positions'] = positions

        return cls(**data)


class PaperTradingBot:
    """24-hour paper trading bot with comprehensive logging."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the paper trading bot."""
        self.args = args
        self.session = None
        self.swarm = None
        self.risk_manager = None
        self.logger = None
        self.running = True
        self.state_file = Path("data/paper_trading_state.json")

        # Setup directories
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.trades_dir = Path("data/trades")
        self.trades_dir.mkdir(exist_ok=True)

        # Handle resume functionality
        if args.resume:
            self.resume_session()
        else:
            self.check_for_previous_session()

        # Setup logging
        self.setup_logging()

        # Load trading swarm and risk manager
        self.load_components()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def check_for_previous_session(self):
        """Check if there's a previous session to resume."""
        # If --resume flag was passed, try to resume
        if self.args.resume:
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r') as f:
                        state_data = json.load(f)

                    # Check if session is recent (within 24 hours)
                    start_time = datetime.fromisoformat(state_data['start_time'])
                    hours_since_start = (datetime.now() - start_time).total_seconds() / 3600

                    if hours_since_start < 24:  # Only resume if less than 24 hours old
                        cprint("🔄 Resuming previous session (--resume flag used)", "cyan", attrs=["bold"])
                        self.resume_session()
                        return
                    else:
                        cprint("⚠️  Previous session too old (>24h), starting fresh", "yellow")
                except Exception as e:
                    cprint(f"⚠️  Error reading previous session: {e}", "yellow")
            else:
                cprint("⚠️  No previous session found to resume, starting fresh", "yellow")

        # Start new session (either --resume not used, or resume failed)
        self.session = TradingSession(start_time=datetime.now())
        if not self.args.resume:
            cprint("🆕 Starting new paper trading session...", "green")

    def resume_session(self):
        """Resume a previous session from state file."""
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            self.session = TradingSession.from_dict(state_data)
            cprint("✅ Previous session resumed successfully!", "green", attrs=["bold"])
            cprint(f"   Bankroll: ${self.session.bankroll:.2f}", "green")
            cprint(f"   Open positions: {len(self.session.positions)}", "green")
            cprint(f"   Closed trades: {len(self.session.closed_trades)}", "green")

        except Exception as e:
            cprint(f"❌ Failed to resume session: {e}", "red")
            cprint("🆕 Starting fresh session instead...", "yellow")
            self.session = TradingSession(start_time=datetime.now())

    def save_state_periodically(self):
        """Save current session state every 5 minutes."""
        try:
            state_data = self.session.to_dict()
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"❌ Failed to save periodic state: {e}")

    def setup_logging(self):
        """Setup comprehensive logging."""
        timestamp = self.session.start_time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/paper_trading_{timestamp}.log"

        # Setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.log_filename = log_filename

        self.logger.info("=" * 60)
        self.logger.info("🚀 PAPER TRADING TEST STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Start Time: {self.session.start_time}")
        self.logger.info(f"Duration: {self.args.hours} hours")
        self.logger.info(f"Scan Interval: {self.args.interval} minutes")
        self.logger.info(f"Log File: {log_filename}")
        self.logger.info(f"Starting Bankroll: ${self.session.bankroll}")
        if self.args.resume:
            self.logger.info("Session Type: RESUMED")
        else:
            self.logger.info("Session Type: NEW")
        self.logger.info("=" * 60)

    def load_components(self):
        """Load trading components."""
        try:
            self.logger.info("Loading AI Swarm...")
            self.swarm = TradingSwarm()
            self.logger.info("✅ AI Swarm loaded successfully")

            self.logger.info("Loading Risk Manager...")
            self.risk_manager = RiskManager(initial_bankroll=CONFIG["starting_bankroll"])
            self.logger.info("✅ Risk Manager loaded successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to load components: {e}")
            raise

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def save_session_state(self):
        """Save current session state for potential resume."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = f"logs/session_state_{timestamp}.json"

        try:
            with open(state_file, 'w') as f:
                json.dump(self.session.to_dict(), f, indent=2, default=str)
            self.logger.info(f"💾 Session state saved to {state_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save session state: {e}")

    async def scan_markets_and_trade(self):
        """Scan markets and execute paper trades."""
        self.session.scans_performed += 1

        self.logger.info(f"\n🔍 SCAN #{self.session.scans_performed} - {datetime.now()}")
        self.logger.info("-" * 40)

        try:
            # Scan for opportunities
            opportunities = await self.swarm.find_opportunities(
                min_edge=CONFIG["min_edge"],
                min_confidence=CONFIG["min_confidence"],
                min_volume=CONFIG["min_volume"],
                min_liquidity=CONFIG["min_liquidity"],
                limit=10,
            )
            self.session.markets_scanned += len(opportunities)
            self.session.signals_generated += len(opportunities)

            # Count AI queries (5 models per opportunity analysis)
            self.session.ai_queries += len(opportunities) * 5

            self.logger.info(f"📊 Found {len(opportunities)} opportunities")

            if opportunities:
                self.logger.info(f"📊 Top opportunities in high-volume markets:")
                for opp in opportunities[:5]:
                    self.logger.info(
                        f"   • {opp.market.question[:40]}... | "
                        f"Vol: ${opp.market.volume:,.0f} | "
                        f"Edge: {opp.edge*100:.1f}%"
                    )

            # Filter opportunities based on our criteria
            valid_opportunities = []
            skipped_count = 0

            for opp in opportunities:
                if (opp.edge >= CONFIG["min_edge"] and
                    opp.confidence >= CONFIG["min_confidence"] and
                    opp.market.liquidity >= CONFIG["min_liquidity"] and
                    opp.market.volume >= CONFIG["min_volume"]):

                    # Check if we can afford the position
                    max_position = min(CONFIG["max_position_size"], self.session.bankroll * 0.1)  # Max 10% of bankroll
                    if max_position >= 10:  # Minimum $10 trade
                        valid_opportunities.append((opp, max_position))
                    else:
                        skipped_count += 1
                        self.logger.debug(f"⏭️  Skipped {opp.market.question[:30]}... - insufficient funds")
                else:
                    skipped_count += 1
                    self.logger.debug(f"⏭️  Skipped {opp.market.question[:30]}... - edge:{opp.edge:.1%} conf:{opp.confidence:.1%}")

            self.session.signals_skipped += skipped_count
            self.logger.info(f"✅ {len(valid_opportunities)} valid opportunities after filtering")
            self.logger.info(f"⏭️  {skipped_count} signals skipped (low edge/confidence)")

            # Execute trades (limit to prevent over-trading)
            trades_executed = 0
            for opp, max_size in valid_opportunities[:3]:  # Max 3 trades per scan
                if self.session.total_trades >= CONFIG["max_daily_trades"]:
                    self.logger.info("🚫 Daily trade limit reached")
                    break

                # Execute trade with slippage logging
                await self.execute_trade(opp, max_size)
                trades_executed += 1

            if trades_executed > 0:
                self.logger.info(f"✅ Executed {trades_executed} paper trades this scan")

        except Exception as e:
            self.logger.error(f"❌ Error during market scan: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def execute_trade(self, signal, max_size):
        """Execute trade with slippage logging"""

        market = signal.market

        # Calculate recommended size using risk manager
        current_bankroll = self.session.bankroll
        recommended_size = self.risk_manager.calculate_position_size(signal, current_bankroll)

        # Calculate max safe size
        max_size_by_liquidity = market.liquidity * CONFIG["max_order_pct_of_liquidity"]
        safe_size = min(recommended_size, max_size_by_liquidity, CONFIG["max_position_size"])

        # Estimate slippage
        slippage_pct = (safe_size / market.liquidity) * 0.5  # Simple model

        self.logger.info(f"💰 PAPER TRADE:")
        self.logger.info(f"   Market: {market.question[:50]}...")
        self.logger.info(f"   Direction: {signal.direction}")
        self.logger.info(f"   Size: ${safe_size:.2f} (of ${recommended_size:.2f} recommended)")
        self.logger.info(f"   Price: ${signal.market_probability:.3f}")
        self.logger.info(f"   Volume: ${market.volume:,.0f}")
        self.logger.info(f"   Liquidity: ${market.liquidity:,.0f}")
        self.logger.info(f"   Est. Slippage: {slippage_pct*100:.2f}%")
        self.logger.info(f"   Edge: {signal.edge*100:.1f}%")

        # Create paper position
        position = PaperPosition(
            market_slug=getattr(market, 'slug', 'unknown'),
            market_question=market.question,
            direction=signal.direction,
            entry_price=signal.market_probability,
            size_usd=safe_size,
            timestamp=datetime.now(),
            market_data={
                'yes_price': market.yes_price,
                'no_price': market.no_price,
                'liquidity': market.liquidity,
                'volume': market.volume
            }
        )

        self.session.positions.append(position)
        self.session.bankroll -= position.size_usd
        self.session.total_trades += 1

        # Update risk manager with new portfolio value
        current_portfolio_value = self.session.bankroll + sum(
            pos.current_value(pos.market_data.get('yes_price', pos.entry_price))
            for pos in self.session.positions
        )
        self.risk_manager.update_portfolio_value(current_portfolio_value)

    def update_positions(self):
        """Update position values and check for exit conditions."""
        if not self.session.positions:
            return

        try:
            self.logger.info(f"\n📊 POSITION UPDATES - {datetime.now()}")
            self.logger.info("-" * 40)

            # Get current market data for our positions
            updated_positions = []

            for position in self.session.positions:
                try:
                    # Try to fetch current market data (may fail in paper trading mode)
                    market = polymarket.get_market_by_slug(position.market_slug)
                    if market:
                        current_price = market.yes_price
                        current_value = position.current_value(current_price)
                        unrealized_pnl = position.unrealized_pnl(current_price)

                        self.logger.info(f"📈 {position.market_question[:40]}...")
                        self.logger.info(f"   Direction: {position.direction} | Entry: {position.entry_price:.3f}")
                        self.logger.info(f"   Current: {current_price:.3f} | Value: ${current_value:.2f}")
                        self.logger.info(f"   P&L: ${unrealized_pnl:+.2f} ({unrealized_pnl/position.size_usd:+.1%})")

                        # Check exit conditions (simplified: close if 50% profit or 20% loss)
                        if unrealized_pnl >= position.size_usd * 0.5:  # 50% profit
                            self.close_position(position, current_price, "TAKE PROFIT")
                        elif unrealized_pnl <= -position.size_usd * 0.2:  # 20% loss
                            self.close_position(position, current_price, "STOP LOSS")
                        else:
                            updated_positions.append(position)
                    else:
                        # Paper trading: no live market data available, keep position as-is
                        self.logger.info(f"📊 {position.market_question[:40]}... (paper trading - no live updates)")
                        self.logger.info(f"   Direction: {position.direction} | Entry: {position.entry_price:.3f}")
                        self.logger.info(f"   Size: ${position.size_usd:.2f} | Holding until end of test")
                        updated_positions.append(position)

                except Exception as e:
                    # Paper trading: API doesn't support individual market fetching
                    self.logger.info(f"📊 {position.market_question[:40]}... (paper trading - API limitation)")
                    self.logger.info(f"   Direction: {position.direction} | Entry: {position.entry_price:.3f}")
                    self.logger.info(f"   Size: ${position.size_usd:.2f} | Holding until end of test")
                    updated_positions.append(position)

            self.session.positions = updated_positions

        except Exception as e:
            self.logger.error(f"❌ Fatal error in position updates: {e}")
            # Continue with existing positions

    def close_position(self, position: PaperPosition, exit_price: float, reason: str):
        """Close a position and record the trade."""
        try:
            exit_value = position.current_value(exit_price)
            realized_pnl = position.unrealized_pnl(exit_price)

            # Calculate edge at entry (this would be stored in position data ideally, but we'll estimate)
            market_price = position.market_data.get('yes_price', position.entry_price)
            edge = abs(position.entry_price - market_price) if position.direction == "YES" else abs((1 - position.entry_price) - (1 - market_price))

            # Record the closed trade
            trade_record = {
                "market_slug": position.market_slug,
                "market_question": position.market_question,
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "market_price_at_entry": market_price,
                "size_usd": position.size_usd,
                "exit_value": exit_value,
                "realized_pnl": realized_pnl,
                "return_pct": realized_pnl / position.size_usd if position.size_usd != 0 else 0,
                "edge": edge,  # Estimated edge at entry
                "confidence": 0.7,  # Default confidence (would be stored with position)
                "entry_time": position.timestamp.isoformat(),
                "exit_time": datetime.now().isoformat(),
                "reason": reason,
                "holding_period_hours": (datetime.now() - position.timestamp).total_seconds() / 3600
            }

            self.session.closed_trades.append(trade_record)
            self.session.bankroll += exit_value  # Add back the position value
            self.session.total_pnl += realized_pnl

            # Update risk manager with new portfolio value
            current_portfolio_value = self.session.bankroll + sum(
                pos.current_value(pos.market_data.get('yes_price', pos.entry_price))
                for pos in self.session.positions
            )
            self.risk_manager.update_portfolio_value(current_portfolio_value)

            # Record trade performance for risk metrics
            self.risk_manager.record_trade_performance(realized_pnl)

            if realized_pnl > 0:
                self.session.winning_trades += 1
            else:
                self.session.losing_trades += 1

            self.logger.info(f"🔒 CLOSED POSITION: {position.market_question[:40]}...")
            self.logger.info(f"   Reason: {reason} | P&L: ${realized_pnl:+.2f} ({realized_pnl/position.size_usd:+.1%})")
            self.logger.info(f"   Bankroll: ${self.session.bankroll:.2f}")

        except Exception as e:
            self.logger.error(f"❌ Error closing position {position.market_slug}: {e}")
            # Still remove from positions to prevent infinite loops
            if position in self.session.positions:
                self.session.positions.remove(position)

    def print_hourly_report(self):
        """Print detailed hourly statistics report."""
        runtime = datetime.now() - self.session.start_time
        hours_elapsed = int(runtime.total_seconds() // 3600)
        minutes_elapsed = int((runtime.total_seconds() % 3600) // 60)
        hours_remaining = max(0, CONFIG['duration_hours'] - hours_elapsed - 1)
        minutes_remaining = 0 if hours_remaining > 0 else max(0, 60 - minutes_elapsed)

        current_value = self.session.bankroll + sum(pos.current_value(pos.market_data.get('yes_price', pos.entry_price)) for pos in self.session.positions)
        return_pct = ((current_value / CONFIG['starting_bankroll']) - 1) * 100

        win_rate = (self.session.winning_trades / max(self.session.total_trades, 1)) * 100

        # Count AI queries (approximate: 5 models per scan, plus position updates)
        ai_queries = self.session.scans_performed * 5 + (len(self.session.positions) * 2)

        print("\n" + "=" * 60)
        print(f"📊 HOURLY REPORT - Hour {hours_elapsed + 1} of {CONFIG['duration_hours']}")
        print("=" * 60)
        print(f"⏱️  Runtime: {hours_elapsed}h {minutes_elapsed}m | Remaining: {hours_remaining}h {minutes_remaining}m")
        # Record daily performance for risk metrics
        daily_return_pct = return_pct / 100.0  # Convert to decimal
        self.risk_manager.record_daily_performance(daily_return_pct)

        print(f"🛡️ VOLUME FILTERS ACTIVE:")
        print(f"   Min Volume: ${CONFIG['min_volume']:,.0f}")
        print(f"   Min Liquidity: ${CONFIG['min_liquidity']:,.0f}")
        print(f"   Max Order % of Liquidity: {CONFIG.get('max_order_pct_of_liquidity', 0.02)*100:.1f}%")

        print(f"📊 MARKET ANALYSIS:")
        print(f"   Markets Scanned: {self.session.markets_scanned}")
        print(f"   Signals Generated: {self.session.signals_generated}")
        print(f"   Signals Skipped: {self.session.signals_skipped}")

        print(f"💰 PERFORMANCE:")
        print(f"   Starting Bankroll: ${CONFIG['starting_bankroll']:,.2f}")
        print(f"   Current Value: ${current_value:,.2f} ({return_pct:+.1f}%)")
        print(f"   Open Positions: {len(self.session.positions)}")
        print(f"   Closed Trades: {len(self.session.closed_trades)} ({self.session.winning_trades} wins, {self.session.losing_trades} losses)")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${self.session.total_pnl:+.2f}")
        print(f"🤖 AI Queries: {ai_queries} | Cost: ${self.swarm.swarm.total_cost:.2f}")
        print("=" * 60)

    def print_final_report(self):
        """Print comprehensive final report."""
        runtime = self.session.end_time - self.session.start_time
        total_hours = int(runtime.total_seconds() // 3600)
        total_minutes = int((runtime.total_seconds() % 3600) // 60)
        total_seconds = int(runtime.total_seconds() % 60)

        final_value = self.session.bankroll
        total_return = final_value - CONFIG['starting_bankroll']
        return_pct = (total_return / CONFIG['starting_bankroll']) * 100

        win_rate = (self.session.winning_trades / max(self.session.total_trades, 1)) * 100

        # Calculate averages
        winning_trades = [t for t in self.session.closed_trades if t['realized_pnl'] > 0]
        losing_trades = [t for t in self.session.closed_trades if t['realized_pnl'] <= 0]

        avg_win = sum(t['realized_pnl'] for t in winning_trades) / max(len(winning_trades), 1)
        avg_loss = sum(t['realized_pnl'] for t in losing_trades) / max(len(losing_trades), 1)

        # Profit factor
        total_wins = sum(t['realized_pnl'] for t in winning_trades)
        total_losses = abs(sum(t['realized_pnl'] for t in losing_trades))
        profit_factor = total_wins / max(total_losses, 0.01)

        # Edge analysis
        all_edges = [t.get('edge', 0) for t in self.session.closed_trades if 'edge' in t]
        all_confidences = [t.get('confidence', 0) for t in self.session.closed_trades if 'confidence' in t]
        avg_edge = sum(all_edges) / max(len(all_edges), 1) * 100
        avg_confidence = sum(all_confidences) / max(len(all_confidences), 1) * 100

        # Cost per trade
        cost_per_trade = self.swarm.swarm.total_cost / max(self.session.total_trades, 1)

        duration_text = f"{self.args.hours}-HOUR" if self.args.hours != 24 else "24-HOUR"
        print("\n" + "=" * 51)
        print(f"🏁 {duration_text} PAPER TRADING TEST COMPLETE")
        print("=" * 51)
        print(f"📅 Started: {self.session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 Ended: {self.session.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Runtime: {total_hours}h {total_minutes}m {total_seconds}s")
        print("💰 PERFORMANCE:")
        print(f"Starting Bankroll: ${CONFIG['starting_bankroll']:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total P&L: ${total_return:+,.2f} ({return_pct:+.2f}%)")
        print("📊 TRADING STATS:")
        print(f"Total Signals Generated: {self.session.signals_generated}")
        print(f"Trades Executed: {self.session.total_trades}")
        print(f"Winning Trades: {self.session.winning_trades} ({win_rate:.1f}%)")
        print(f"Losing Trades: {self.session.losing_trades}")
        print(f"Average Win: ${avg_win:+.2f}")
        print(f"Average Loss: ${avg_loss:+.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print("🎯 EDGE ANALYSIS:")
        print(f"Average Edge at Entry: {avg_edge:.1f}%")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        print(f"Signals Skipped (low edge): {self.session.signals_skipped}")

        # Get performance report from risk manager
        perf_report = self.risk_manager.get_performance_report()
        if 'sharpe_ratio' in perf_report:
            print(f"📊 RISK METRICS:")
            print(f"Sharpe Ratio: {perf_report['sharpe_ratio']:.2f}")
            print(f"Sortino Ratio: {perf_report['sortino_ratio']:.2f}")
            print(f"Max Drawdown: {perf_report['max_drawdown']:.1%}")
            print(f"Current Drawdown: {perf_report['current_drawdown']:.1%}")
            print(f"Profit Factor: {perf_report['profit_factor']:.2f}")

        print("🤖 AI COSTS:")
        print(f"Total Queries: {self.session.ai_queries}")
        print(f"Total API Cost: ${self.swarm.swarm.total_cost:.2f}")
        print(f"Cost per Trade: ${cost_per_trade:.2f}")
        print(f"📁 Full log saved to: {self.log_filename}")
        print(f"📁 Trade history saved to: {self.data_dir}/paper_trades_{self.session.start_time.strftime('%Y%m%d_%H%M%S')}.json")
        print("=" * 51)

    def print_statistics(self):
        """Print current session statistics."""
        runtime = datetime.now() - self.session.start_time
        hours_elapsed = runtime.total_seconds() / 3600

        win_rate = (self.session.winning_trades / max(self.session.total_trades, 1)) * 100
        avg_trade_size = sum(t['size_usd'] for t in self.session.closed_trades) / max(len(self.session.closed_trades), 1)

        self.logger.info(f"\n📊 SESSION STATISTICS - {datetime.now()}")
        self.logger.info("=" * 60)
        self.logger.info(f"Runtime: {hours_elapsed:.1f} hours")
        self.logger.info(f"Scans Performed: {self.session.scans_performed}")
        self.logger.info(f"Markets Scanned: {self.session.markets_scanned}")
        self.logger.info(f"Total Trades: {self.session.total_trades}")
        self.logger.info(f"Open Positions: {len(self.session.positions)}")
        self.logger.info(f"Closed Trades: {len(self.session.closed_trades)}")
        self.logger.info(f"Winning Trades: {self.session.winning_trades}")
        self.logger.info(f"Losing Trades: {self.session.losing_trades}")
        self.logger.info(f"Win Rate: {win_rate:.1f}%")
        self.logger.info(f"Average Trade Size: ${avg_trade_size:.2f}")
        self.logger.info(f"Total P&L: ${self.session.total_pnl:+.2f}")
        self.logger.info(f"Current Bankroll: ${self.session.bankroll:.2f}")
        self.logger.info(f"Return: ${(self.session.bankroll - CONFIG['starting_bankroll']):+.2f} ({((self.session.bankroll/CONFIG['starting_bankroll'])-1)*100:+.1f}%)")
        self.logger.info("=" * 60)

    async def run(self):
        """Run the paper trading test."""
        duration_hours = self.args.hours
        scan_interval_minutes = self.args.interval

        self.logger.info(f"🎯 Starting {duration_hours}-hour paper trading test...")
        self.logger.info(f"Duration: {duration_hours} hours")
        self.logger.info(f"Scan Interval: {scan_interval_minutes} minutes")

        end_time = self.session.start_time + timedelta(hours=duration_hours)
        scan_interval_seconds = scan_interval_minutes * 60

        last_scan_time = datetime.now() - timedelta(seconds=scan_interval_seconds)  # Start ready for immediate scan
        last_update_time = datetime.now()
        last_hourly_report = datetime.now()
        last_state_save = datetime.now()

        try:
            while self.running and datetime.now() < end_time:
                current_time = datetime.now()

                # Perform market scan
                if (current_time - last_scan_time).total_seconds() >= scan_interval_seconds:
                    try:
                        await self.scan_markets_and_trade()
                        self.save_state_periodically()  # Save state after each scan
                    except Exception as e:
                        self.logger.error(f"❌ Error during market scan: {e}")
                        self.logger.error("Continuing with next scan...")
                    last_scan_time = current_time

                # Update positions every 5 minutes
                if (current_time - last_update_time).total_seconds() >= 300:  # 5 minutes
                    self.update_positions()
                    last_update_time = current_time

                # Print hourly report every hour
                if (current_time - last_hourly_report).total_seconds() >= 3600:  # 1 hour
                    self.print_hourly_report()
                    last_hourly_report = current_time

                # Save state every 5 minutes
                if (current_time - last_state_save).total_seconds() >= 300:  # 5 minutes
                    self.save_state_periodically()
                    last_state_save = current_time

                # Brief pause to prevent excessive CPU usage
                time.sleep(10)

        except Exception as e:
            self.logger.error(f"❌ Fatal error during trading: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        finally:
            # Final position updates
            self.logger.info("\n🏁 TEST COMPLETION - Final Updates")
            self.update_positions()

            # Close all remaining positions (paper trading - use entry price since API doesn't support live fetches)
            self.logger.info("\n🔒 CLOSING ALL REMAINING POSITIONS")
            positions_to_close = self.session.positions.copy()
            for position in positions_to_close:
                try:
                    # Paper trading: API doesn't support individual market fetching, close at entry price
                    self.logger.info(f"🔒 Closing {position.market_question[:40]}... at entry price")
                    self.close_position(position, position.entry_price, "END OF TEST")
                except Exception as e:
                    self.logger.error(f"❌ Error closing position {position.market_slug}: {e}")
                    # Force close at entry price
                    try:
                        self.close_position(position, position.entry_price, "END OF TEST (ERROR)")
                    except:
                        pass  # Last resort - just remove from session
                    # Force close at entry price as fallback
                    try:
                        self.close_position(position, position.entry_price, "END OF TEST (ERROR)")
                    except Exception as e2:
                        self.logger.error(f"❌ Failed to force close position: {e2}")
                        # Last resort: just remove from positions
                        if position in self.session.positions:
                            self.session.positions.remove(position)

            # Final session data
            self.session.end_time = datetime.now()

            # Save trade history
            trade_history_file = f"{self.data_dir}/paper_trades_{self.session.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            try:
                trade_data = {
                    "session_info": {
                        "start_time": self.session.start_time.isoformat(),
                        "end_time": self.session.end_time.isoformat(),
                        "config": self.session.config,
                        "final_bankroll": self.session.bankroll,
                        "total_pnl": self.session.total_pnl
                    },
                    "closed_trades": self.session.closed_trades,
                    "final_positions": [pos.to_dict() for pos in self.session.positions]
                }
                with open(trade_history_file, 'w') as f:
                    json.dump(trade_data, f, indent=2, default=str)
                self.logger.info(f"💾 Trade history saved to {trade_history_file}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save trade history: {e}")

            # Print comprehensive final report
            self.print_final_report()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🌙 Polymarket AI - Paper Trading Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/paper_trading_24h.py                    # Run 24 hours
  python scripts/paper_trading_24h.py --hours 48         # Run 48 hours
  python scripts/paper_trading_24h.py --interval 15      # Scan every 15 min
  python scripts/paper_trading_24h.py --resume           # Resume previous session

Stop with Ctrl+C for immediate final report.
        """
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Duration of the test in hours (default: 24)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Market scan interval in minutes (default: 30)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous session if available"
    )

    args = parser.parse_args()

    duration_text = f"{args.hours}-Hour" if args.hours != 24 else "24-Hour"
    cprint(f"🌙 Polymarket AI - {duration_text} Paper Trading Test", "cyan", attrs=["bold"])
    cprint("=" * 60, "cyan")

    if args.resume:
        cprint("🔄 Resume mode: Will attempt to resume previous session", "yellow")
    else:
        cprint("🆕 Fresh session: Starting new paper trading test", "green")

    cprint("Use Ctrl+C to stop early and get a final report", "yellow")
    print()

    # Configuration summary
    cprint("📋 Test Configuration:", "green")
    cprint(f"  Duration: {args.hours} hours", "white")
    cprint(f"  Scan Interval: {args.interval} minutes", "white")
    cprint(f"  Starting Bankroll: ${CONFIG['starting_bankroll']}", "white")
    cprint(f"  Min Edge: {CONFIG['min_edge']:.1%}", "white")
    cprint(f"  Min Confidence: {CONFIG['min_confidence']:.1%}", "white")
    cprint(f"  Min Volume: ${CONFIG['min_volume']:,.0f}", "white")
    cprint(f"  Min Liquidity: ${CONFIG['min_liquidity']:,.0f}", "white")
    cprint(f"  Max Position Size: ${CONFIG['max_position_size']}", "white")
    print()

    try:
        bot = PaperTradingBot(args)
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        cprint("\n👋 Test interrupted by user", "yellow")
    except Exception as e:
        cprint(f"❌ Fatal error: {e}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
