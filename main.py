#!/usr/bin/env python3
"""
🌙 Polymarket AI Trading Bot - Main CLI Entry Point

A sophisticated automated trading system for Polymarket prediction markets
using multi-model AI analysis, advanced risk management, and Kelly criterion sizing.

Author: Polymarket Trading Bot
Version: 1.0.0
"""

import os
import sys
import argparse
import signal
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from termcolor import colored
from loguru import logger

# Import bot components
from connectors import polymarket
from agents import TradingSwarm
from models import model_factory
from strategies import RiskManager, RiskLimits


# Version and metadata
VERSION = "1.0.0"
BANNER = f"""
🌙 Polymarket AI Trading Bot v{VERSION}
═══════════════════════════════════════════════════════════════
🤖 Multi-Model AI Analysis | 🛡️ Advanced Risk Management
💰 Kelly Criterion Sizing | 📊 Real-time P&L Tracking
═══════════════════════════════════════════════════════════════
"""


class TradingBotCLI:
    """Main CLI interface for the Polymarket AI Trading Bot."""

    def __init__(self) -> None:
        """Initialize CLI with components."""
        self.swarm = None
        self.risk_manager = None
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure loguru logging."""
        # Remove default handler
        logger.remove()

        # Console logging with colors
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )

        # File logging
        log_file = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
        os.makedirs("logs", exist_ok=True)
        logger.add(
            log_file,
            rotation="1 day",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )

        logger.info("🚀 Trading Bot CLI initialized")

    def load_components(self) -> None:
        """Lazy load trading components."""
        if not self.swarm:
            try:
                logger.info("Loading AI Swarm...")
                self.swarm = TradingSwarm()
            except Exception as e:
                logger.error(f"Failed to load AI Swarm: {e}")
                raise

        if not self.risk_manager:
            try:
                logger.info("Loading Risk Manager...")
                limits = RiskLimits()
                self.risk_manager = RiskManager(limits)
            except Exception as e:
                logger.error(f"Failed to load Risk Manager: {e}")
                raise

    async def show_markets(self, limit=20, category=None, min_volume=0):
        """List top markets with filtering."""
        print(colored("📊 Fetching markets...", "blue"))

        try:
            # Get markets with filtering
            markets = await polymarket.get_markets(limit=limit * 2, min_volume=min_volume, active_only=True)

            # Apply filters
            filtered_markets = []
            for market in markets:
                if market.volume < min_volume:
                    continue
                if category and category.lower() not in market.category.lower():
                    continue
                filtered_markets.append(market)

            # Sort by volume and take top N
            filtered_markets.sort(key=lambda x: x.volume, reverse=True)
            display_markets = filtered_markets[:limit]

            if not display_markets:
                print(colored("❌ No markets found matching criteria", "red"))
                return

            print(colored(f"\n📈 TOP {len(display_markets)} MARKETS", "cyan", attrs=["bold"]))
            print(colored("═" * 80, "cyan"))

            for i, market in enumerate(display_markets, 1):
                print(colored(f"{i:2d}. ", "white"), end="")

                # Truncate long titles
                title = market.question
                if len(title) > 60:
                    title = title[:57] + "..."

                print(colored(f"{title}", "white"))

                # Format prices
                yes_price = colored(f"{market.yes_price:.1%}", "green")
                no_price = colored(f"{market.no_price:.1%}", "red")

                # Format volume and liquidity
                volume = market.volume
                liquidity = market.liquidity

                vol_str = f"${volume:,.0f}" if volume >= 1000 else f"${volume:.0f}"
                liq_str = f"${liquidity:,.0f}" if liquidity >= 1000 else f"${liquidity:.0f}"

                print(colored(f"      YES: {yes_price} | NO: {no_price} | Vol: {vol_str} | Liq: {liq_str}", "grey"))

                # Category
                print(colored(f"      Category: {market.category.upper()}", "yellow"))

                # End date
                if market.end_date:
                    try:
                        # Parse string end_date to datetime
                        end_date = datetime.fromisoformat(market.end_date.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        if end_date.tzinfo is None:
                            end_date = end_date.replace(tzinfo=timezone.utc)

                        days_until = (end_date - now).days
                        if days_until > 0:
                            date_str = f"{end_date.strftime('%Y-%m-%d')} ({days_until}d)"
                        else:
                            date_str = f"{end_date.strftime('%Y-%m-%d')} (EXPIRED)"
                    except Exception:
                        # Fallback if datetime parsing fails
                        date_str = market.end_date[:10] if len(market.end_date) >= 10 else market.end_date
                    print(colored(f"      Expires: {date_str}", "grey"))
                else:
                    print(colored("      Expires: No end date", "grey"))

                print()

            # Summary stats
            total_volume = sum(m.volume for m in display_markets)
            total_liquidity = sum(m.liquidity for m in display_markets)
            avg_volume = total_volume / len(display_markets)

            print(colored("📊 SUMMARY", "cyan", attrs=["bold"]))
            print(colored(f"   Total Volume: ${total_volume:,.0f}", "white"))
            print(colored(f"   Average Volume: ${avg_volume:,.0f}", "white"))
            print(colored(f"   Total Liquidity: ${total_liquidity:,.0f}", "white"))

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            print(colored(f"❌ Failed to fetch markets: {e}", "red"))

    def analyze_market(self, market_slug, verbose=False):
        """Analyze a specific market."""
        print(colored(f"🎯 Analyzing market: {market_slug}", "blue", attrs=["bold"]))

        try:
            # Get market details
            market = polymarket.get_market_by_slug(market_slug)
            if not market:
                print(colored(f"❌ Market not found: {market_slug}", "red"))
                return

            print(colored(f"📊 {market.question}", "white"))
            print(colored(f"   Category: {market.category.upper()}", "yellow"))
            print(colored(f"   Volume: ${market.volume:,.0f}", "white"))

            # Load components for analysis
            self.load_components()

            # Analyze the market
            signal = self.swarm.analyze_market(market)

            if not signal:
                print(colored("❌ Analysis failed", "red"))
                return

            # Display results
            print(colored("\n📈 ANALYSIS RESULTS", "green", attrs=["bold"]))
            print(colored("═" * 50, "green"))

            print(colored("🎲 PROBABILITY ANALYSIS", "cyan"))
            print(colored(f"   Weighted Probability: {signal.probability:.1%}", "white"))
            print(colored(f"   Unweighted Average: {signal.probability:.1%}", "grey"))
            print(colored(f"   Confidence Score: {signal.confidence:.1%}", "white"))

            print(colored("\n💰 MARKET POSITION", "cyan"))
            print(colored(f"   Market Price: {signal.market_probability:.1%}", "white"))
            print(colored(f"   Edge: {signal.edge:+.1%}", "green" if signal.edge > 0 else "red"))

            print(colored("\n💎 EXPECTED VALUE", "cyan"))
            print(colored(f"   Expected Value: {signal.expected_value:.1%}", "magenta"))

            if signal.expected_value > 0:
                print(colored("   ✅ POSITIVE EXPECTED VALUE - Good trade opportunity!", "green", attrs=["bold"]))
            else:
                print(colored("   ❌ NEGATIVE EXPECTED VALUE - Avoid this trade", "red", attrs=["bold"]))

            print(colored("\n🤖 MODEL BREAKDOWN", "cyan"))
            for model, prob in signal.model_votes.items():
                weight = signal.model_weights.get(model, 1.0)
                weighted_prob = prob * weight
                print(colored(f"   {model}: {prob:.1%} (weight: {weight:.1f}) → {weighted_prob:.1%}", "white"))

            if verbose:
                print(colored("\n📝 CONSENSUS SUMMARY", "cyan"))
                print(colored(signal.consensus_summary, "white"))

                if signal.news_context:
                    print(colored("\n📰 NEWS CONTEXT", "cyan"))
                    news_preview = signal.news_context[:300] + "..." if len(signal.news_context) > 300 else signal.news_context
                    print(colored(news_preview, "white"))

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            print(colored(f"❌ Analysis failed: {e}", "red"))

    def scan_opportunities(self, min_edge=0.10, min_volume=10000, top=5):
        """Scan for trading opportunities."""
        print(colored("🔍 Scanning for trading opportunities...", "blue", attrs=["bold"]))

        try:
            self.load_components()

            # Find opportunities
            opportunities = self.swarm.find_opportunities(limit=top * 3)

            # Filter by edge
            opportunities = [opp for opp in opportunities if opp.edge >= min_edge]

            # Sort by expected value and take top N
            opportunities.sort(key=lambda x: x.expected_value, reverse=True)
            top_opportunities = opportunities[:top]

            if not top_opportunities:
                print(colored("❌ No opportunities found matching criteria", "red"))
                return

            print(colored(f"\n🎯 TOP {len(top_opportunities)} OPPORTUNITIES (EV ≥ {min_edge:.1%})", "green", attrs=["bold"]))
            print(colored("═" * 80, "green"))

            for i, opp in enumerate(top_opportunities, 1):
                print(colored(f"{i}. ", "white"), end="")

                # Market title
                title = opp.market.question
                if len(title) > 50:
                    title = title[:47] + "..."
                print(colored(f"{title}", "white"))

                # Key metrics
                prob_str = colored(f"{opp.probability:.1%}", "cyan")
                edge_str = colored(f"{opp.edge:+.1%}", "green" if opp.edge > 0 else "red")
                ev_str = colored(f"{opp.expected_value:.1f}", "magenta")

                print(colored(f"   Probability: {prob_str} | Edge: {edge_str} | EV: {ev_str}", "white"))

                # Confidence and volume
                conf_str = colored(f"{opp.confidence:.1%}", "yellow")
                vol_str = f"${opp.market.volume:,.0f}"
                print(colored(f"   Confidence: {conf_str} | Volume: {vol_str}", "grey"))

                # Direction recommendation
                direction = colored(opp.direction, "green" if opp.direction == "YES" else "red", attrs=["bold"])
                print(colored(f"   Recommended: {direction}", "white"))

                print()

        except Exception as e:
            logger.error(f"Opportunity scan failed: {e}")
            print(colored(f"❌ Scan failed: {e}", "red"))

    async def start_trading(self, paper=True, live=False, interval=60):
        """Start the automated trading bot."""
        if live and not paper:
            # Require confirmation for live trading
            print(colored("⚠️  LIVE TRADING MODE - This will execute real trades!", "red", attrs=["bold"]))
            confirmation = input(colored("Type 'YES' to confirm live trading: ", "red")).strip()
            if confirmation != "YES":
                print(colored("❌ Live trading cancelled", "yellow"))
                return

        mode = "LIVE" if live else "PAPER"
        print(colored(f"🚀 Starting {mode} trading bot (interval: {interval}min)", "cyan", attrs=["bold"]))

        try:
            from agents import run_trading_bot
            run_trading_bot(scan_interval_minutes=interval)
        except KeyboardInterrupt:
            print(colored("\n👋 Trading bot stopped by user", "yellow"))
        except Exception as e:
            logger.error(f"Trading bot failed: {e}")
            print(colored(f"❌ Trading bot failed: {e}", "red"))

    def show_status(self):
        """Show current bot status."""
        print(colored("📊 BOT STATUS", "cyan", attrs=["bold"]))
        print(colored("═" * 40, "cyan"))

        try:
            # Risk manager status
            if self.risk_manager:
                stats = self.risk_manager.get_daily_stats()

                print(colored("💰 P&L TRACKING", "green"))
                print(colored(f"   Daily P&L: ${stats['daily_pnl']:.2f}", "white"))
                print(colored(f"   Current Drawdown: {stats['current_drawdown']:.1%}", "white"))
                print(colored(f"   Active Positions: {stats['active_positions']}", "white"))
                print(colored(f"   Total Exposure: ${stats['total_exposure']:.2f}", "white"))

                if stats['trading_paused']:
                    print(colored(f"   ⚠️  Trading Paused: {stats['pause_reason']}", "red"))

                print()

            # AI Swarm status
            if self.swarm:
                print(colored("🤖 AI SWARM STATUS", "blue"))
                print(colored(f"   Models Loaded: {len(self.swarm.models)}", "white"))
                print(colored(f"   Signals Generated: {self.swarm.signals_generated}", "white"))
                print(colored(f"   Total API Cost: ${self.swarm.total_cost:.4f}", "white"))

                # Show cost breakdown
                cost_breakdown = self.swarm.get_cost_summary()
                for model, cost in cost_breakdown['cost_by_model'].items():
                    if cost > 0:
                        print(colored(f"   {model}: ${cost:.4f}", "grey"))

                print()

            # System status
            print(colored("🖥️  SYSTEM STATUS", "yellow"))
            print(colored(f"   Version: {VERSION}", "white"))
            print(colored(f"   Mode: {'PAPER' if os.getenv('PAPER_TRADING', 'true').lower() == 'true' else 'LIVE'}", "white"))
            print(colored(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white"))

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            print(colored(f"❌ Status check failed: {e}", "red"))

    def backtest_strategy(self, days=30):
        """Backtest the trading strategy."""
        print(colored(f"📈 Backtesting strategy for {days} days...", "blue", attrs=["bold"]))

        # Placeholder for backtesting functionality
        # In a real implementation, this would:
        # 1. Load historical market data
        # 2. Simulate trades based on signals
        # 3. Calculate P&L, win rate, Sharpe ratio, etc.

        print(colored("⚠️  Backtesting not yet implemented", "yellow"))
        print(colored("   This feature requires historical market data integration", "grey"))

    def collect_historical_data(self, days=365):
        """Collect historical market data."""
        print(colored(f"📚 COLLECTING HISTORICAL DATA - {days} days", "cyan", attrs=["bold"]))
        print(colored("═" * 55, "cyan"))

        try:
            from data import create_historical_collector

            # Create collector
            collector = create_historical_collector()
            print(colored("✅ Connected to historical database", "green"))

            if days >= 365:
                print(colored("🚀 Starting full historical backfill...", "yellow"))
                # Run async collection
                import asyncio
                stats = asyncio.run(collector.backfill_all_data())
                print(colored("✅ Full backfill completed!", "green"))
                print(colored(f"   Markets collected: {stats['markets_collected']}", "white"))
                print(colored(f"   Price snapshots: {stats['price_snapshots']}", "white"))
            else:
                print(colored(f"📅 Collecting data from last {days} days...", "yellow"))
                # Run daily collection
                import asyncio
                from data import create_collection_scheduler
                scheduler = create_collection_scheduler(collector)
                stats = asyncio.run(scheduler.daily_collection_job())
                print(colored("✅ Daily collection completed!", "green"))
                print(colored(f"   New markets: {stats['new_markets']}", "white"))

            # Show database stats
            db_stats = collector.get_collection_stats()
            print(colored("\n📊 DATABASE STATISTICS:", "cyan"))
            print(colored(f"   Total markets: {db_stats['total_markets']}", "white"))
            print(colored(f"   Price snapshots: {db_stats['total_price_snapshots']}", "white"))
            print(colored(f"   Trade records: {db_stats['total_trade_records']}", "white"))

            if db_stats.get('markets_by_category'):
                print(colored("   Markets by category:", "white"))
                for category, count in db_stats['markets_by_category'].items():
                    print(colored(f"     {category}: {count}", "white"))

        except Exception as e:
            logger.error(f"Historical data collection failed: {e}")
            print(colored(f"❌ Historical data collection failed: {e}", "red"))

    def analyze_historical_patterns(self, category=None, output_file=None):
        """Analyze historical market patterns and generate insights."""
        print(colored("📊 ANALYZING HISTORICAL MARKET PATTERNS", "cyan", attrs=["bold"]))
        print(colored("═" * 50, "cyan"))

        try:
            from analysis import create_pattern_analyzer

            # Create analyzer
            analyzer = create_pattern_analyzer()
            print(colored("✅ Connected to historical data", "green"))

            if category:
                print(colored(f"🎯 Analyzing category: {category}", "yellow"))
            else:
                print(colored("🎯 Analyzing all categories", "yellow"))

            # Generate comprehensive report
            print(colored("\n🔬 Running pattern analysis...", "blue"))
            report = analyzer.generate_insights_report()

            # Show key insights
            patterns = analyzer.find_profitable_patterns()
            if patterns:
                print(colored(f"\n💡 Found {len(patterns)} profitable patterns", "green"))
                top_pattern = patterns[0]
                print(colored(f"🏆 Top Pattern: {top_pattern['pattern']}", "green"))
                print(colored(".1%", "green"))
            else:
                print(colored("\n⚠️  No profitable patterns identified", "yellow"))

            # Category analysis
            category_df = analyzer.analyze_category_accuracy()
            if not category_df.empty:
                print(colored(f"\n📈 Analyzed {len(category_df)} categories", "blue"))
                top_category = category_df.iloc[0]
                print(colored(f"🏅 Best Category: {top_category['category']} ({top_category['total_markets']} markets)", "blue"))
                print(colored(".1%", "blue"))

            # Calibration check
            calibration_df = analyzer.calculate_calibration_curve()
            if not calibration_df.empty:
                avg_error = calibration_df['abs_error'].mean()
                print(colored(".1%", "cyan"))

            print(colored("\n✅ Analysis complete!", "green"))
            if output_file:
                print(colored(f"📁 Report saved to: {output_file}", "white"))
            else:
                print(colored("📁 Report saved to: reports/historical_analysis_YYYYMMDD.md", "white"))

        except Exception as e:
            logger.error(f"Historical pattern analysis failed: {e}")
            print(colored(f"❌ Historical pattern analysis failed: {e}", "red"))

    def analyze_ai_accuracy(self, update_resolutions=False, plot_calibration=None, plot_trends=None, suggest_weights=False):
        """Analyze AI model accuracy and calibration."""
        print(colored("🤖 AI MODEL ACCURACY ANALYSIS", "cyan", attrs=["bold"]))
        print(colored("═" * 45, "cyan"))

        try:
            from src.analysis.ai_accuracy_tracker import create_accuracy_tracker

            # Create accuracy tracker
            tracker = create_accuracy_tracker()
            print(colored("✅ Connected to AI predictions database", "green"))

            # Update resolutions if requested
            if update_resolutions:
                print(colored("\n🔄 Updating prediction outcomes with resolved markets...", "yellow"))
                import asyncio
                updated = asyncio.run(tracker.update_resolutions())
                print(colored(f"✅ Updated {updated} prediction outcomes", "green"))

            # Get performance summary
            summary = tracker.get_performance_summary()
            print(colored(f"\n📊 PREDICTIONS SUMMARY:", "blue"))
            print(colored(f"   Total predictions: {summary['total_predictions']}", "white"))
            print(colored(f"   Resolved predictions: {summary['resolved_predictions']}", "white"))
            print(colored(".1%", "white"))

            # Model accuracy analysis
            if summary.get('model_accuracy'):
                print(colored(f"\n🎯 MODEL ACCURACY RANKING:", "blue"))
                model_acc = summary['model_accuracy']
                sorted_models = sorted(model_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)

                for i, (model, metrics) in enumerate(sorted_models, 1):
                    print(colored(f"  {i}. {model.upper()}: {metrics['accuracy']:.1%} accuracy", "green"))
                    print(colored(f"       Predictions: {metrics['predictions']}", "white"))
                    print(colored(f"       Brier Score: {metrics['brier_score']:.3f} (lower is better)", "white"))

            # Ensemble performance
            if summary.get('ensemble_performance') and 'error' not in summary['ensemble_performance']:
                ensemble = summary['ensemble_performance']
                print(colored(f"\n🤝 ENSEMBLE PERFORMANCE:", "blue"))
                print(colored(f"   Ensemble Accuracy: {ensemble['ensemble_accuracy']:.1%}", "green"))
                print(colored(f"   Ensemble Brier Score: {ensemble['ensemble_avg_brier']:.3f}", "white"))
                if 'best_individual_accuracy' in ensemble:
                    print(colored(f"   Best Individual: {ensemble['best_individual_model']} ({ensemble['best_individual_accuracy']:.1%})", "white"))
                    print(colored(f"   Ensemble Improvement: {ensemble['ensemble_improvement_pct']:+.1f}%", "cyan"))
            elif summary.get('ensemble_performance', {}).get('error'):
                print(colored(f"\n🤝 ENSEMBLE PERFORMANCE: {summary['ensemble_performance']['error']}", "yellow"))
            # Generate plots if requested
            if plot_calibration:
                print(colored(f"\n📈 Generating calibration curve plots...", "yellow"))
                tracker.plot_calibration_curves(plot_calibration)
                print(colored(f"✅ Saved calibration plots to {plot_calibration}", "green"))

            if plot_trends:
                print(colored(f"\n📈 Generating accuracy trend plots...", "yellow"))
                tracker.plot_accuracy_over_time(plot_trends)
                print(colored(f"✅ Saved trend plots to {plot_trends}", "green"))

            # Suggest weight updates if requested
            if suggest_weights:
                print(colored(f"\n⚖️ ANALYZING OPTIMAL MODEL WEIGHTS...", "yellow"))
                suggestions = tracker.suggest_weight_updates()

                if 'error' not in suggestions:
                    print(colored("Current weights:", "white"))
                    current = suggestions['current_weights']
                    for model, weight in current.items():
                        print(colored(f"  {model}: {weight}", "white"))

                    print(colored("\nSuggested weights:", "green"))
                    suggested = suggestions['suggested_weights']
                    for model, weight in suggested.items():
                        change = suggestions['changes'][model]['change']
                        change_indicator = f"({change:+.1f})" if abs(change) > 0.05 else ""
                        color = "green" if change > 0.05 else "red" if change < -0.05 else "white"
                        print(colored(f"  {model}: {weight} {change_indicator}", color))

                    print(colored(f"\n💡 {suggestions['reasoning']}", "cyan"))
                else:
                    print(colored(f"❌ Weight analysis failed: {suggestions['error']}", "red"))

            print(colored("\n✅ AI accuracy analysis complete!", "green"))

        except Exception as e:
            logger.error(f"AI accuracy analysis failed: {e}")
            print(colored(f"❌ AI accuracy analysis failed: {e}", "red"))

    def generate_calibration_report(self, plots=False, output_file=None):
        """Generate comprehensive model calibration report."""
        try:
            from src.analysis.model_calibration import model_calibration

            print(colored("📊 GENERATING MODEL CALIBRATION REPORT...", "cyan", attrs=["bold"]))

            # Generate report
            report = model_calibration.generate_calibration_report()

            # Save to file
            if output_file:
                output_path = output_file
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"reports/calibration_report_{timestamp}.md"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

            print(colored(f"💾 Report saved to {output_path}", "green"))

            # Print summary to console
            stats = model_calibration.get_stats_summary()
            print(colored("\n📈 CALIBRATION SUMMARY:", "yellow", attrs=["bold"]))
            print(colored(f"   Total Predictions: {stats['total_predictions']:,}", "white"))
            print(colored(f"   Models Tracked: {stats['models_tracked']}", "white"))
            print(colored(f"   Best Model: {stats.get('best_model', 'N/A')}", "white"))
            print(colored(f"   Best Brier Score: {stats.get('best_brier', 0.25):.3f}", "white"))
            print(colored(f"   Avg Calibration Error: {stats.get('avg_calibration_error', 0):.3f}", "white"))

            # Generate plots if requested
            if plots:
                print(colored("\n📈 Generating calibration plots...", "yellow"))
                try:
                    import matplotlib.pyplot as plt

                    models = ["claude", "gemini", "gpt", "deepseek", "perplexity"]
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    axes = axes.flatten()

                    for i, model in enumerate(models):
                        if i < len(axes):
                            buckets = model_calibration.calculate_calibration_curve(model)
                            if buckets:
                                predicted = [(b.predicted_range[0] + b.predicted_range[1]) / 2 for b in buckets]
                                actual = [b.actual_yes_rate for b in buckets]

                                ax = axes[i]
                                ax.scatter(predicted, actual, alpha=0.6)
                                ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect calibration line
                                ax.set_title(f"{model.title()} Calibration")
                                ax.set_xlabel("Predicted Probability")
                                ax.set_ylabel("Actual Frequency")
                                ax.grid(True, alpha=0.3)

                    plt.tight_layout()

                    plot_path = f"reports/calibration_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    print(colored(f"✅ Calibration plots saved to {plot_path}", "green"))

                except ImportError:
                    print(colored("⚠️ matplotlib not available - skipping plots", "yellow"))
                except Exception as e:
                    print(colored(f"❌ Plot generation failed: {e}", "red"))

            print(colored("\n✅ Calibration report complete!", "green"))

        except Exception as e:
            logger.error(f"Calibration report generation failed: {e}")
            print(colored(f"❌ Calibration report failed: {e}", "red"))

    def recalibrate_models(self, force=False):
        """Update calibration functions with new data."""
        try:
            from src.analysis.model_calibration import auto_calibrator

            print(colored("🎯 RECALIBRATING MODELS...", "cyan", attrs=["bold"]))

            # Rebuild calibration functions
            old_status = auto_calibrator.get_calibration_status()
            auto_calibrator._build_adjustment_functions()

            new_status = auto_calibrator.get_calibration_status()

            print(colored("\n🔄 RECALIBRATION RESULTS:", "yellow", attrs=["bold"]))
            for model in old_status.keys():
                old_available = old_status[model]
                new_available = new_status[model]
                status_icon = "✅" if new_available else "❌"
                change_icon = "🆕" if not old_available and new_available else "🔄" if old_available and new_available else ""
                print(colored(f"   {status_icon} {model}: {'Available' if new_available else 'Unavailable'} {change_icon}", "white"))

            # Show calibration statistics
            from src.analysis.model_calibration import model_calibration
            rankings = model_calibration.rank_models_by_accuracy()

            if not rankings.empty:
                print(colored("\n📊 CURRENT MODEL PERFORMANCE:", "yellow", attrs=["bold"]))
                for _, row in rankings.iterrows():
                    brier_color = "green" if row['brier_score'] < 0.22 else "yellow" if row['brier_score'] < 0.25 else "red"
                    print(colored(f"   {row['model']}: Brier={row['brier_score']:.3f}, Cal_Error={row['calibration_error']:.3f}", brier_color))

            print(colored("\n✅ Recalibration complete!", "green"))

        except Exception as e:
            logger.error(f"Model recalibration failed: {e}")
            print(colored(f"❌ Recalibration failed: {e}", "red"))

    def show_portfolio(self):
        """Show current portfolio summary."""
        try:
            if not hasattr(self, 'swarm') or not self.swarm:
                self.load_components()

            summary = self.swarm.get_portfolio_summary()

            if summary.get("total_positions", 0) == 0:
                print(colored("📊 No positions currently managed", "yellow"))
                return

            print(colored("📊 PORTFOLIO SUMMARY", "cyan", attrs=["bold"]))
            print(colored("=" * 50, "cyan"))

            print(colored(f"Total Positions: {summary['total_positions']}", "white"))
            print(colored(f"Total Value: ${summary['total_value']:,.2f}", "white"))
            print(colored(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.1f}%)", "green" if summary['total_pnl'] >= 0 else "red"))
            print(colored(f"Winners/Losers: {summary['winners']}/{summary['losers']}", "white"))

            if summary.get('largest_winner'):
                winner = summary['largest_winner']
                print(colored(f"Largest Winner: ${winner['pnl']:.2f} ({winner['pnl_pct']:+.1f}%)", "green"))

            if summary.get('largest_loser'):
                loser = summary['largest_loser']
                print(colored(f"Largest Loser: ${loser['pnl']:.2f} ({loser['pnl_pct']:+.1f}%)", "red"))

            print(colored("\n✅ Portfolio summary complete!", "green"))

        except Exception as e:
            print(colored(f"❌ Failed to show portfolio: {e}", "red"))

    def show_positions(self):
        """Show detailed position information."""
        try:
            if not hasattr(self, 'swarm') or not self.swarm:
                self.load_components()

            pm = self.swarm.get_position_manager()

            if not pm.positions:
                print(colored("📊 No positions currently managed", "yellow"))
                return

            print(colored("📊 MANAGED POSITIONS", "cyan", attrs=["bold"]))
            print(colored("=" * 80, "cyan"))

            for position in pm.positions.values():
                market_name = getattr(position.market, 'question', 'Unknown Market')[:60]

                status_color = "red" if position.is_underwater else "green"
                pnl_str = f"{position.unrealized_pnl:+.0f}" if not position.is_underwater else f"{position.unrealized_pnl:.0f}"

                print(colored(f"Position: {position.position_id}", "yellow", attrs=["bold"]))
                print(colored(f"  Market: {market_name}", "white"))
                print(colored(f"  Direction: {position.direction}", "white"))
                print(colored(f"  Entry: ${position.entry_price:.2f} | Current: ${position.current_price:.2f}", "white"))
                print(colored(f"  P&L: {pnl_str} ({position.unrealized_pnl_pct:+.1f}%)", status_color))
                print(colored(f"  Size: ${position.entry_size:.0f} | Value: ${position.current_value:.0f}", "white"))
                print(colored(f"  Held: {position.hours_held:.0f}h | Expiry: {position.hours_until_expiry:.0f}h", "white"))

                if position.scale_ins:
                    scale_in_total = sum(s["size"] for s in position.scale_ins)
                    print(colored(f"  Scale-ins: {len(position.scale_ins)} (${scale_in_total:.0f} total)", "blue"))

                if position.scale_outs:
                    scale_out_total = sum(s["size"] for s in position.scale_outs)
                    print(colored(f"  Scale-outs: {len(position.scale_outs)} (${scale_out_total:.0f} realized)", "blue"))

                print()

            print(colored("✅ Position details complete!", "green"))

        except Exception as e:
            print(colored(f"❌ Failed to show positions: {e}", "red"))

    async def manage_positions(self, interval_minutes: int = 60):
        """Start position management loop."""
        try:
            if not hasattr(self, 'swarm') or not self.swarm:
                self.load_components()

            print(colored(f"🔄 Starting position management (every {interval_minutes} minutes)", "cyan", attrs=["bold"]))
            print(colored("Press Ctrl+C to stop", "yellow"))

            await self.swarm.run_position_management(interval_minutes)

        except KeyboardInterrupt:
            print(colored("\n👋 Position management stopped by user", "yellow"))
        except Exception as e:
            print(colored(f"❌ Position management failed: {e}", "red"))

    async def scan_consensus_divergences(self, min_divergence=0.08, limit=10, output_file=None):
        """Scan for Polymarket divergences from external consensus."""
        try:
            from src.connectors.polymarket_client import polymarket
            from src.services.consensus_detector import consensus_detector

            print(colored("🎯 SCANNING FOR POLYMARKET CONSENSUS DIVERGENCES", "cyan", attrs=["bold"]))
            print(colored("Note: External sources are READ-ONLY data for Polymarket signals", "yellow"))
            print(colored("=" * 70, "cyan"))

            # Get active markets from Polymarket
            print(colored("📡 Fetching Polymarket markets...", "yellow"))
            pm_markets = await polymarket.get_markets(limit=limit * 2, min_volume=50000, min_liquidity=25000)  # Fetch more for filtering

            if not pm_markets:
                print(colored("❌ No Polymarket markets found", "red"))
                return

            # Filter for active markets with decent volume
            active_markets = [
                m for m in pm_markets
                if m.volume > 10000 and m.liquidity > 5000
            ][:limit]

            print(colored(f"📊 Analyzing {len(active_markets)} high-volume Polymarket markets", "white"))

            # Find divergences (this will run async)
            import asyncio
            signals = asyncio.run(consensus_detector.find_divergences(active_markets, min_divergence))

            if not signals:
                print(colored(f"\n❌ No consensus divergences found (min: {min_divergence:.1%})", "yellow"))
                return

            print(colored(f"\n📊 CONSENSUS DIVERGENCE SIGNALS (Polymarket Only)", "green", attrs=["bold"]))
            print(colored("=" * 70, "green"))

            # Prepare output data
            output_data = {
                "scan_timestamp": datetime.now().isoformat(),
                "min_divergence": min_divergence,
                "markets_scanned": len(active_markets),
                "signals_found": len(signals),
                "signals": []
            }

            for i, signal in enumerate(signals[:10], 1):  # Show top 10
                market_name = getattr(signal.market, 'question', 'Unknown Market')[:60]

                print(colored(f"\n{i}. \"{market_name}\"", "yellow", attrs=["bold"]))
                print(colored(f"   Polymarket: {signal.polymarket_price:.1%} | Consensus: {signal.external_consensus:.1%}", "white"))
                print(colored(f"   Divergence: {signal.divergence:+.1%} | Direction: {signal.direction}", "white"))
                print(colored(f"   Sources: {', '.join(signal.sources_agreeing)} ({signal.sources_count})", "white"))
                print(colored(f"   Confidence: {signal.confidence:.1%}", "white"))

                # Add to output data
                output_data["signals"].append({
                    "rank": i,
                    "market_question": market_name,
                    "polymarket_price": signal.polymarket_price,
                    "external_consensus": signal.external_consensus,
                    "divergence": signal.divergence,
                    "direction": signal.direction,
                    "sources_agreeing": signal.sources_agreeing,
                    "sources_count": signal.sources_count,
                    "confidence": signal.confidence
                })

            # Save to file if requested
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(colored(f"\n💾 Results saved to {output_file}", "green"))

            print(colored(f"\n✅ Found {len(signals)} consensus divergence signals", "green"))
            print(colored("💡 These are signals for POLYMARKET trading only", "yellow"))

        except Exception as e:
            logger.error(f"Consensus scan failed: {e}")
            print(colored(f"❌ Consensus scan failed: {e}", "red"))

    async def scan_contrarian_opportunities(self, signal_type=None, limit=10, min_score=0.5, output_file=None):
        """Scan for contrarian trading opportunities."""
        try:
            from src.connectors.polymarket_client import polymarket
            from src.strategies.contrarian_detector import contrarian_detector

            print(colored("🔄 SCANNING FOR CONTRARIAN OPPORTUNITIES", "yellow", attrs=["bold"]))
            print(colored("Finding when the crowd is wrong...", "yellow"))
            print(colored("=" * 70, "yellow"))

            # Get active markets from Polymarket
            print(colored("📡 Fetching Polymarket markets...", "yellow"))
            pm_markets = await polymarket.get_markets(limit=limit * 2, min_volume=50000, min_liquidity=25000)  # Fetch more for filtering

            if not pm_markets:
                print(colored("❌ No Polymarket markets found", "red"))
                return

            # Filter for active markets with decent volume
            active_markets = [
                m for m in pm_markets
                if m.volume > 5000 and m.liquidity > 2000
            ][:limit]

            print(colored(f"📊 Analyzing {len(active_markets)} high-volume Polymarket markets", "white"))

            # Determine signal types to scan
            signal_types = None
            if signal_type:
                signal_types = [signal_type]
                print(colored(f"🎯 Scanning for {signal_type} signals only", "cyan"))

            # Scan for contrarian opportunities
            signals = contrarian_detector.scan_all_contrarian(active_markets, signal_types)

            # Filter by minimum score
            signals = [s for s in signals if s.contrarian_score >= min_score]

            if not signals:
                print(colored(f"\n❌ No contrarian signals found (min score: {min_score:.1%})", "yellow"))
                return

            print(colored(f"\n🔄 CONTRARIAN OPPORTUNITIES FOUND", "red", attrs=["bold"]))
            print(colored("=" * 70, "red"))

            # Prepare output data
            output_data = {
                "scan_timestamp": datetime.now().isoformat(),
                "signal_type_filter": signal_type,
                "min_score": min_score,
                "markets_scanned": len(active_markets),
                "signals_found": len(signals),
                "signals": []
            }

            for i, signal in enumerate(signals[:15], 1):  # Show top 15
                market_name = getattr(signal.market, 'question', 'Unknown Market')[:60]

                print(colored(f"\n{i}. \"{market_name}\"", "yellow", attrs=["bold"]))
                print(colored(f"   Type: {signal.signal_type}", "white"))
                print(colored(f"   Direction: {signal.suggested_direction} (current price: {signal.current_price:.1%})", "white"))
                print(colored(f"   Contrarian Score: {signal.contrarian_score:.1%}", "white"))
                print(colored(f"   Confidence: {signal.confidence:.1%}", "white"))

                print(colored("   Evidence:", "green"))
                for evidence in signal.evidence[:3]:  # Top 3 evidence points
                    print(colored(f"   • {evidence}", "green"))

                print(colored("   Risks:", "red"))
                for risk in signal.risks[:2]:  # Top 2 risks
                    print(colored(f"   • {risk}", "red"))

                # Add to output data
                output_data["signals"].append({
                    "rank": i,
                    "market_question": market_name,
                    "signal_type": signal.signal_type,
                    "current_price": signal.current_price,
                    "suggested_direction": signal.suggested_direction,
                    "contrarian_score": signal.contrarian_score,
                    "confidence": signal.confidence,
                    "evidence": signal.evidence,
                    "risks": signal.risks
                })

            # Save to file if requested
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                print(colored(f"\n💾 Results saved to {output_file}", "green"))

            print(colored(f"\n✅ Found {len(signals)} contrarian opportunities", "red"))
            print(colored("⚠️  Contrarian signals are high-risk, high-reward", "yellow"))
            print(colored("💡 When everyone agrees, someone is usually wrong", "yellow"))

        except Exception as e:
            logger.error(f"Contrarian scan failed: {e}")
            print(colored(f"❌ Contrarian scan failed: {e}", "red"))

    def run_trading_swarm_v2(self, paper=True, live=False, interval=30, max_trades=10,
                            min_edge=0.08, min_confidence=0.5, config='config/categories.yaml'):
        """Run the full TradingSwarm V2 with all features."""
        try:
            # Determine trading mode
            paper_trading = paper and not live

            if live:
                print(colored("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!", "red", attrs=["bold"]))
                confirm = input("Type 'YES' to confirm live trading: ")
                if confirm.upper() != 'YES':
                    print(colored("Live trading cancelled", "yellow"))
                    return
            else:
                print(colored("📝 PAPER TRADING MODE - No real money at risk", "green"))

            # Import and initialize TradingSwarm V2
            from src.agents.trading_swarm_v2 import TradingSwarmV2

            swarm = TradingSwarmV2(
                paper_trading=paper_trading,
                config_path=config
            )

            print(colored("🚀 Starting TradingSwarm V2 with full feature integration...", "cyan", attrs=["bold"]))
            print(colored(f"⏱️  Scan interval: {interval} minutes", "white"))
            print(colored(f"📊 Max trades/day: {max_trades}", "white"))
            print(colored(f"🎯 Min edge: {min_edge*100:.0f}%", "white"))
            print(colored(f"🎯 Min confidence: {min_confidence*100:.0f}%", "white"))
            print(colored("=" * 70, "cyan"))

            # Run the trading loop
            import asyncio
            asyncio.run(swarm.run_trading_loop(
                interval_minutes=interval,
                max_trades_per_day=max_trades,
                min_edge=min_edge,
                min_confidence=min_confidence
            ))

        except KeyboardInterrupt:
            print(colored("\n🛑 TradingSwarm V2 stopped by user", "yellow"))
        except Exception as e:
            logger.error(f"TradingSwarm V2 failed: {e}")
            print(colored(f"❌ TradingSwarm V2 failed: {e}", "red"))

    async def cmd_scan(self, args):
        """Scan for trading opportunities using TradingSwarm V2."""
        try:
            from src.agents.trading_swarm_v2 import TradingSwarmV2

            print(colored("🔍 SCANNING FOR TRADING OPPORTUNITIES", "cyan", attrs=["bold"]))
            print(colored(f"Settings: edge≥{args.min_edge*100:.0f}%, conf≥{args.min_confidence*100:.0f}%, limit={args.limit}", "white"))

            swarm = TradingSwarmV2(paper_trading=True)
            signals = await swarm.find_opportunities(
                min_edge=args.min_edge,
                min_confidence=args.min_confidence,
                limit=args.limit
            )

            self.print_signals(signals)

            if hasattr(args, 'output') and args.output:
                self.save_signals_to_file(signals, args.output)

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            print(colored(f"❌ Scan failed: {e}", "red"))

    async def cmd_scan_contrarian(self, args):
        """Scan for contrarian opportunities."""
        try:
            from src.strategies.contrarian_detector import ContrarianDetector
            from src.connectors.polymarket_client import polymarket

            print(colored("🔄 SCANNING FOR CONTRARIAN OPPORTUNITIES", "yellow", attrs=["bold"]))
            print(colored("Finding when the crowd is wrong...", "yellow"))

            detector = ContrarianDetector()
            markets = await polymarket.get_markets(limit=args.limit * 2)  # Get more for filtering
            active_markets = [m for m in markets[:args.limit] if getattr(m, 'volume', 0) > 5000]

            signal_types = [args.type] if hasattr(args, 'type') and args.type else None
            signals = await detector.scan_all_contrarian(active_markets, signal_types)

            # Filter by minimum score
            signals = [s for s in signals if s.contrarian_score >= args.min_score]

            self.print_contrarian_signals(signals[:15])  # Top 15

            if hasattr(args, 'output') and args.output:
                self.save_contrarian_signals_to_file(signals, args.output)

        except Exception as e:
            logger.error(f"Contrarian scan failed: {e}")
            print(colored(f"❌ Contrarian scan failed: {e}", "red"))

    async def cmd_scan_news(self, args):
        """Scan for news-driven opportunities."""
        try:
            from src.services.news_monitor import NewsAggregator
            from src.agents.trading_swarm_v2 import TradingSwarmV2
            from src.connectors.polymarket_client import polymarket

            print(colored("📰 SCANNING FOR NEWS-DRIVEN OPPORTUNITIES", "green", attrs=["bold"]))

            # Get recent news
            news_agg = NewsAggregator([])
            categories = ["politics", "crypto", "sports", "entertainment"]
            breaking_news = await news_agg.get_breaking_news(categories)

            if hasattr(args, 'breaking_only') and args.breaking_only:
                breaking_news = [n for n in breaking_news if getattr(n, 'is_breaking', False)]

            print(colored(f"Found {len(breaking_news)} relevant news items", "white"))

            # Get markets and analyze with news context
            swarm = TradingSwarmV2(paper_trading=True)
            markets = await polymarket.get_markets(limit=args.limit)

            signals = []
            for market in markets:
                signal = await swarm.analyze_market(market, include_news=True)
                if signal and signal.news_context and signal.is_actionable:
                    signals.append(signal)

            signals.sort(key=lambda s: len(s.news_context), reverse=True)
            self.print_signals(signals[:args.limit])

            if hasattr(args, 'output') and args.output:
                self.save_signals_to_file(signals, args.output)

        except Exception as e:
            logger.error(f"News scan failed: {e}")
            print(colored(f"❌ News scan failed: {e}", "red"))

    async def cmd_analyze(self, args):
        """Analyze a specific market."""
        try:
            from src.agents.trading_swarm_v2 import TradingSwarmV2
            from src.connectors.polymarket_client import polymarket

            print(colored(f"📊 ANALYZING MARKET: {args.slug}", "blue", attrs=["bold"]))

            # Find market
            market = await polymarket.get_market_by_slug(args.slug)
            if not market:
                # Try as condition ID
                markets = await polymarket.get_markets(limit=100)
                market = next((m for m in markets if getattr(m, 'condition_id', '') == args.slug), None)

            if not market:
                print(colored(f"❌ Market not found: {args.slug}", "red"))
                return

            print(colored(f"Found: {getattr(market, 'question', 'Unknown')[:60]}...", "white"))

            # Analyze
            swarm = TradingSwarmV2(paper_trading=True)
            signal = await swarm.analyze_market(
                market,
                deep_analysis=getattr(args, 'deep', False),
                include_news=not getattr(args, 'no_news', False)
            )

            if signal:
                self.print_detailed_signal(signal)

                if hasattr(args, 'output') and args.output:
                    self.save_signal_to_file(signal, args.output)
            else:
                print(colored("❌ Analysis failed", "red"))

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            print(colored(f"❌ Analysis failed: {e}", "red"))

    async def cmd_trade(self, args):
        """Run the trading bot."""
        try:
            from src.agents.trading_swarm_v2 import TradingSwarmV2

            paper_trading = getattr(args, 'paper', True) and not getattr(args, 'live', False)

            if getattr(args, 'live', False):
                print(colored("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!", "red", attrs=["bold"]))
                confirm = input("Type 'YES' to confirm live trading: ")
                if confirm.upper() != 'YES':
                    print(colored("Live trading cancelled", "yellow"))
                    return
            else:
                print(colored("📝 PAPER TRADING MODE - No real money at risk", "green"))

            swarm = TradingSwarmV2(paper_trading=paper_trading)

            print(colored("🚀 Starting trading bot...", "cyan", attrs=["bold"]))
            print(colored(f"Settings: interval={args.interval}m, max_trades={args.max_trades}/day", "white"))

            await swarm.run_trading_loop(
                interval_minutes=args.interval,
                max_trades_per_day=args.max_trades,
                min_edge=getattr(args, 'min_edge', 0.08),
                min_confidence=getattr(args, 'min_confidence', 0.5)
            )

        except KeyboardInterrupt:
            print(colored("\n🛑 Trading bot stopped by user", "yellow"))
        except Exception as e:
            logger.error(f"Trading bot failed: {e}")
            print(colored(f"❌ Trading bot failed: {e}", "red"))

    async def cmd_manage_positions(self, args):
        """Run position management."""
        try:
            from src.strategies.position_manager import PositionManager
            from src.agents.trading_swarm import TradingSwarm  # Use original for management

            print(colored("📊 POSITION MANAGEMENT", "blue", attrs=["bold"]))
            print(colored(f"Running every {args.interval} minutes", "white"))

            # Initialize (mock swarm for now)
            swarm = TradingSwarm(paper_trading=True)
            pm = PositionManager(swarm)

            while True:
                try:
                    await pm.update_all_positions()
                    status = pm.get_portfolio_summary()

                    print(colored(f"📊 Portfolio: {status.get('total_positions', 0)} positions, ${status.get('total_value', 0):.2f} value", "white"))

                    await asyncio.sleep(args.interval * 60)

                except KeyboardInterrupt:
                    print(colored("\n🛑 Position management stopped", "yellow"))
                    break
                except Exception as e:
                    logger.error(f"Position management error: {e}")
                    await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Position management failed: {e}")
            print(colored(f"❌ Position management failed: {e}", "red"))

    def cmd_status(self):
        """Show current system status."""
        try:
            import asyncio
            from src.agents.trading_swarm_v2 import TradingSwarmV2

            async def get_status():
                swarm = TradingSwarmV2(paper_trading=True)
                status = await swarm.get_portfolio_status()

                print(colored("📊 SYSTEM STATUS", "cyan", attrs=["bold"]))
                print("=" * 50)

                # Runtime
                runtime = status.get('runtime_hours', 0)
                print(colored(f"⏱️  Runtime: {runtime:.1f} hours", "white"))

                # Trading stats
                signals = status.get('signals_generated', 0)
                trades = status.get('trades_executed', 0)
                costs = status.get('total_api_cost', 0)
                print(colored(f"📈 Signals: {signals} | Trades: {trades} | API Cost: ${costs:.2f}", "white"))

                # Components
                components = status.get('components', {})
                active = sum(1 for v in components.values() if v)
                total = len(components)
                print(colored(f"🔧 Components: {active}/{total} active", "white"))

                # Risk status
                risk = status.get('risk', {})
                if risk:
                    drawdown = risk.get('current_drawdown', 0) * 100
                    paused = risk.get('should_pause', False)
                    bankroll = risk.get('available_bankroll', 0)
                    print(colored(f"🛡️  Drawdown: {drawdown:.1f}% | Bankroll: ${bankroll:.2f} | Paused: {paused}", "white"))

                # Positions
                positions = status.get('positions', {})
                if positions:
                    total_pos = positions.get('total_positions', 0)
                    total_value = positions.get('total_value', 0)
                    print(colored(f"📊 Positions: {total_pos} open | Value: ${total_value:.2f}", "white"))

                print("=" * 50)

            asyncio.run(get_status())

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            print(colored(f"❌ Status check failed: {e}", "red"))

    def cmd_positions(self):
        """Show open positions."""
        try:
            print(colored("📊 OPEN POSITIONS", "green", attrs=["bold"]))
            print(colored("Position management not fully implemented yet", "yellow"))
            # TODO: Integrate with actual position tracking

        except Exception as e:
            logger.error(f"Positions check failed: {e}")
            print(colored(f"❌ Positions check failed: {e}", "red"))

    async def cmd_test(self):
        """Test all components."""
        print(colored("🧪 TESTING ALL COMPONENTS", "magenta", attrs=["bold"]))
        print("=" * 50)

        results = {}

        # Test AI models
        print(colored("1. Testing AI Models...", "yellow"))
        try:
            from src.models.model_factory import model_factory, test_models
            test_models()
            results["AI Models"] = "✅"
        except Exception as e:
            results["AI Models"] = f"❌ {str(e)[:50]}..."

        # Test Polymarket
        print(colored("2. Testing Polymarket...", "yellow"))
        try:
            from src.connectors.polymarket_client import polymarket
            markets = await polymarket.get_markets(limit=3)
            results["Polymarket"] = f"✅ ({len(markets)} markets)"
        except Exception as e:
            results["Polymarket"] = f"❌ {str(e)[:50]}..."

        # Test components
        components = [
            ("Risk Manager", "src.strategies.risk_manager", "RiskManager"),
            ("Entry Timing", "src.strategies.entry_timing", "EntryTimingOptimizer"),
            ("Category Specialist", "src.strategies.category_specialist", "CategorySpecialist"),
            ("Contrarian Detector", "src.strategies.contrarian_detector", "ContrarianDetector"),
            ("Position Manager", "src.strategies.position_manager", "PositionManager"),
            ("News Monitor", "src.services.news_monitor", "NewsAggregator"),
            ("Consensus Detector", "src.services.external_odds", "ConsensusDetector"),
            ("Pattern Analyzer", "src.analysis.pattern_analyzer", "PatternAnalyzer"),
            ("Model Calibration", "src.analysis.model_calibration", "ModelCalibration"),
            ("TradingSwarm V2", "src.agents.trading_swarm_v2", "TradingSwarmV2"),
        ]

        print(colored("3. Testing Components...", "yellow"))
        for name, module, class_name in components:
            try:
                mod = __import__(module, fromlist=[class_name])
                cls = getattr(mod, class_name)
                # Simple instantiation test
                if class_name == "NewsAggregator":
                    instance = cls([])  # Empty sources
                elif class_name == "PositionManager":
                    continue  # Skip complex initialization
                elif class_name == "TradingSwarmV2":
                    continue  # Skip complex initialization
                else:
                    instance = cls()
                results[name] = "✅"
            except Exception as e:
                results[name] = f"❌ {str(e)[:50]}..."

        # Print results
        print("\n" + "="*50)
        print(colored("TEST RESULTS", "magenta", attrs=["bold"]))
        print("="*50)
        for component, status in results.items():
            color = "green" if status.startswith("✅") else "red"
            print(colored(f"{status} {component}", color))

        passed = sum(1 for s in results.values() if s.startswith("✅"))
        total = len(results)
        print(f"\n{passed}/{total} components passed")

        if passed == total:
            print(colored("🎉 All components working!", "green", attrs=["bold"]))
        elif passed >= total * 0.8:
            print(colored("✅ Most components working", "yellow"))
        else:
            print(colored("⚠️  Some components need attention", "red"))

    def cmd_collect_history(self, args):
        """Collect historical data."""
        try:
            from src.analysis.historical_collector import HistoricalDataCollector

            print(colored("📚 COLLECTING HISTORICAL DATA", "blue", attrs=["bold"]))
            print(colored(f"Collecting {getattr(args, 'days', 365)} days of history", "white"))

            collector = HistoricalDataCollector()

            if getattr(args, 'markets_only', False):
                print("Collecting market data only...")
                # Implementation would go here
                print(colored("✅ Market data collection not implemented yet", "yellow"))
            elif getattr(args, 'prices_only', False):
                print("Collecting price data only...")
                # Implementation would go here
                print(colored("✅ Price data collection not implemented yet", "yellow"))
            else:
                print("Collecting all historical data...")
                # Implementation would go here
                print(colored("✅ Full historical collection not implemented yet", "yellow"))

        except Exception as e:
            logger.error(f"History collection failed: {e}")
            print(colored(f"❌ History collection failed: {e}", "red"))

    def cmd_analyze_history(self, args):
        """Analyze historical patterns."""
        try:
            from src.analysis.pattern_analyzer import PatternAnalyzer

            print(colored("📈 ANALYZING HISTORICAL PATTERNS", "purple", attrs=["bold"]))

            analyzer = PatternAnalyzer()

            if hasattr(args, 'category') and args.category:
                print(colored(f"Analyzing {args.category} category", "white"))
            else:
                print("Analyzing all categories...")

            # Generate insights report
            report = analyzer.generate_insights_report()
            print(report[:500] + "..." if len(report) > 500 else report)

            if hasattr(args, 'output') and args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(colored(f"💾 Report saved to {args.output}", "green"))

        except Exception as e:
            logger.error(f"History analysis failed: {e}")
            print(colored(f"❌ History analysis failed: {e}", "red"))

    def cmd_calibration_report(self, args):
        """Show model calibration report."""
        try:
            from src.analysis.model_calibration import ModelCalibration

            print(colored("🎯 MODEL CALIBRATION REPORT", "orange", attrs=["bold"]))

            calibrator = ModelCalibration()

            # Generate report
            report = calibrator.generate_calibration_report()
            print(report)

            if hasattr(args, 'plots') and args.plots:
                print(colored("📊 Generating calibration plots...", "yellow"))
                # Plot generation would go here
                print(colored("✅ Plot generation not implemented yet", "yellow"))

            if hasattr(args, 'output') and args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(colored(f"💾 Report saved to {args.output}", "green"))

        except Exception as e:
            logger.error(f"Calibration report failed: {e}")
            print(colored(f"❌ Calibration report failed: {e}", "red"))

    def cmd_performance_report(self, args):
        """Show trading performance report."""
        try:
            print(colored("📊 TRADING PERFORMANCE REPORT", "cyan", attrs=["bold"]))
            print(colored(f"Analyzing last {getattr(args, 'days', 30)} days", "white"))

            # Performance analysis would go here
            print(colored("✅ Performance analysis not implemented yet", "yellow"))
            print("Would show: P&L, win rate, Sharpe ratio, max drawdown, etc.")

        except Exception as e:
            logger.error(f"Performance report failed: {e}")
            print(colored(f"❌ Performance report failed: {e}", "red"))

    # Utility methods for printing and saving
    def print_signals(self, signals):
        """Pretty print trading signals."""
        if not signals:
            print(colored("❌ No signals found", "yellow"))
            return

        print(f"\n{'='*80}")
        print(colored(f"📊 TRADING SIGNALS ({len(signals)} found)", "green", attrs=["bold"]))
        print(f"{'='*80}\n")

        for i, signal in enumerate(signals, 1):
            market_name = getattr(signal.market, 'question', 'Unknown Market')[:55]
            market_volume = getattr(signal.market, 'volume', 0)
            market_liquidity = getattr(signal.market, 'liquidity', 0)
            quality_color = {
                "A+": "green", "A": "green", "A-": "yellow",
                "B+": "yellow", "B": "yellow", "B-": "red",
                "C": "red", "D": "red"
            }.get(signal.signal_quality, "white")

            print(colored(f"{i}. {market_name}...", "white"))
            print(colored(f"   Volume: ${market_volume:,.0f} | Liquidity: ${market_liquidity:,.0f}", "cyan"))
            print(colored(f"   Direction: {signal.direction} | Edge: {signal.edge*100:.1f}% | Confidence: {signal.confidence*100:.0f}%", "white"))
            print(colored(f"   Size: ${signal.recommended_size:.2f} | Quality: {signal.signal_quality}", quality_color))
            print(colored(f"   Expected Value: {signal.expected_value*100:.1f}%", "white"))

            if signal.external_consensus:
                ext_sources = signal.external_consensus.get('sources_count', 0)
                if ext_sources > 0:
                    print(colored(f"   External Sources: {ext_sources}", "cyan"))

            if signal.contrarian_signals:
                contra_count = len(signal.contrarian_signals)
                print(colored(f"   Contrarian Signals: {contra_count}", "yellow"))

            print()

    def print_contrarian_signals(self, signals):
        """Pretty print contrarian signals."""
        if not signals:
            print(colored("❌ No contrarian signals found", "yellow"))
            return

        print(f"\n{'='*80}")
        print(colored(f"🔄 CONTRARIAN SIGNALS ({len(signals)} found)", "red", attrs=["bold"]))
        print(f"{'='*80}\n")

        for i, signal in enumerate(signals, 1):
            market_name = getattr(signal.market, 'question', 'Unknown Market')[:50]

            print(colored(f"{i}. {market_name}...", "white"))
            print(colored(f"   Type: {signal.signal_type}", "yellow"))
            print(colored(f"   Direction: {signal.suggested_direction} | Score: {signal.contrarian_score:.1%} | Confidence: {signal.confidence:.1%}", "white"))

            if signal.evidence:
                evidence = signal.evidence[0][:60] + "..." if len(signal.evidence[0]) > 60 else signal.evidence[0]
                print(colored(f"   Evidence: {evidence}", "green"))

            if signal.risks:
                risk = signal.risks[0][:60] + "..." if len(signal.risks[0]) > 60 else signal.risks[0]
                print(colored(f"   Risk: {risk}", "red"))

            print()

    def print_detailed_signal(self, signal):
        """Print detailed signal analysis."""
        if not signal:
            print(colored("❌ No signal to display", "red"))
            return

        market_name = getattr(signal.market, 'question', 'Unknown Market')

        print(f"\n{'='*80}")
        print(colored(f"📊 DETAILED ANALYSIS: {market_name[:50]}...", "blue", attrs=["bold"]))
        print(f"{'='*80}")

        # Market info
        market_volume = getattr(signal.market, 'volume', 0)
        market_liquidity = getattr(signal.market, 'liquidity', 0)
        max_safe_size = min(signal.recommended_size, market_liquidity * 0.02)
        est_slippage = (signal.recommended_size / market_liquidity) * 50 if market_liquidity > 0 else 0

        print(colored("🏪 MARKET INFO", "blue", attrs=["bold"]))
        print(f"Volume (24h): ${market_volume:,.0f}")
        print(f"Liquidity: ${market_liquidity:,.0f}")
        print(f"Max Safe Order: ${max_safe_size:.2f} (2% of liquidity)")
        print(f"Est. Slippage: {est_slippage:.2f}%")

        # Basic info
        print(colored("\n🎯 SIGNAL SUMMARY", "cyan", attrs=["bold"]))
        print(f"Direction: {signal.direction}")
        print(f"Edge: {signal.edge*100:.1f}%")
        print(f"Confidence: {signal.confidence*100:.0f}%")
        print(f"Quality: {signal.signal_quality}")
        print(f"Actionable: {'✅ YES' if signal.is_actionable else '❌ NO'}")
        if signal.skip_reason:
            print(f"Skip Reason: {signal.skip_reason}")

        # Trading details
        print(colored("\n💰 TRADING DETAILS", "green", attrs=["bold"]))
        print(f"Recommended Size: ${signal.recommended_size:.2f}")
        print(f"Max Safe Size: ${max_safe_size:.2f}")
        print(f"Expected Value: {signal.expected_value*100:.1f}%")
        print(f"Kelly Fraction: {signal.kelly_fraction:.2%}")

        # Probabilities
        print(colored("\n🎲 PROBABILITIES", "magenta", attrs=["bold"]))
        print(f"AI Probability: {signal.ai_probability*100:.1f}%")
        print(f"Calibrated: {signal.calibrated_probability*100:.1f}%")
        print(f"Final: {signal.final_probability*100:.1f}%")
        print(f"Market: {signal.market_probability*100:.1f}%")

        # Enhancements
        if signal.external_consensus:
            print(colored("\n📊 EXTERNAL CONSENSUS", "orange", attrs=["bold"]))
            sources = signal.external_consensus.get('sources_count', 0)
            weighted = signal.external_consensus.get('weighted_average', 0)
            print(f"Sources: {sources}")
            print(f"Weighted Average: {weighted*100:.1f}%")

        if signal.contrarian_signals:
            print(colored("\n⚠️ CONTRARIAN SIGNALS", "red", attrs=["bold"]))
            for contra in signal.contrarian_signals[:3]:
                print(f"• {contra.get('signal_type', 'unknown')}: {contra.get('contrarian_score', 0):.1%}")

        if signal.news_context:
            print(colored("\n📰 NEWS CONTEXT", "green", attrs=["bold"]))
            for news in signal.news_context[:3]:
                title = news.get('headline', 'Unknown')[:50]
                print(f"• {title}...")

        if signal.category_config:
            print(colored("\n🏷️ CATEGORY CONFIG", "purple", attrs=["bold"]))
            enabled = signal.category_config.get('enabled', True)
            min_edge = signal.category_config.get('min_edge', 0) * 100
            print(f"Enabled: {enabled}")
            print(f"Min Edge: {min_edge:.0f}%")

        print(f"\n{'='*80}")

    def save_signals_to_file(self, signals, filename):
        """Save signals to JSON file."""
        try:
            import json
            data = {
                "timestamp": datetime.now().isoformat(),
                "signals_count": len(signals),
                "signals": [
                    {
                        "market": getattr(s.market, 'question', 'Unknown'),
                        "direction": s.direction,
                        "edge": s.edge,
                        "confidence": s.confidence,
                        "quality": s.signal_quality,
                        "size": s.recommended_size,
                        "expected_value": s.expected_value
                    } for s in signals
                ]
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            print(colored(f"💾 Signals saved to {filename}", "green"))

        except Exception as e:
            print(colored(f"❌ Failed to save signals: {e}", "red"))

    def save_contrarian_signals_to_file(self, signals, filename):
        """Save contrarian signals to JSON file."""
        try:
            import json
            data = {
                "timestamp": datetime.now().isoformat(),
                "signals_count": len(signals),
                "signals": [
                    {
                        "market": getattr(s.market, 'question', 'Unknown'),
                        "type": s.signal_type,
                        "direction": s.suggested_direction,
                        "score": s.contrarian_score,
                        "confidence": s.confidence,
                        "evidence": s.evidence[:2],  # First 2 evidence points
                        "risks": s.risks[:2]  # First 2 risks
                    } for s in signals
                ]
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            print(colored(f"💾 Contrarian signals saved to {filename}", "green"))

        except Exception as e:
            print(colored(f"❌ Failed to save contrarian signals: {e}", "red"))

    def save_signal_to_file(self, signal, filename):
        """Save detailed signal to JSON file."""
        try:
            import json
            data = {
                "timestamp": datetime.now().isoformat(),
                "market": getattr(signal.market, 'question', 'Unknown'),
                "signal": {
                    "direction": signal.direction,
                    "edge": signal.edge,
                    "confidence": signal.confidence,
                    "quality": signal.signal_quality,
                    "actionable": signal.is_actionable,
                    "size": signal.recommended_size,
                    "probabilities": {
                        "ai": signal.ai_probability,
                        "calibrated": signal.calibrated_probability,
                        "final": signal.final_probability,
                        "market": signal.market_probability
                    },
                    "enhancements": {
                        "external_sources": signal.external_consensus.get('sources_count', 0) if signal.external_consensus else 0,
                        "contrarian_signals": len(signal.contrarian_signals),
                        "news_items": len(signal.news_context)
                    }
                }
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            print(colored(f"💾 Detailed signal saved to {filename}", "green"))

        except Exception as e:
            print(colored(f"❌ Failed to save signal: {e}", "red"))


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="🌙 Polymarket AI Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py markets --limit 10 --min-volume 50000
  python main.py analyze will-bitcoin-hit-100k --verbose
  python main.py scan --min-edge 0.05 --top 10
  python main.py trade --paper
  python main.py status
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Markets command
    markets_parser = subparsers.add_parser('markets', help='List top markets')
    markets_parser.add_argument('--limit', type=int, default=20, help='Number of markets to show')
    markets_parser.add_argument('--category', help='Filter by category')
    markets_parser.add_argument('--min-volume', type=float, default=0, help='Minimum volume filter')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze specific market')
    analyze_parser.add_argument('market_slug', help='Market slug to analyze')
    analyze_parser.add_argument('--verbose', action='store_true', help='Show detailed analysis')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for opportunities')
    scan_parser.add_argument('--min-edge', type=float, default=0.10, help='Minimum edge required')
    scan_parser.add_argument('--min-volume', type=float, default=10000, help='Minimum volume')
    scan_parser.add_argument('--top', type=int, default=5, help='Show top N opportunities')

    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Start trading bot')
    trade_parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    trade_parser.add_argument('--live', action='store_true', help='Live trading mode (requires confirmation)')
    trade_parser.add_argument('--interval', type=int, default=60, help='Scan interval in minutes')

    # Status command
    subparsers.add_parser('status', help='Show current bot status')

    # Calibration commands
    calibration_parser = subparsers.add_parser('calibration-report', help='Generate model calibration report')
    calibration_parser.add_argument('--plots', action='store_true', help='Generate calibration plots')
    calibration_parser.add_argument('--output', type=str, help='Output file path')

    recalibrate_parser = subparsers.add_parser('recalibrate', help='Update calibration functions with new data')
    recalibrate_parser.add_argument('--force', action='store_true', help='Force rebuild even with insufficient data')

    # Portfolio management commands
    portfolio_parser = subparsers.add_parser('portfolio', help='Show current portfolio status')
    positions_parser = subparsers.add_parser('positions', help='Show detailed position information')
    manage_parser = subparsers.add_parser('manage-positions', help='Start position management loop')
    manage_parser.add_argument('--interval', type=int, default=60, help='Management interval in minutes')

    # Consensus scanning command
    consensus_parser = subparsers.add_parser('scan-consensus', help='Find Polymarket divergences from external consensus')
    consensus_parser.add_argument('--min-divergence', type=float, default=0.08, help='Minimum divergence threshold (0.08 = 8%)')
    consensus_parser.add_argument('--limit', type=int, default=10, help='Maximum markets to scan')
    consensus_parser.add_argument('--output', type=str, help='Output file path')

    # Contrarian scanning command
    contrarian_parser = subparsers.add_parser('scan-contrarian', help='Find contrarian trading opportunities')
    contrarian_parser.add_argument('--type', type=str, help='Signal type to scan for (sentiment_divergence, sharp_vs_public, overreaction, consensus_trap)')
    contrarian_parser.add_argument('--limit', type=int, default=10, help='Maximum markets to scan')
    contrarian_parser.add_argument('--min-score', type=float, default=0.5, help='Minimum contrarian score (0-1)')
    contrarian_parser.add_argument('--output', type=str, help='Output file path')


    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest trading strategy')
    backtest_parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')

    # Collect history command
    collect_parser = subparsers.add_parser('collect-history', help='Collect historical market data')
    collect_parser.add_argument('--days', type=int, default=365, help='Number of days back to collect (365 for full backfill, 1 for daily update)')

    # Analyze history command
    analyze_parser = subparsers.add_parser('analyze-history', help='Analyze historical market patterns')
    analyze_parser.add_argument('--category', help='Analyze specific category only')
    analyze_parser.add_argument('--output', help='Output file path (default: reports/historical_analysis_YYYYMMDD.md)')

    # AI accuracy command
    accuracy_parser = subparsers.add_parser('ai-accuracy', help='Analyze AI model accuracy and calibration')
    accuracy_parser.add_argument('--update-resolutions', action='store_true', help='Update prediction outcomes with resolved markets')
    accuracy_parser.add_argument('--plot-calibration', help='Generate calibration curve plots (requires matplotlib)')
    accuracy_parser.add_argument('--plot-trends', help='Generate accuracy trend plots (requires matplotlib)')
    accuracy_parser.add_argument('--suggest-weights', action='store_true', help='Suggest optimal model weights based on performance')

    # Pre-flight check command
    preflight_parser = subparsers.add_parser('preflight', help='Run comprehensive pre-flight checks before paper trading')
    preflight_parser.add_argument('--skip-ai', action='store_true', help='Skip AI model connectivity tests (faster)')
    preflight_parser.add_argument('--quick', action='store_true', help='Run only critical checks (faster)')

    return parser


def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    logger.info("Loaded environment variables")

    # Show banner
    print(colored(BANNER, "cyan"))

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = TradingBotCLI()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print(colored("\n\n👋 Interrupted by user", "yellow"))
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Execute command
        if args.command == 'markets':
            cli.show_markets(
                limit=args.limit,
                category=args.category,
                min_volume=args.min_volume
            )
        elif args.command == 'analyze':
            cli.analyze_market(args.market_slug, verbose=args.verbose)
        elif args.command == 'scan':
            cli.scan_opportunities(
                min_edge=args.min_edge,
                min_volume=args.min_volume,
                top=args.top
            )
        elif args.command == 'trade':
            cli.start_trading(
                paper=args.paper,
                live=args.live,
                interval=args.interval
            )
        elif args.command == 'calibration-report':
            cli.generate_calibration_report(
                plots=args.plots,
                output_file=args.output
            )
        elif args.command == 'recalibrate':
            cli.recalibrate_models(force=args.force)
        elif args.command == 'portfolio':
            cli.show_portfolio()
        elif args.command == 'positions':
            cli.show_positions()
        elif args.command == 'manage-positions':
            import asyncio
            asyncio.run(cli.manage_positions(args.interval))
        elif args.command == 'scan-consensus':
            cli.scan_consensus_divergences(
                min_divergence=args.min_divergence,
                limit=args.limit,
                output_file=args.output
            )
        elif args.command == 'swarm-v2':
            cli.run_trading_swarm_v2(
                paper=args.paper,
                live=args.live,
                interval=args.interval,
                max_trades=args.max_trades,
                min_edge=args.min_edge,
                min_confidence=args.min_confidence,
                config=args.config
            )
        elif args.command == 'scan':
            import asyncio
            asyncio.run(cli.cmd_scan(args))
        elif args.command == 'scan-contrarian':
            import asyncio
            asyncio.run(cli.cmd_scan_contrarian(args))
        elif args.command == 'scan-consensus':
            cli.scan_consensus_divergences(
                min_divergence=args.min_divergence,
                limit=args.limit,
                output_file=getattr(args, 'output', None)
            )
        elif args.command == 'scan-news':
            import asyncio
            asyncio.run(cli.cmd_scan_news(args))
        elif args.command == 'analyze':
            import asyncio
            asyncio.run(cli.cmd_analyze(args))
        elif args.command == 'collect-history':
            cli.cmd_collect_history(args)
        elif args.command == 'analyze-history':
            cli.cmd_analyze_history(args)
        elif args.command == 'calibration-report':
            cli.cmd_calibration_report(args)
        elif args.command == 'performance-report':
            cli.cmd_performance_report(args)
        elif args.command == 'trade':
            import asyncio
            asyncio.run(cli.cmd_trade(args))
        elif args.command == 'manage-positions':
            import asyncio
            asyncio.run(cli.cmd_manage_positions(args))
        elif args.command == 'status':
            cli.cmd_status()
        elif args.command == 'positions':
            cli.cmd_positions()
        elif args.command == 'test':
            import asyncio
            asyncio.run(cli.cmd_test())
        elif args.command == 'status':
            cli.show_status()
        elif args.command == 'backtest':
            cli.backtest_strategy(days=args.days)
        elif args.command == 'collect-history':
            cli.collect_historical_data(days=args.days)
        elif args.command == 'analyze-history':
            cli.analyze_historical_patterns(category=args.category, output_file=args.output)
        elif args.command == 'ai-accuracy':
            cli.analyze_ai_accuracy(
                update_resolutions=args.update_resolutions,
                plot_calibration=args.plot_calibration,
                plot_trends=args.plot_trends,
                suggest_weights=args.suggest_weights
            )
        elif args.command == 'preflight':
            import asyncio
            # Import and run the pre-flight check
            sys.path.insert(0, 'scripts')
            try:
                from pre_flight_check import main as preflight_main
                exit_code = asyncio.run(preflight_main())
                sys.exit(exit_code)
            except ImportError as e:
                print(colored(f"❌ Pre-flight check script not found: {e}", "red"))
                print(colored("Run: python scripts/pre_flight_check.py", "yellow"))
                sys.exit(1)

    except Exception as e:
        logger.error(f"CLI command failed: {e}")
        print(colored(f"❌ Command failed: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
