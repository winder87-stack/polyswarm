"""
TradingSwarm V2 - Full Integration of All Features

Complete trading system integrating AI swarm analysis, news monitoring,
external consensus, timing optimization, category specialization,
position management, contrarian detection, risk management,
historical analysis, and model calibration.
"""

import asyncio
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from src.connectors import Market

from loguru import logger

# Core components (with graceful fallbacks)
try:
    from src.agents.swarm_agent import SwarmAgent
except ImportError:
    logger.warning("SwarmAgent not available")
    SwarmAgent = None

try:
    from src.connectors.polymarket_client import polymarket, Market
except ImportError:
    logger.warning("Polymarket client not available")
    polymarket = None

try:
    from src.models.model_factory import model_factory
except ImportError:
    logger.warning("Model factory not available")
    model_factory = None

# Strategy components
try:
    from src.strategies.risk_manager import RiskManager, RiskLimits, DrawdownProtector, PerformanceTracker
except ImportError:
    logger.warning("Risk manager not available")
    RiskManager = DrawdownProtector = PerformanceTracker = None

try:
    from src.strategies.entry_timing import EntryTimingOptimizer
except ImportError:
    logger.warning("Entry timing not available")
    EntryTimingOptimizer = None

try:
    from src.strategies.position_manager import PositionManager
except ImportError:
    logger.warning("Position manager not available")
    PositionManager = None

try:
    from src.strategies.category_specialist import CategorySpecialist
except ImportError:
    logger.warning("Category specialist not available")
    CategorySpecialist = None

try:
    from src.strategies.contrarian_detector import ContrarianDetector
except ImportError:
    logger.warning("Contrarian detector not available")
    ContrarianDetector = None

# Service components
try:
    from src.services.news_monitor import NewsAggregator
except ImportError:
    logger.warning("News monitor not available")
    NewsAggregator = None

try:
    from src.services.external_odds import ConsensusDetector
except ImportError:
    logger.warning("Consensus detector not available")
    ConsensusDetector = None

# Analysis components
try:
    from src.analysis.pattern_analyzer import PatternAnalyzer
except ImportError:
    logger.warning("Pattern analyzer not available")
    PatternAnalyzer = None

try:
    from src.analysis.model_calibration import AutoCalibrator, ModelCalibration
except ImportError:
    logger.warning("Model calibration not available")
    AutoCalibrator = ModelCalibration = None

try:
    from src.analysis.ai_accuracy_tracker import AIAccuracyTracker
except ImportError:
    logger.warning("AI accuracy tracker not available")
    AIAccuracyTracker = None


@dataclass
class EnhancedTradingSignal:
    """Trading signal with all enhancement data"""
    market: Any  # Market object
    direction: str  # "YES" or "NO"

    # Core probability analysis
    ai_probability: float  # Raw AI swarm output
    calibrated_probability: float  # After model calibration
    final_probability: float  # Final combined probability

    # Market comparison
    market_probability: float  # Current market price
    edge: float  # final_probability - market_probability

    # Confidence & sizing
    confidence: float  # 0-1 overall confidence
    expected_value: float  # edge * confidence
    kelly_fraction: float  # Optimal Kelly bet size
    recommended_size: float  # Recommended USD position size

    # AI details
    model_votes: Dict[str, float]  # Individual model probabilities
    reasoning: str  # AI swarm reasoning

    # Enhancement data
    news_context: List[Dict] = field(default_factory=list)  # Recent news
    external_consensus: Optional[Dict] = None  # External odds data
    timing_signal: Optional[Dict] = None  # Entry timing analysis
    category_config: Optional[Dict] = None  # Category-specific settings
    contrarian_signals: List[Dict] = field(default_factory=list)  # Contrarian signals
    historical_edge_adjustment: float = 0.0  # Historical pattern adjustment

    # Status flags
    is_actionable: bool = False  # Ready to trade
    is_breaking_news: bool = False  # Breaking news trade
    skip_reason: Optional[str] = None  # Why not actionable

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    signal_quality: str = field(init=False)  # A+, A, B+, etc.

    def __post_init__(self) -> None:
        """Calculate signal quality after initialization"""
        self.signal_quality = self._calculate_quality()

    def _calculate_quality(self) -> str:
        """Rate signal quality based on multiple factors"""
        score = 0

        # Edge quality
        if self.edge >= 0.12:
            score += 2
        elif self.edge >= 0.08:
            score += 1

        # Confidence quality
        if self.confidence >= 0.7:
            score += 2
        elif self.confidence >= 0.5:
            score += 1

        # External consensus agreement
        if self.external_consensus and self.external_consensus.get("sources_count", 0) >= 2:
            divergence = abs(self.external_consensus.get("divergence", 0))
            if divergence >= 0.05:
                score += 1

        # Contrarian signals
        if self.contrarian_signals:
            score += 1

        # Timing quality
        if self.timing_signal and self.timing_signal.get("overall_score", 0) >= 0.7:
            score += 1

        # News context
        if self.news_context:
            score += 1

        # Quality tiers
        if score >= 7:
            return "A+"
        elif score >= 6:
            return "A"
        elif score >= 5:
            return "A-"
        elif score >= 4:
            return "B+"
        elif score >= 3:
            return "B"
        elif score >= 2:
            return "B-"
        elif score >= 1:
            return "C"
        else:
            return "D"


class TradingSwarmV2:
    """
    Full-featured trading swarm with complete feature integration.

    Orchestrates all components:
    - AI Swarm analysis with calibration
    - News monitoring and breaking news detection
    - External consensus and arbitrage detection
    - Entry timing optimization
    - Category specialization
    - Position management and scaling
    - Contrarian signal detection
    - Risk management with Kelly sizing
    - Historical pattern analysis
    - Model calibration and accuracy tracking
    """

    def __init__(
        self,
        paper_trading: bool = True,
        config_path: str = "config/categories.yaml"
    ):
        """Initialize the complete trading system."""
        self.paper_trading = paper_trading
        self.config_path = config_path

        # Core AI components
        self.swarm = SwarmAgent() if SwarmAgent else None
        self.calibrator = AutoCalibrator(ModelCalibration()) if AutoCalibrator and ModelCalibration else None
        self.accuracy_tracker = AIAccuracyTracker() if AIAccuracyTracker else None

        # Risk management
        self.risk_manager = RiskManager() if RiskManager else None
        self.drawdown = DrawdownProtector() if DrawdownProtector else None
        self.performance = PerformanceTracker() if PerformanceTracker else None

        # Enhancement components
        self.timing = EntryTimingOptimizer() if EntryTimingOptimizer else None
        self.category = CategorySpecialist(config_path) if CategorySpecialist else None
        self.contrarian = ContrarianDetector() if ContrarianDetector else None
        self.position_manager = PositionManager(self) if PositionManager else None

        # Data components
        self.news = NewsAggregator() if NewsAggregator else None  # Will create default sources
        self.consensus = ConsensusDetector() if ConsensusDetector else None
        self.patterns = PatternAnalyzer() if PatternAnalyzer else None

        # Trading state
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_api_cost = 0.0
        self.start_time = datetime.now()

        # Log initialization status
        components = []
        if self.swarm: components.append("AI Swarm")
        if self.calibrator: components.append("Calibration")
        if self.risk_manager: components.append("Risk Management")
        if self.timing: components.append("Entry Timing")
        if self.category: components.append("Category Specialist")
        if self.contrarian: components.append("Contrarian Detector")
        if self.position_manager: components.append("Position Manager")
        if self.news: components.append("News Monitor")
        if self.consensus: components.append("Consensus Detector")
        if self.patterns: components.append("Pattern Analysis")
        if self.accuracy_tracker: components.append("Accuracy Tracking")

        logger.info(f"🚀 TradingSwarm V2 initialized (paper={paper_trading})")
        logger.info(f"📦 Active components: {', '.join(components)}")

    async def analyze_market(
        self,
        market: Any,
        deep_analysis: bool = False,
        include_news: bool = True,
        include_external: bool = True
    ) -> Optional[EnhancedTradingSignal]:
        """
        Complete market analysis with all enhancements.

        Pipeline:
        1. Category filtering and configuration
        2. Gather context (news, external consensus, contrarian signals)
        3. AI swarm analysis with enhanced prompting
        4. Model calibration and historical adjustment
        5. Risk calculation and position sizing
        6. Entry timing analysis
        7. Final actionability check
        """
        try:
            logger.info(f"📊 Analyzing: {getattr(market, 'question', 'Unknown')[:60]}...")

            # 1. Category check - should we trade this category?
            if self.category:
                should_trade, reason = self.category.should_trade_category(market)
                if not should_trade:
                    logger.debug(f"⏭️ Skipping: {reason}")
                    return EnhancedTradingSignal(
                        market=market,
                        direction="",
                        ai_probability=0,
                        calibrated_probability=0,
                        final_probability=0,
                        market_probability=getattr(market, 'yes_price', 0.5),
                        edge=0,
                        confidence=0,
                        expected_value=0,
                        kelly_fraction=0,
                        recommended_size=0,
                        model_votes={},
                        reasoning="",
                        is_actionable=False,
                        skip_reason=reason
                    )

            # 2. Get category-specific settings
            category_config = None
            model_weights = {}
            thresholds = {"min_edge": 0.08, "min_confidence": 0.5}  # Defaults

            if self.category:
                category_config = self.category.get_config(market)
                model_weights = self.category.get_adjusted_weights(market)
                thresholds = self.category.get_adjusted_thresholds(market)

            # 3. Gather enhancement context (run in parallel)
            context_tasks = []

            # News context
            if include_news and self.news:
                context_tasks.append(self.news.get_recent_news(market))
            else:
                context_tasks.append(asyncio.sleep(0))

            # External consensus
            if include_external and self.consensus:
                context_tasks.append(self.consensus.get_external_consensus(market))
            else:
                context_tasks.append(asyncio.sleep(0))

            # Contrarian signals
            if self.contrarian:
                context_tasks.append(self.contrarian.scan_market(market))
            else:
                context_tasks.append(asyncio.sleep(0))

            # Execute context gathering
            context_results = await asyncio.gather(*context_tasks, return_exceptions=True)

            # Parse results with error handling
            if isinstance(context_results[0], Exception):
                logger.debug(f"News context fetch failed: {context_results[0]}")
            if isinstance(context_results[1], Exception):
                logger.debug(f"External consensus fetch failed: {context_results[1]}")
            if isinstance(context_results[2], Exception):
                logger.debug(f"Contrarian scan failed: {context_results[2]}")

            news_context = context_results[0] if not isinstance(context_results[0], Exception) else []
            external_consensus = context_results[1] if not isinstance(context_results[1], Exception) else None
            contrarian_signals = context_results[2] if not isinstance(context_results[2], Exception) else []

            # 4. Build enhanced analysis prompt
            prompt = self._build_enhanced_prompt(
                market=market,
                news=news_context,
                external=external_consensus,
                contrarian=contrarian_signals,
                category_config=category_config
            )

            # 5. AI swarm analysis
            if not self.swarm:
                logger.error("AI Swarm not available")
                return None

            if deep_analysis:
                response = await self.swarm.deep_analysis(market, prompt)
            else:
                response = await self.swarm.query(prompt, model_weights=model_weights or None)

            # Track API costs
            self.total_api_cost += response.get("total_cost", 0)

            # 6. Extract and enhance probabilities
            ai_probability = response.get("weighted_probability", 0.5)
            model_votes = response.get("model_probabilities", {})

            # Apply model calibration
            calibrated_probability = ai_probability
            if self.calibrator and model_votes:
                calibrated_probability = self.calibrator.calibrate_ensemble(model_votes)

            # Apply historical pattern adjustment
            historical_adjustment = 0.0
            if self.patterns:
                try:
                    historical_adjustment = self.patterns.get_historical_edge_adjustment(
                        market, calibrated_probability
                    )
                except Exception as e:
                    logger.debug(f"Historical adjustment failed: {e}")

            # Combine with external consensus
            final_probability = calibrated_probability
            if external_consensus and external_consensus.get("sources_count", 0) >= 2:
                external_weight = external_consensus.get("weighted_average", calibrated_probability)
                final_probability = 0.7 * calibrated_probability + 0.3 * external_weight

                # Boost confidence if AI agrees with external
                agreement = abs(calibrated_probability - external_weight)
                if agreement < 0.05:
                    response["confidence"] = min(response.get("confidence", 0.5) * 1.2, 1.0)

            # 7. Calculate trading metrics
            market_price = getattr(market, 'yes_price', 0.5)
            edge = final_probability - market_price
            direction = "YES" if edge > 0 else "NO"

            # Adjust for NO direction
            if direction == "NO":
                edge = market_price - (1 - final_probability)
                market_price = getattr(market, 'no_price', 0.5)  # Switch to NO price

            edge = abs(edge)
            confidence = response.get("confidence", 0.5)

            # Expected value calculation
            expected_value = edge * confidence

            # Kelly sizing with uncertainty
            kelly_fraction = 0.0
            if self.risk_manager:
                kelly_fraction = self.risk_manager.calculate_kelly_with_uncertainty(
                    estimated_probability=final_probability if direction == "YES" else (1 - final_probability),
                    confidence=confidence,
                    price=market_price
                )

            # Base position size
            bankroll = self.risk_manager.get_available_bankroll() if self.risk_manager else 1000
            base_size = kelly_fraction * bankroll

            # 8. Entry timing analysis
            timing_signal = None
            timing_multiplier = 1.0

            if self.timing:
                try:
                    timing_signal = await self.timing.analyze_timing(market, direction, base_size)
                    timing_multiplier = getattr(timing_signal, 'size_multiplier', 1.0)
                except Exception as e:
                    logger.debug(f"Timing analysis failed: {e}")

            # 9. Apply all adjustments to position size
            recommended_size = base_size * timing_multiplier

            if self.risk_manager:
                recommended_size = self.risk_manager.apply_position_limits(recommended_size, market)

            # 10. Final actionability check
            is_actionable = (
                edge >= thresholds["min_edge"] and
                confidence >= thresholds["min_confidence"] and
                recommended_size >= 5.0 and  # Minimum $5 trade
                (not timing_signal or getattr(timing_signal, 'should_trade_now', True))
            )

            # 11. Check for breaking news
            is_breaking_news = any(
                item.get("is_breaking", False) or item.get("impact_score", 0) > 0.8
                for item in news_context
            ) if news_context else False

            # 12. Build enhanced signal
            signal = EnhancedTradingSignal(
                market=market,
                direction=direction,
                ai_probability=ai_probability,
                calibrated_probability=calibrated_probability,
                final_probability=final_probability,
                market_probability=market_price,
                edge=edge,
                confidence=confidence,
                expected_value=expected_value,
                kelly_fraction=kelly_fraction,
                recommended_size=recommended_size,
                model_votes=model_votes,
                reasoning=response.get("reasoning", ""),
                news_context=news_context,
                external_consensus=external_consensus,
                timing_signal=timing_signal.__dict__ if timing_signal else None,
                category_config=category_config.__dict__ if category_config else None,
                contrarian_signals=[c.__dict__ for c in contrarian_signals] if contrarian_signals else [],
                historical_edge_adjustment=historical_adjustment,
                is_actionable=is_actionable,
                is_breaking_news=is_breaking_news
            )

            # 13. Track for accuracy analysis
            if self.accuracy_tracker:
                try:
                    self.accuracy_tracker.record_prediction(signal)
                except Exception as e:
                    logger.debug(f"Accuracy tracking failed: {e}")

            self.signals_generated += 1

            return signal

        except Exception as e:
            logger.exception(f"Market analysis failed: {e}")
            return None

    def _build_enhanced_prompt(
        self,
        market: Any,
        news: List[Dict],
        external: Optional[Dict],
        contrarian: List,
        category_config: Optional[Any]
    ) -> str:
        """Build comprehensive analysis prompt with all context."""
        question = getattr(market, 'question', 'Unknown Question')
        yes_price = getattr(market, 'yes_price', 0.5)
        no_price = getattr(market, 'no_price', 0.5)
        volume = getattr(market, 'volume', 0)
        liquidity = getattr(market, 'liquidity', 0)
        category = getattr(market, 'category', 'unknown')

        prompt = f"""POLYMARKET TRADING ANALYSIS

QUESTION: {question}

CURRENT MARKET STATE:
- YES Price: {yes_price*100:.1f}% (${yes_price:.3f})
- NO Price: {no_price*100:.1f}% (${no_price:.3f})
- Volume: ${volume:,.0f}
- Liquidity: ${liquidity:,.0f}
- Category: {category}
- Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

        # Add category-specific context
        if category_config:
            prompt += f"""
CATEGORY CONTEXT:
- Trading enabled: {getattr(category_config, 'enabled', True)}
- Minimum edge: {getattr(category_config, 'min_edge', 0.08)*100:.0f}%
- News sources: {', '.join(getattr(category_config, 'news_sources', []))}
"""

        # Add news context
        if news:
            prompt += f"""
📰 RECENT NEWS CONTEXT ({len(news)} items):
"""
            for i, item in enumerate(news[:5], 1):
                headline = item.get('headline', 'Unknown')
                source = item.get('source', 'Unknown')
                time_ago = item.get('time_ago', 'Unknown')
                impact = item.get('impact_score', 0)
                prompt += f"{i}. [{source}] {headline} ({time_ago}) - Impact: {impact:.1f}\n"

        # Add external consensus
        if external and external.get("sources_count", 0) > 0:
            prompt += f"""
📊 EXTERNAL CONSENSUS ({external.get('sources_count', 0)} sources):
- Weighted Average: {external.get('weighted_average', 0)*100:.1f}%
- Simple Average: {external.get('simple_average', 0)*100:.1f}%
- Sources: {', '.join(external.get('sources', {}).keys())}
"""

        # Add contrarian signals
        if contrarian:
            prompt += f"""
⚠️ CONTRARIAN SIGNALS ({len(contrarian)} detected):
"""
            for i, signal in enumerate(contrarian[:3], 1):
                signal_type = signal.get('signal_type', 'unknown')
                evidence = signal.get('evidence', [''])[0] if signal.get('evidence') else ''
                confidence = signal.get('confidence', 0)
                prompt += f"{i}. {signal_type}: {evidence} (conf: {confidence:.1f})\n"

        prompt += """
ANALYSIS REQUIREMENTS:

1. **Probability Assessment**: Estimate the true probability of YES outcome (0-100%)

2. **Context Integration**: Consider how news, external consensus, and contrarian signals affect your estimate

3. **Market Efficiency**: Is the current market price reflecting all available information?

4. **Edge Identification**: Where is the mispricing? What creates the edge?

5. **Risk Factors**: What could go wrong? Are there tail risks?

6. **Confidence Level**: How confident are you in your estimate? (0-100%)

STRUCTURED RESPONSE FORMAT:
- Probability: [XX]% (with brief justification)
- Key Factors: [3-5 main factors influencing outcome]
- Market Analysis: [Is market efficient? Over/under-priced?]
- Confidence: [XX]% (why this level?)
- Edge Assessment: [Where's the alpha?]

Your analysis:"""

        return prompt

    async def find_opportunities(
        self,
        min_edge: float = 0.08,
        min_confidence: float = 0.5,
        limit: int = 10,
        include_news: bool = True,
        include_external: bool = True
    ) -> List[EnhancedTradingSignal]:
        """
        Scan markets for trading opportunities using full analysis pipeline.

        Returns sorted list of actionable signals.
        """
        logger.info(f"🔍 Scanning for opportunities (edge≥{min_edge*100:.0f}%, conf≥{min_confidence*100:.0f}%, limit={limit})")

        try:
            # Get markets from Polymarket
            if not polymarket:
                logger.error("Polymarket client not available")
                return []

            markets = await polymarket.get_markets(
                limit=min(limit * 3, 50),  # Get more to filter
                min_volume=25000,    # $25k minimum volume (recommended)
                min_liquidity=25000  # $25k minimum liquidity (anti-slippage)
            )

            if not markets:
                logger.warning("No markets found matching criteria")
                return []

            logger.info(f"📊 Analyzing {len(markets)} markets with full pipeline...")

            # Analyze markets (with rate limiting)
            signals = []
            semaphore = asyncio.Semaphore(5)  # Limit concurrent analyses

            async def analyze_with_limit(market):
                async with semaphore:
                    signal = await self.analyze_market(
                        market,
                        deep_analysis=False,
                        include_news=include_news,
                        include_external=include_external
                    )
                    if signal and signal.is_actionable and signal.edge >= min_edge and signal.confidence >= min_confidence:
                        return signal
                    return None

            # Run analyses in parallel with rate limiting
            tasks = [analyze_with_limit(market) for market in markets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid signals
            exception_count = 0
            for result in results:
                if isinstance(result, Exception):
                    exception_count += 1
                    continue
                if result:
                    signals.append(result)
            if exception_count:
                logger.warning(f"{exception_count} market analyses raised exceptions (skipped)")

            # Sort by quality score (edge * confidence)
            signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)

            logger.info(f"✅ Found {len(signals)} actionable signals")

            return signals[:limit]

        except Exception as e:
            logger.exception(f"Opportunity scanning failed: {e}")
            return []

    async def execute_signal(self, signal: EnhancedTradingSignal) -> Dict:
        """
        Execute a trading signal with full risk management.

        Returns execution result.
        """
        if not signal.is_actionable:
            return {
                "status": "skipped",
                "reason": signal.skip_reason or "Signal not actionable"
            }

        try:
            # Pre-execution risk checks
            if self.risk_manager:
                can_trade, reason = self.risk_manager.can_trade(signal)
                if not can_trade:
                    return {"status": "blocked", "reason": reason}

            # Drawdown check
            if self.drawdown and self.drawdown.should_pause_trading():
                return {"status": "paused", "reason": "Drawdown limit reached"}

            # Position size validation
            if signal.recommended_size < 1.0:
                return {"status": "skipped", "reason": "Position size too small"}

            # Execute trade
            if self.paper_trading:
                # Paper trade simulation
                result = {
                    "status": "paper_executed",
                    "market": getattr(signal.market, 'question', 'Unknown')[:50],
                    "direction": signal.direction,
                    "size": signal.recommended_size,
                    "price": signal.market_probability,
                    "edge": signal.edge,
                    "expected_value": signal.expected_value,
                    "signal_quality": signal.signal_quality,
                    "timestamp": datetime.now().isoformat()
                }

                logger.success(
                    f"📝 PAPER TRADE: {signal.direction} ${signal.recommended_size:.2f} "
                    f"@ {signal.market_probability:.2f} (edge: {signal.edge:.1%}, "
                    f"quality: {signal.signal_quality})"
                )

            else:
                # Live trade execution
                if not polymarket:
                    return {"status": "error", "reason": "Polymarket client not available"}

                execution_result = await polymarket.place_order_for_market(
                    market=signal.market,
                    direction=signal.direction,
                    size_usd=signal.recommended_size,
                    order_type="limit"
                )

                result = {
                    "status": "executed",
                    "execution": execution_result,
                    "market": getattr(signal.market, 'question', 'Unknown')[:50],
                    "direction": signal.direction,
                    "size": signal.recommended_size,
                    "timestamp": datetime.now().isoformat()
                }

                logger.success(
                    f"💰 LIVE TRADE: {signal.direction} ${signal.recommended_size:.2f} "
                    f"(quality: {signal.signal_quality})"
                )

            # Track position if position manager available
            if self.position_manager:
                try:
                    await self.position_manager.add_position(signal, result)
                except Exception as e:
                    logger.warning(f"Position tracking failed: {e}")

            # Update performance tracker
            if self.performance:
                try:
                    # This would be updated with actual P&L later
                    pass
                except Exception as e:
                    logger.debug(f"Performance tracking failed: {e}")

            self.trades_executed += 1

            return result

        except Exception as e:
            logger.exception(f"Signal execution failed: {e}")
            return {"status": "error", "reason": str(e)}

    async def run_trading_loop(
        self,
        interval_minutes: int = 30,
        max_trades_per_day: int = 10,
        min_edge: float = 0.08,
        min_confidence: float = 0.5
    ):
        """
        Main automated trading loop.

        Continuously scans for opportunities and executes trades
        with comprehensive risk management.
        """
        logger.info(f"🚀 Starting TradingSwarm V2 loop (interval={interval_minutes}m, max_trades/day={max_trades_per_day})")

        trades_today = 0
        last_reset = datetime.now().date()

        while True:
            try:
                # Reset daily trade counter
                if datetime.now().date() > last_reset:
                    trades_today = 0
                    last_reset = datetime.now().date()
                    logger.info("🌅 New trading day started")

                # Check if we've hit daily trade limit
                if trades_today >= max_trades_per_day:
                    logger.info(f"📊 Daily trade limit reached ({max_trades_per_day})")
                    await asyncio.sleep(interval_minutes * 60)
                    continue

                # Find trading opportunities
                signals = await self.find_opportunities(
                    min_edge=min_edge,
                    min_confidence=min_confidence,
                    limit=min(5, max_trades_per_day - trades_today)
                )

                # Execute signals
                for signal in signals:
                    if trades_today >= max_trades_per_day:
                        break

                    result = await self.execute_signal(signal)
                    if result.get("status") in ["paper_executed", "executed"]:
                        trades_today += 1

                # Position management and updates
                if self.position_manager:
                    try:
                        await self.position_manager.update_all_positions()
                    except Exception as e:
                        logger.warning(f"Position management failed: {e}")

                # Update risk metrics
                if self.drawdown and self.performance:
                    try:
                        # Update drawdown and performance with current portfolio value
                        # This would integrate with actual portfolio tracking
                        pass
                    except Exception as e:
                        logger.debug(f"Risk metrics update failed: {e}")

                # Log current status
                self._log_status()

                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("🛑 Trading loop stopped by user")
                self._log_final_stats()
                break
            except Exception as e:
                logger.exception(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Brief pause on errors

    def _log_status(self):
        """Log current trading status."""
        runtime = datetime.now() - self.start_time
        hours = runtime.total_seconds() / 3600

        status = (
            f"📈 Status: {hours:.1f}h runtime | "
            f"Signals: {self.signals_generated} | "
            f"Trades: {self.trades_executed} | "
            f"API Cost: ${self.total_api_cost:.2f}"
        )

        if self.drawdown:
            dd_pct = self.drawdown.current_drawdown * 100
            status += f" | Drawdown: {dd_pct:.1f}%"

        logger.info(status)

    def _log_final_stats(self):
        """Log final trading statistics."""
        runtime = datetime.now() - self.start_time
        hours = runtime.total_seconds() / 3600

        logger.info("🏁 TradingSwarm V2 Session Complete")
        logger.info("=" * 50)
        logger.info(f"Runtime: {hours:.1f} hours")
        logger.info(f"Signals Generated: {self.signals_generated}")
        logger.info(f"Trades Executed: {self.trades_executed}")
        logger.info(f"API Cost: ${self.total_api_cost:.2f}")
        logger.info(f"Average Cost per Signal: ${self.total_api_cost/max(self.signals_generated, 1):.3f}")

        if self.performance:
            logger.info("Performance metrics would be displayed here")

        logger.info("=" * 50)

    async def get_portfolio_status(self) -> Dict:
        """Get comprehensive portfolio status."""
        status = {
            "paper_trading": self.paper_trading,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "total_api_cost": self.total_api_cost,
            "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "components": {}
        }

        # Component status
        status["components"] = {
            "swarm": self.swarm is not None,
            "calibrator": self.calibrator is not None,
            "risk_manager": self.risk_manager is not None,
            "timing": self.timing is not None,
            "category": self.category is not None,
            "contrarian": self.contrarian is not None,
            "position_manager": self.position_manager is not None,
            "news": self.news is not None,
            "consensus": self.consensus is not None,
            "patterns": self.patterns is not None,
            "accuracy_tracker": self.accuracy_tracker is not None,
        }

        # Position status
        if self.position_manager:
            try:
                position_summary = self.position_manager.get_portfolio_summary()
                status["positions"] = position_summary
            except Exception as e:
                logger.debug(f"Position status failed: {e}")
                status["positions"] = {"error": str(e)}

        # Risk status
        if self.risk_manager and self.drawdown:
            try:
                status["risk"] = {
                    "current_drawdown": self.drawdown.current_drawdown,
                    "should_pause": self.drawdown.should_pause_trading(),
                    "available_bankroll": self.risk_manager.get_available_bankroll()
                }
            except Exception as e:
                logger.debug(f"Risk status failed: {e}")

        return status
