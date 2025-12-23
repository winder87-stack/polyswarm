"""
Trading Swarm Agent

Combines AI swarm analysis with Polymarket trading execution
for automated prediction market trading.

Author: Polymarket Trading Bot
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from statistics import mean, stdev

from termcolor import colored
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler

from .swarm_agent import SwarmAgent
from src.connectors import polymarket, Market
from src.analysis.ai_accuracy_tracker import create_accuracy_tracker
from src.analysis.model_calibration import model_calibration
from src.strategies.entry_timing import entry_timing_optimizer
from src.strategies.position_manager import create_position_manager
from src.strategies.category_specialist import category_specialist
from src.strategies.contrarian_detector import contrarian_detector
from src.services.consensus_detector import consensus_detector

# Import shared signal definitions
from .signals import TradingSignal


class TradingSwarm:
    """Combines AI swarm analysis with Polymarket trading execution."""

    def __init__(self) -> None:
        """Initialize advanced trading swarm with expected value analysis and Kelly sizing."""
        # Load enhanced configuration from environment
        # NOTE: keep env var compatibility with older configs, but default to repo risk rules.
        self.min_edge = float(os.getenv("MIN_EDGE", "0.08"))  # Minimum edge to trade (8% default)
        self.max_position = float(os.getenv("MAX_POSITION_SIZE", os.getenv("MAX_POSITION", "100")))  # Max position size in USDC
        # IMPORTANT: matches repo slippage protection rules (Min Liquidity: $25,000)
        self.min_liquidity = float(os.getenv("MIN_LIQUIDITY", "25000"))
        self.min_volume = float(os.getenv("MIN_VOLUME", "50000"))  # Minimum volume (50k default)
        self.min_market_hours = int(os.getenv("MIN_MARKET_HOURS", "24"))  # Min hours until expiry
        self.expected_value_threshold = float(os.getenv("EV_THRESHOLD", "0.02"))  # Min EV to trade
        self.fee_rate = float(os.getenv("FEE_RATE", "0.02"))  # Polymarket fee rate (2%)
        self.kelly_fraction = float(os.getenv("KELLY_FRACTION", "0.25"))  # Kelly fraction (conservative)
        self.daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "500"))  # Daily loss limit
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"

        # Initialize AI swarm
        try:
            self.swarm = SwarmAgent()
            logger.info("✅ Advanced AI Swarm initialized")
        except Exception as e:
            logger.error(f"Failed to initialize AI swarm: {e}")
            raise

        # Initialize category specialist
        self.category_specialist = category_specialist
        logger.info("✅ Category specialist integrated")

        # Initialize calibration tracker
        self.calibration_tracker = model_calibration
        logger.info("✅ Model calibration tracker integrated")

        # Initialize position manager
        self.position_manager = create_position_manager(self)
        logger.info("✅ Position manager integrated")

        # Initialize consensus detector (READ-ONLY external data)
        self.consensus_detector = consensus_detector
        logger.info("✅ Consensus detector integrated (READ-ONLY external data)")

        # Initialize contrarian detector
        self.contrarian_detector = contrarian_detector
        logger.info("✅ Contrarian detector integrated")

        # Trading statistics and P&L tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.daily_pnl = 0.0
        self.daily_start_time = datetime.now().date()
        self.position_log = []  # Track open positions

        # AI Accuracy tracking
        self.accuracy_tracker = create_accuracy_tracker()

        # Configuration summary
        mode = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        print(colored(f"\n🎯 Advanced Trading Swarm initialized in {mode} mode", "cyan", attrs=["bold"]))
        print(colored(f"📊 Min edge: {self.min_edge:.1%} | Min EV: {self.expected_value_threshold:.1%}", "white"))
        print(colored(f"💰 Max position: ${self.max_position:,.0f} | Kelly fraction: {self.kelly_fraction:.1%}", "white"))
        print(colored(f"🏪 Min liquidity: ${self.min_liquidity:,.0f} | Min volume: ${self.min_volume:,.0f}", "white"))
        print(colored(f"⏰ Min hours to expiry: {self.min_market_hours}h | Fee rate: {self.fee_rate:.1%}", "white"))

    async def analyze_market(self, market: Market, recent_news: Optional[List[Any]] = None) -> Optional[TradingSignal]:
        """
        Analyze a market using AI swarm and generate trading signal.

        Args:
            market: Market to analyze
            recent_news: Optional list of recent NewsItem objects for context

        Returns:
            TradingSignal or None if analysis fails
        """
        try:
            # ===== CATEGORY CHECK =====
            should_trade, reason = self.category_specialist.should_trade_category(market)
            if not should_trade:
                market_volume = getattr(market, 'volume', 0)
                market_liquidity = getattr(market, 'liquidity', 0)
                logger.info(f"⏭️ Skipping {market.question[:40]}...: {reason} (vol=${market_volume:,.0f}, liq=${market_liquidity:,.0f})")
                return None

            print(colored(f"\n🎯 Analyzing market: {market.question[:80]}...", "blue", attrs=["bold"]))

            # Get category-adjusted model weights and thresholds
            category_weights = self.category_specialist.get_adjusted_weights(market)
            category_thresholds = self.category_specialist.get_adjusted_thresholds(market)

            logger.info(f"🎯 Category: {getattr(market, 'category', 'unknown')} | Weights: {category_weights}")

            # Build analysis prompt (news-aware if news provided)
            if recent_news:
                prompt = self._build_news_aware_prompt(market, recent_news)
                print(colored(f"📰 Including {len(recent_news)} recent news items in analysis", "blue"))
            else:
                prompt = self._build_analysis_prompt(market)

            # Choose analysis method based on trade importance
            # Note: We don't know the edge yet, so we use a preliminary check
            is_high_stakes = getattr(market, 'volume', 0) > 50000 or getattr(market, 'liquidity', 0) > 10000

            if is_high_stakes:
                logger.info("🔬 High-stakes market detected - running deep analysis")
                # Use comprehensive deep analysis for important markets
                deep_result = await self.swarm.deep_analysis(market, recent_news or "")
                ai_probability = deep_result["final_probability"]

                # Store analysis details for logging
                analysis_methods = ["deep_analysis"]
                analysis_details = deep_result

                # Extract model votes from deep analysis
                model_votes = deep_result.get("method_probabilities", {})
                probabilities = list(model_votes.values()) if model_votes else [ai_probability]
                confidence = self._calculate_confidence(probabilities) if len(probabilities) > 1 else 0.7

                # Mock swarm_result for compatibility
                swarm_result = {"consensus": f"Deep analysis: {ai_probability:.1%}", "consensus_summary": "Deep analysis completed", "responses": {}}

            else:
                # Use standard swarm query for regular trades
                swarm_result = await self.swarm.query(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=30,
                    custom_weights=category_weights
                )
                ai_probability = self._extract_probability(swarm_result["consensus"])

                analysis_methods = ["standard_swarm"]
                analysis_details = swarm_result

                # Extract probabilities from responses
                model_votes = {}
                probabilities = []

                for provider, response_data in swarm_result["responses"].items():
                    if response_data["success"] and response_data["content"]:
                        prob = self._extract_probability(response_data["content"])
                        if prob is not None:
                            model_votes[provider] = prob
                            probabilities.append(prob)
                            print(colored(f"📊 {provider}: {prob:.1%}", "white"))
                        else:
                            print(colored(f"❓ {provider}: probability not found", "yellow"))

                if not probabilities:
                    print(colored("❌ No valid probabilities extracted", "red"))
                    return None

                confidence = self._calculate_confidence(probabilities)

            # Determine direction and edge
            market_prob = market.yes_price  # YES price represents probability of YES outcome

            if ai_probability > market_prob:
                direction = "YES"
                edge = ai_probability - market_prob
            else:
                direction = "NO"
                edge = (1 - ai_probability) - (1 - market_prob)  # Edge for NO position

            # Apply category-adjusted thresholds
            min_edge_threshold = category_thresholds["min_edge"]
            min_confidence_threshold = category_thresholds["min_confidence"]

            print(colored(f"🎲 AI Probability: {ai_probability:.1%}", "cyan"))
            print(colored(f"📈 Market Price: {market_prob:.1%}", "cyan"))
            print(colored(f"🎯 Edge: {edge:+.1%} (min: {min_edge_threshold:.1%})", "green" if edge > 0 else "red"))
            print(colored(f"📊 Confidence: {confidence:.1%} (min: {min_confidence_threshold:.1%})", "cyan"))

            # Check category-adjusted thresholds
            if abs(edge) < min_edge_threshold:
                print(colored(f"❌ Edge too small: {abs(edge):.1%} < {min_edge_threshold:.1%}", "red"))
                return None

            if confidence < min_confidence_threshold:
                print(colored(f"❌ Confidence too low: {confidence:.1%} < {min_confidence_threshold:.1%}", "red"))
                return None

            # Calculate expected value
            expected_value = edge * confidence

            # Calculate news impact if news provided
            news_impact_score = 0.0
            news_summary = ""
            if recent_news:
                # Calculate average impact of all news items
                impacts = [self.score_news_impact(news, market) for news in recent_news]
                news_impact_score = sum(impacts) / len(impacts) if impacts else 0.0

                # Create news summary
                top_news = sorted(zip(recent_news, impacts), key=lambda x: x[1], reverse=True)[:3]
                news_summaries = []
                for news, impact in top_news:
                    time_ago = self._format_time_ago(news.timestamp)
                    news_summaries.append(f"{news.headline[:50]}... ({time_ago})")
                news_summary = " | ".join(news_summaries)

            # ===== EXTERNAL CONSENSUS INTEGRATION (READ-ONLY) =====
            # Get consensus from external prediction platforms (READ-ONLY data)
            try:
                external_consensus = await self.consensus_detector.get_external_consensus(market)

                # Combine AI probability with external consensus
                if external_consensus["sources_count"] >= 2:
                    # Weight: 70% AI swarm, 30% external consensus
                    combined_probability = 0.7 * ai_probability + 0.3 * external_consensus["weighted_average"]

                    # Boost confidence if AI and external agree
                    agreement = abs(ai_probability - external_consensus["weighted_average"])
                    if agreement < 0.05:  # Within 5%
                        confidence *= 1.2  # 20% confidence boost
                        reasoning += f" | External consensus agrees ({external_consensus['sources_count']} sources)"
                    elif agreement > 0.15:  # Significant disagreement
                        confidence *= 0.9  # 10% confidence reduction
                        reasoning += f" | External consensus differs by {agreement:.1%}"

                    # Update probability with combined estimate
                    ai_probability = combined_probability

                    logger.info(f"🎲 External consensus: {external_consensus['weighted_average']:.1%} from {external_consensus['sources_count']} sources")
                    logger.info(f"📊 Combined probability: {ai_probability:.1%} (READ-ONLY external data)")

                else:
                    logger.debug("Insufficient external consensus data for this market")

            except Exception as e:
                logger.warning(f"External consensus integration failed: {e}")

            # ===== CONTRARIAN SIGNAL DETECTION =====
            # Check for contrarian opportunities
            contrarian_signals = []
            try:
                all_contrarian = await self.contrarian_detector.scan_all_contrarian([market])
                contrarian_signals = all_contrarian[:3]  # Top 3 contrarian signals

                if contrarian_signals:
                    # Boost confidence if contrarian signals agree with AI direction
                    ai_direction = "YES" if ai_probability > 0.5 else "NO"
                    contrarian_agrees = any(
                        signal.suggested_direction == ai_direction and signal.contrarian_score > 0.6
                        for signal in contrarian_signals
                    )

                    if contrarian_agrees:
                        confidence *= 1.15  # 15% confidence boost
                        reasoning += f" | Contrarian analysis agrees ({len(contrarian_signals)} signals)"

                    logger.info(f"🔄 Found {len(contrarian_signals)} contrarian signals")

            except Exception as e:
                logger.warning(f"Contrarian signal detection failed: {e}")

            # Create trading signal with analysis details
            if "deep_analysis" in analysis_methods:
                # Deep analysis result
                reasoning = f"Deep analysis: {ai_probability:.1%} probability from {len(analysis_details.get('method_probabilities', {}))} methods"
                consensus_summary = f"Methods: {', '.join(analysis_details.get('method_probabilities', {}).keys())}"
                signal_model_weights = analysis_details.get("method_weights", {})
            else:
                # Standard swarm result
                reasoning = f"AI swarm consensus: {len(probabilities)} models, avg probability {ai_probability:.1%}"
                consensus_summary = swarm_result.get("consensus_summary", "")
                signal_model_weights = category_weights  # Category-adjusted weights

            signal = TradingSignal(
                market=market,
                direction=direction,
                confidence=confidence,
                probability=ai_probability,
                market_probability=market_prob,
                edge=edge,
                expected_value=expected_value,
                reasoning=reasoning,
                model_votes=model_votes,
                model_weights=signal_model_weights,
                consensus_summary=consensus_summary,
                news_context=news_summary,
                timestamp=datetime.now(),
                news_items=recent_news if recent_news else None,
                news_impact_score=news_impact_score,
                contrarian_signals=contrarian_signals,
                is_breaking_news_trade=False
            )

            # Add analysis details if available (extend TradingSignal if needed)
            if hasattr(signal, 'analysis_details'):
                signal.analysis_details = analysis_details
            if hasattr(signal, 'analysis_methods'):
                signal.analysis_methods = analysis_methods

            # Record prediction for accuracy tracking
            self.accuracy_tracker.record_prediction(signal)

            # Record predictions for calibration (each model's contribution)
            market_id = getattr(market, 'condition_id', getattr(market, 'slug', 'unknown'))
            category = getattr(market, 'category', 'unknown')

            if "deep_analysis" in analysis_methods:
                # Record method-level predictions for calibration
                method_probs = analysis_details.get("method_probabilities", {})
                for method, prob in method_probs.items():
                    self.calibration_tracker.record_prediction(method, prob, market_id, category)
            else:
                # Record individual model predictions
                for model, prob in model_votes.items():
                    self.calibration_tracker.record_prediction(model, prob, market_id, category)

            self.signals_generated += 1
            return signal

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            print(colored(f"❌ Analysis failed: {e}", "red"))
            return None

    def get_position_manager(self) -> 'PositionManager':
        """Get the position manager instance."""
        return self.position_manager

    async def run_position_management(self, interval_minutes: int = 60):
        """
        Run the position management loop.

        This will continuously monitor and manage open positions.
        """
        await self.position_manager.run_management_loop(interval_minutes)

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary from position manager."""
        return self.position_manager.get_portfolio_summary()

    async def analyze_market_with_news(self, market: Any, news_item: Any) -> Dict[str, Any]:
        """
        Analyze a market with breaking news context for fast trading decisions.

        Args:
            market: Market object to analyze
            news_item: NewsItem with breaking news

        Returns:
            Dict with analysis results including edge and confidence
        """
        try:
            # Build enhanced prompt with news context
            prompt = self._build_news_analysis_prompt(market, news_item)

            # Query AI swarm
            responses = await self.swarm.query(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=800,
                timeout=30  # Faster timeout for breaking news
            )

            # Parse responses for edge and confidence
            consensus = self._parse_news_analysis_responses(responses, market)

            logger.info(f"📈 News analysis complete - Edge: {consensus['edge']:+.1%}, Confidence: {consensus['confidence']:.1%}")
            return consensus

        except Exception as e:
            logger.error(f"❌ Error in news analysis for market {getattr(market, 'slug', 'unknown')}: {e}")
            # Return conservative defaults
            return {
                "edge": 0.0,
                "confidence": 0.5,
                "probability": 0.5,
                "reasoning": f"Analysis failed: {e}",
                "fast_trade_recommended": False
            }

    def _build_news_analysis_prompt(self, market: Any, news_item: Any) -> str:
        """Build analysis prompt that includes breaking news context."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        prompt = f"""BREAKING NEWS ANALYSIS - {current_time}

Market: {market.question}
Current Yes Price: {getattr(market, 'yes_price', 'Unknown'):.3f}
Current No Price: {getattr(market, 'no_price', 'Unknown'):.3f}
Liquidity: {getattr(market, 'liquidity', 'Unknown'):,.0f}
Volume: {getattr(market, 'volume', 'Unknown'):,.0f}

BREAKING NEWS ({news_item.source}):
Headline: {news_item.headline}
Summary: {news_item.summary}
Category: {news_item.category}
Published: {news_item.timestamp.strftime('%H:%M:%S UTC')}

INSTRUCTIONS:
This is BREAKING NEWS that may immediately impact this prediction market.

1. Analyze how this news affects the market outcome probability
2. Estimate the new "true" probability after this news
3. Calculate the edge (new_probability - current_market_price)
4. Assess your confidence in this analysis (0-100%)

Respond with format:
NEW_PROBABILITY: [0.00-1.00]
EDGE: [±0.000]
CONFIDENCE: [0-100]
REASONING: [2-3 sentences explaining your analysis]
FAST_TRADE: [YES/NO - only if edge > 15% and confidence > 70%]
"""

        return prompt

    def _parse_news_analysis_responses(self, responses: Dict[str, Any], market: Any) -> Dict[str, Any]:
        """Parse AI responses for news analysis."""
        try:
            # Extract consensus from responses
            all_probabilities = []
            all_edges = []
            all_confidences = []
            all_reasonings = []

            for response in responses.get('responses', []):
                content = response.get('content', '')

                # Parse probability
                prob_match = re.search(r'NEW_PROBABILITY:\s*([0-9.]+)', content, re.IGNORECASE)
                if prob_match:
                    all_probabilities.append(float(prob_match.group(1)))

                # Parse edge
                edge_match = re.search(r'EDGE:\s*([+-]?[0-9.]+)', content, re.IGNORECASE)
                if edge_match:
                    all_edges.append(float(edge_match.group(1)))

                # Parse confidence
                conf_match = re.search(r'CONFIDENCE:\s*([0-9]+)', content, re.IGNORECASE)
                if conf_match:
                    all_confidences.append(int(conf_match.group(1)))

                # Extract reasoning
                reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', content, re.DOTALL | re.IGNORECASE)
                if reasoning_match:
                    all_reasonings.append(reasoning_match.group(1).strip())

            # Calculate consensus
            if all_probabilities:
                avg_probability = sum(all_probabilities) / len(all_probabilities)
                market_price = getattr(market, 'yes_price', 0.5)
                avg_edge = avg_probability - market_price
            else:
                avg_probability = 0.5
                avg_edge = 0.0

            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 50
            confidence_norm = avg_confidence / 100.0

            # Combine reasonings
            reasoning = " | ".join(all_reasonings[:2]) if all_reasonings else "No reasoning provided"

            # Determine if fast trade is recommended
            fast_trade = (abs(avg_edge) >= self.expected_value_threshold and
                         confidence_norm >= 0.7)

            return {
                "probability": avg_probability,
                "edge": avg_edge,
                "confidence": confidence_norm,
                "reasoning": reasoning,
                "fast_trade_recommended": fast_trade,
                "model_responses": len(responses.get('responses', []))
            }

        except Exception as e:
            logger.error(f"❌ Error parsing news analysis responses: {e}")
            return {
                "probability": 0.5,
                "edge": 0.0,
                "confidence": 0.5,
                "reasoning": f"Parsing failed: {e}",
                "fast_trade_recommended": False
            }

    def _build_news_aware_prompt(self, market: Market, recent_news: List[Any]) -> str:
        """
        Build analysis prompt that includes recent news context.

        Args:
            market: Market to analyze
            recent_news: List of NewsItem objects

        Returns:
            Enhanced prompt string with news context
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Filter news to last 24 hours and relevant items
        cutoff_time = datetime.now() - timedelta(hours=24)
        relevant_news = []

        for news in recent_news:
            # Check if news is recent enough
            if news.timestamp > cutoff_time:
                # Calculate relevance to market
                relevance = self.score_news_impact(news, market)
                if relevance > 0.3:  # Only include moderately relevant news
                    relevant_news.append((news, relevance))

        # Sort by relevance and recency
        relevant_news.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        relevant_news = relevant_news[:5]  # Limit to top 5 most relevant

        # Build news section
        news_section = ""
        if relevant_news:
            news_section = "\n📰 RECENT RELATED NEWS (last 24h):\n"
            for i, (news, relevance) in enumerate(relevant_news, 1):
                time_ago = self._format_time_ago(news.timestamp)
                news_section += f"{i}. {news.headline} - {news.source} - {time_ago}\n"
                news_section += f"   Summary: {news.summary[:150]}{'...' if len(news.summary) > 150 else ''}\n"
                news_section += f"   Relevance: {relevance:.1%}\n\n"

        # Build the enhanced prompt
        prompt = f"""NEWS-AWARE MARKET ANALYSIS - {current_time}

MARKET: {market.question}
CURRENT PRICE: YES ${market.yes_price:.3f} / NO ${market.no_price:.3f}
LIQUIDITY: ${market.liquidity:,.0f}
VOLUME: ${market.volume:,.0f}
MARKET SLUG: {getattr(market, 'slug', 'unknown')}

{news_section}
ANALYSIS INSTRUCTIONS:

1. Consider how the recent news affects the probability of this market outcome
2. Has the market already priced in this news? Look at the current market price.
3. Is there an edge based on news the market hasn't fully reacted to?
4. Account for news recency - more recent news should have stronger impact
5. Consider market efficiency - major news is usually priced in quickly

Respond with your probability estimate considering all available information.
"""

        return prompt

    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as human-readable time ago."""
        now = datetime.now()
        diff = now - timestamp

        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "just now"

    def score_news_impact(self, news: Any, market: Any) -> float:
        """
        Score how much a news item should affect a market.

        Returns 0-1:
        - 0.9+: Direct resolution (e.g., "Trump wins Iowa" for "Will Trump win Iowa?")
        - 0.7-0.9: Highly relevant (same topic, new information)
        - 0.4-0.7: Moderately relevant (related topic)
        - 0.0-0.4: Tangentially related or irrelevant

        Uses keyword matching + semantic analysis.
        """
        try:
            market_text = f"{market.question} {getattr(market, 'description', '')}".lower()
            news_text = f"{news.headline} {news.summary}".lower()

            # Extract key entities and topics
            market_words = set(re.findall(r'\b\w+\b', market_text))
            news_words = set(re.findall(r'\b\w+\b', news_text))

            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'will', 'be'}
            market_words = market_words - stop_words
            news_words = news_words - stop_words

            # Calculate word overlap
            overlap = len(market_words.intersection(news_words))
            union = len(market_words.union(news_words))

            if union == 0:
                return 0.0

            base_similarity = overlap / union

            # Boost for direct matches in headline
            headline_boost = 0.0
            headline_lower = news.headline.lower()
            question_lower = market.question.lower()

            # Check for direct resolution indicators
            if any(phrase in headline_lower for phrase in ['wins', 'won', 'loses', 'lost', 'defeats', 'defeated', 'victory', 'victorious']):
                if any(word in question_lower for word in ['win', 'wins', 'victory', 'defeat']):
                    headline_boost = 0.5

            # Check for company/product specific news
            company_indicators = ['announces', 'launches', 'releases', 'acquires', 'merges', 'bankruptcy', 'lawsuit']
            if any(indicator in headline_lower for indicator in company_indicators):
                headline_boost = 0.3

            # Time decay - more recent news has higher impact
            hours_old = (datetime.now() - news.timestamp).total_seconds() / 3600
            time_decay = max(0.1, 1.0 - (hours_old / 24.0))  # Full impact for <1h, decays over 24h

            # Source credibility boost
            source_boost = 0.0
            credible_sources = ['reuters', 'ap news', 'bbc', 'bloomberg', 'wsj', 'nyt', 'cnn']
            if any(source.lower() in news.source.lower() for source in credible_sources):
                source_boost = 0.1

            # Category relevance
            category_boost = 0.0
            market_categories = self._extract_market_categories(market)
            if news.category in market_categories:
                category_boost = 0.2

            final_score = min(1.0, (base_similarity * 0.6) + headline_boost + source_boost + category_boost) * time_decay

            return final_score

        except Exception as e:
            logger.warning(f"Error scoring news impact: {e}")
            return 0.0

    def _extract_market_categories(self, market: Any) -> List[str]:
        """Extract relevant categories for a market."""
        question = market.question.lower()
        categories = []

        # Politics
        if any(word in question for word in ['election', 'president', 'government', 'political', 'party', 'candidate']):
            categories.append('politics')

        # Sports
        if any(word in question for word in ['game', 'match', 'championship', 'tournament', 'player', 'team', 'score']):
            categories.append('sports')

        # Crypto/Business
        if any(word in question for word in ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'stock', 'company', 'market']):
            categories.append('crypto')
            categories.append('business')

        # Entertainment
        if any(word in question for word in ['movie', 'film', 'actor', 'celebrity', 'award', 'music', 'tv']):
            categories.append('entertainment')

        return categories

    async def analyze_with_breaking_news(
        self,
        market: Any,
        news: Any,
        fast_mode: bool = True
    ) -> Optional[TradingSignal]:
        """
        Fast analysis when breaking news hits.

        Args:
            market: Market to analyze
            news: Breaking NewsItem
            fast_mode: Whether to use fast analysis mode

        Returns:
            TradingSignal or None if analysis fails
        """
        start_time = time.time()
        news_impact = self.score_news_impact(news, market)

        try:
            print(colored(f"🚨 BREAKING NEWS ANALYSIS: {news.headline[:60]}...", "red", attrs=["bold"]))
            print(colored(f"   Market: {market.question[:50]}...", "red"))
            print(colored(f"   News Impact Score: {news_impact:.1%}", "red"))

            if news_impact < 0.4:
                print(colored("   ⏭️  News impact too low, skipping fast analysis", "yellow"))
                return None

            # Build news-aware prompt
            prompt = self._build_news_aware_prompt(market, [news])

            # Fast mode: query only fastest models, prioritize Perplexity for news analysis
            if fast_mode:
                print(colored("   ⚡ FAST MODE: News-aware analysis with Perplexity priority", "yellow"))

                # Always query Perplexity first (has web search capabilities)
                responses = {}
                perplexity_insights = ""

                # Step 1: Query Perplexity first for news analysis
                if "perplexity" in self.swarm.models:
                    try:
                        perplexity_prompt = f"""You have real-time web search capabilities. Analyze this breaking news and its potential impact on the market.

{news.headline}

{news.summary}

Market: {market.question}

Provide insights on:
1. How this news might affect the market outcome
2. Any related recent developments
3. Market sentiment indicators

Keep response under 200 words."""

                        perplexity_response = await self.swarm.models["perplexity"].generate_response(
                            system_prompt="You are a financial news expert with real-time web search. Provide concise, actionable insights.",
                            user_content=perplexity_prompt,
                            temperature=0.3,
                            max_tokens=400,
                            timeout=30
                        )

                        if perplexity_response.success:
                            perplexity_insights = perplexity_response.content
                            responses["perplexity"] = perplexity_response
                            print(colored("   📰 Perplexity news analysis: received", "green"))
                        else:
                            print(colored(f"   ❌ Perplexity: {perplexity_response.error}", "red"))
                    except Exception as e:
                        print(colored(f"   ❌ Perplexity: {e}", "red"))
                else:
                    print(colored("   ⚠️  Perplexity model not available", "yellow"))

                # Step 2: Query other fast models with Perplexity insights
                fast_models = ["deepseek"]  # Other fast models

                enhanced_prompt = prompt
                if perplexity_insights:
                    enhanced_prompt += f"\n\n📰 PERPLEXITY NEWS ANALYSIS:\n{perplexity_insights}\n\n"

                for model_key in fast_models:
                    if model_key in self.swarm.models:
                        try:
                            response = await self.swarm.models[model_key].generate_response(
                                system_prompt="You are a probability estimation expert. Use the news analysis provided to make an informed probability estimate.",
                                user_content=enhanced_prompt,
                                temperature=0.2,  # Lower temperature for consistency
                                max_tokens=600,
                                timeout=25  # Shorter timeout
                            )
                            if response.success:
                                responses[model_key] = response
                                print(colored(f"   ✅ {model_key}: response received", "green"))
                            else:
                                print(colored(f"   ❌ {model_key}: {response.error}", "red"))
                        except Exception as e:
                            print(colored(f"   ❌ {model_key}: {e}", "red"))
                    else:
                        print(colored(f"   ⚠️  {model_key}: model not available", "yellow"))
            else:
                # Normal mode: query all models
                swarm_result = await self.swarm.query(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=800,
                    timeout=45
                )
                responses = swarm_result.get("responses", {})

            # Parse responses
            model_votes = {}
            probabilities = []

            for provider, response_data in responses.items():
                if isinstance(response_data, dict) and response_data.get("success"):
                    content = response_data.get("content", "")
                elif hasattr(response_data, 'content'):
                    content = response_data.content
                else:
                    continue

                prob = self._extract_probability(content)
                if prob is not None:
                    model_votes[provider] = prob
                    probabilities.append(prob)

            if not probabilities:
                print(colored("❌ No valid probabilities extracted from breaking news analysis", "red"))
                return None

            # Calculate consensus with fast mode adjustments
            ai_probability = mean(probabilities)
            confidence = self._calculate_confidence(probabilities)

            # Lower confidence threshold for fast mode
            min_confidence = 0.4 if fast_mode else 0.6
            if confidence < min_confidence:
                print(colored(f"❌ Confidence too low: {confidence:.1%} < {min_confidence:.1%}", "red"))
                return None

            # Determine direction and edge
            market_prob = market.yes_price
            edge = ai_probability - market_prob

            # Require minimum edge for breaking news trades
            min_edge = 0.08 if fast_mode else self.min_edge
            if abs(edge) < min_edge:
                print(colored(f"❌ Edge too small: {abs(edge):.1%} < {min_edge:.1%}", "red"))
                return None

            direction = "YES" if ai_probability > market_prob else "NO"
            expected_value = self._calculate_expected_value(edge, confidence, market_prob)

            # Create signal with news context
            signal = TradingSignal(
                market=market,
                direction=direction,
                confidence=confidence,
                probability=ai_probability,
                market_probability=market_prob,
                edge=edge,
                expected_value=expected_value,
                reasoning=f"BREAKING NEWS: {news.headline[:100]}... Impact: {news_impact:.1%}",
                model_votes=model_votes,
                model_weights={k: 1.0 for k in model_votes.keys()},  # Equal weights for fast mode
                consensus_summary=f"Breaking news consensus: {ai_probability:.1%} probability",
                news_context=f"BREAKING: {news.headline} - {news.source}",
                timestamp=datetime.now(),
                news_items=[news],
                news_impact_score=news_impact,
                contrarian_signals=[],  # No contrarian analysis for breaking news fast mode
                is_breaking_news_trade=True,
                news_to_trade_latency_seconds=time.time() - start_time
            )

            print(colored("🚀 BREAKING NEWS TRADE SIGNAL GENERATED!", "green", attrs=["bold"]))
            print(colored(f"   Direction: {direction} | Edge: {edge:+.1%} | Confidence: {confidence:.1%}", "green"))
            print(colored(f"   News-to-Trade Latency: {signal.news_to_trade_latency_seconds:.1f}s", "green"))

            # Record breaking news prediction for accuracy tracking
            self.accuracy_tracker.record_prediction(signal)

            return signal

        except Exception as e:
            logger.error(f"Breaking news analysis failed: {e}")
            print(colored(f"❌ Breaking news analysis failed: {e}", "red"))
            return None

    def _build_analysis_prompt(self, market: Market) -> str:
        """Build analysis prompt for AI models."""
        prompt = f"""
Analyze this prediction market and provide your probability estimate:

Market: {market.question}

Description: {market.description or 'No description available'}

Category: {market.category}

Current Market Data:
- YES price: {market.yes_price:.1%}
- NO price: {market.no_price:.1%}
- Volume: ${market.volume:,.0f}
- Liquidity: ${market.liquidity:,.0f}

{f'End Date: {market.end_date}' if market.end_date else ''}

Please provide:
1. Your probability estimate for the YES outcome (between 0% and 100%)
2. Brief reasoning for your estimate
3. Key factors influencing your analysis

Format your probability as "XX%" or "XX percent" in your response.
"""
        return prompt.strip()

    def _extract_probability(self, text: str) -> Optional[float]:
        """
        Extract probability from text using regex patterns.

        Args:
            text: Text to search for probability

        Returns:
            Probability as float (0.0 to 1.0) or None if not found
        """
        # Common patterns for probability extraction
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # "75%" or "75.5%"
            r'(\d+(?:\.\d+)?)\s*percent',  # "75 percent"
            r'probability(?:\s+(?:of|is|estimate))?\s*:?\s*(\d+(?:\.\d+)?)%?',  # "probability: 75%"
            r'(\d+(?:\.\d+)?)%\s+(?:chance|likelihood|probability)',  # "75% chance"
        ]

        text_lower = text.lower()

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Take the first valid probability found
                for match in matches:
                    try:
                        prob = float(match)
                        if 0 <= prob <= 100:
                            return prob / 100.0  # Convert to 0-1 range
                    except ValueError:
                        continue

        return None

    def _calculate_confidence(self, predictions: List[float]) -> float:
        """
        Calculate confidence based on model agreement.

        Args:
            predictions: List of probability predictions

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if len(predictions) <= 1:
            return 0.5  # Low confidence with single prediction

        # Calculate standard deviation
        try:
            std = stdev(predictions)
            mean_prob = mean(predictions)

            # Confidence decreases with higher standard deviation
            # Perfect agreement (std=0) = 1.0 confidence
            # High disagreement (std approaches 0.5) = 0.0 confidence
            confidence = max(0.0, 1.0 - (std / 0.5))

            return confidence

        except Exception:
            return 0.5  # Default confidence on error

    def _get_news_context(self, market: Market) -> str:
        """Get real-time news context from Perplexity (has web search)."""
        if "perplexity" not in self.swarm.models:
            return "No news context available (Perplexity not configured)"

        try:
            # Create news query for Perplexity
            news_prompt = f"""
What are the latest news and developments related to: "{market.question}"?

Provide a brief summary of recent relevant news, current status, and any factors that might affect the outcome.
Focus on factual information from the last 30 days.
Keep it concise but informative.
"""

            # Query only Perplexity for news (it has web search)
            from src.models import model_factory
            perplexity = model_factory.get_model("perplexity", "sonar-pro")

            response = perplexity.generate_response(
                system_prompt="You are a news research assistant with access to current information. Provide factual, recent news summaries.",
                user_content=news_prompt,
                temperature=0.3,
                max_tokens=512
            )

            if response and response.content:
                return response.content.strip()
            else:
                return "News context unavailable"

        except Exception as e:
            logger.warning(f"News context retrieval failed: {e}")
            return "News context retrieval failed"

    def _build_enhanced_prompt(self, market: Market, news_context: str) -> str:
        """Build enhanced analysis prompt with current date/time and news context."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        prompt = f"""
Current Date/Time: {current_time}

MARKET ANALYSIS REQUEST:

Question: {market.question}

Market Details:
- Category: {market.category}
- Current YES Price: {market.yes_price:.1%}
- Current NO Price: {market.no_price:.1%}
- 24h Volume: ${market.volume:,.0f}
- Liquidity: ${market.liquidity:,.0f}

{f'End Date: {market.end_date}' if market.end_date else 'No end date specified'}

RECENT NEWS & CONTEXT:
{news_context}

ANALYSIS REQUIREMENTS:

Please provide your probability estimate for the YES outcome occurring.

Also provide:
1. Your confidence level in this estimate (0-100%)
2. Key factors FOR the YES outcome
3. Key factors AGAINST the YES outcome
4. How recent news/developments impact your estimate

Format your probability as "XX%" or "XX percent" and confidence as "XX% confident".
"""
        return prompt.strip()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for market analysis."""
        return """You are an expert quantitative analyst specializing in prediction markets and probabilistic forecasting.

Your task is to analyze prediction market questions and provide well-reasoned probability estimates based on available data, logic, and current context.

Guidelines:
- Consider all available information objectively
- Factor in recency and reliability of information
- Account for market efficiency and current pricing
- Provide clear reasoning for your estimates
- Express confidence levels based on information quality and agreement

Be precise, analytical, and evidence-based in your assessments."""

    def _calculate_expected_value(self, edge: float, confidence: float, market_prob: float) -> float:
        """Calculate expected value considering fees and position sizing."""
        if edge <= 0:
            return 0.0

        # For simplicity, use edge * confidence as normalized EV
        # In a full implementation, this would include:
        # - Kelly criterion calculations
        # - Fee adjustments (maker/taker fees)
        # - Position size optimization
        # - Risk-adjusted returns

        return edge * confidence

    def _build_reasoning(self, model_votes: Dict[str, float], confidence: float, expected_value: float) -> str:
        """Build comprehensive reasoning string."""
        model_count = len(model_votes)
        avg_probability = sum(model_votes.values()) / model_count

        reasoning = f"""AI Swarm Analysis: {model_count} models
• Average Probability: {avg_probability:.1%}
• Model Agreement: {confidence:.1%}
• Expected Value: {expected_value:.3f}
• Models: {', '.join(f'{k}({v:.1%})' for k, v in model_votes.items())}"""

        return reasoning

    async def find_opportunities(
        self,
        limit: int = 20,
        min_edge: float = 0.08,
        min_confidence: float = 0.6,
        min_volume: float = 50000,
        min_liquidity: float = 25000.0
    ) -> List[TradingSignal]:
        """
        Smart market scanning with advanced filtering and expected value analysis.

        Filters:
        - Minimum liquidity and volume (configurable)
        - Markets ending in > min_market_hours
        - Prioritizes high-volume, good-spread markets

        Args:
            limit: Maximum markets to analyze
            min_edge: Minimum edge threshold (0.08 = 8%)
            min_confidence: Minimum confidence threshold (0.6 = 60%)
            min_volume: Minimum 24h volume
            min_liquidity: Minimum liquidity

        Returns:
            List of trading signals with positive expected value
        """
        print(colored(f"\n🔍 Smart market scanning for opportunities...", "blue", attrs=["bold"]))
        print(colored(f"Filters: vol≥${min_volume:,.0f}, liq≥${min_liquidity:,.0f}, edge≥{min_edge:.1%}, conf≥{min_confidence:.1%}", "white"))

        try:
            # Get active markets (skip in paper trading if no API keys)
            if self.paper_trading and not os.getenv("POLYGON_WALLET_PRIVATE_KEY"):
                logger.info("📝 Paper trading mode: Skipping real market data fetch (no API keys)")
                print(colored("📝 Paper trading: Using mock market data for testing", "yellow"))
                # Return empty list for paper trading without API keys
                return []

            markets = await polymarket.get_markets(limit=limit * 3, min_volume=min_volume, min_liquidity=min_liquidity, active_only=True)  # Get more to filter

            # Smart market filtering
            filtered_markets = []
            total_filtered = {'volume': 0, 'liquidity': 0, 'expiry': 0, 'spread': 0}

            for market in markets:
                # Basic filters with logging
                if market.volume < min_volume:
                    total_filtered['volume'] += 1
                    logger.debug(f"Market {market.question[:30]}...: vol=${market.volume:,.0f} < ${min_volume:,.0f} (volume filter)")
                    continue
                if market.liquidity < min_liquidity:
                    total_filtered['liquidity'] += 1
                    logger.debug(f"Market {market.question[:30]}...: liq=${market.liquidity:,.0f} < ${min_liquidity:,.0f} (liquidity filter)")
                    continue

                # Time-to-expiry filter
                if not market.end_date:
                    continue

                # Ensure end_date is a datetime object
                end_date = market.end_date
                if isinstance(end_date, str):
                    try:
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        logger.debug(f"Could not parse end_date for market {market.condition_id}: {end_date}")
                        continue

                # Handle timezone-aware vs timezone-naive datetime comparison
                now = datetime.now()
                if end_date.tzinfo is not None and now.tzinfo is None:
                    # end_date is timezone-aware, make now timezone-aware
                    from datetime import timezone
                    now = now.replace(tzinfo=timezone.utc)
                elif end_date.tzinfo is None and now.tzinfo is not None:
                    # now is timezone-aware, make end_date timezone-aware
                    from datetime import timezone
                    end_date = end_date.replace(tzinfo=timezone.utc)

                hours_until_expiry = (end_date - now).total_seconds() / 3600
                if hours_until_expiry <= 0:
                    total_filtered['expiry'] += 1
                    logger.debug(f"Market {market.question[:30]}...: EXPIRED {abs(hours_until_expiry):.1f}h ago (expired filter)")
                    continue  # Market has already expired
                elif hours_until_expiry < self.min_market_hours:
                    total_filtered['expiry'] += 1
                    logger.debug(f"Market {market.question[:30]}...: {hours_until_expiry:.1f}h < {self.min_market_hours}h (expiry filter)")
                    continue  # Too close to expiry (risky)

                # Spread filter (avoid illiquid markets)
                spread = abs(market.yes_price - (1 - market.no_price))
                if spread > 0.1:  # Skip markets with wide spreads
                    total_filtered['spread'] += 1
                    logger.debug(f"Market {market.question[:30]}...: spread={spread:.3f} > 0.1 (spread filter)")
                    continue

                filtered_markets.append((market, hours_until_expiry))

            # Sort by priority: volume * liquidity * (1/spread)
            filtered_markets.sort(
                key=lambda x: x[0].volume * x[0].liquidity / max(spread, 0.01),
                reverse=True
            )

            markets_only = [m[0] for m in filtered_markets[:limit]]

            # Log filtering summary
            logger.info(f"Market filtering summary: {len(markets)} total -> {len(markets_only)} passed")
            logger.info(f"Filtered out: volume={total_filtered['volume']}, liquidity={total_filtered['liquidity']}, expiry={total_filtered['expiry']}, spread={total_filtered['spread']}")

            print(colored(f"📊 Filtered to {len(markets_only)} high-quality markets", "white"))

            # Analyze each market with enhanced pipeline
            opportunities = []
            for i, market in enumerate(markets_only, 1):
                print(colored(f"\n[{i}/{len(markets_only)}] ", "white"), end="")

                signal = await self.analyze_market(market)
                if signal and signal.expected_value >= self.expected_value_threshold and signal.edge >= min_edge and signal.confidence >= min_confidence:
                    opportunities.append(signal)
                    print(colored(f"✅ EV: {signal.expected_value:.1%} | Edge: {signal.edge:+.1%} | Conf: {signal.confidence:.1%}", "green", attrs=["bold"]))
                elif signal and (signal.edge < min_edge or signal.confidence < min_confidence):
                    print(colored(f"⏭️  Below thresholds: Edge {signal.edge:+.1%} (min {min_edge:.1%}) | Conf {signal.confidence:.1%} (min {min_confidence:.1%})", "yellow"))
                elif signal and signal.expected_value < self.expected_value_threshold:
                    print(colored(f"⏭️  EV too low: {signal.expected_value:.3f} (min {self.expected_value_threshold:.1%})", "yellow"))
                else:
                    print(colored("❌ Analysis failed", "red"))

            print(colored(f"\n🎯 Found {len(opportunities)} high-EV trading opportunities", "green", attrs=["bold"]))
            return opportunities

        except Exception as e:
            logger.error(f"Market scanning failed: {e}")
            print(colored(f"❌ Scanning failed: {e}", "red"))
            return []

    async def execute_signal(self, signal: TradingSignal) -> Dict:
        """
        Execute a trading signal with comprehensive timing and risk analysis.

        Args:
            signal: Trading signal to execute

        Returns:
            Dict with execution status and details
        """
        try:
            # ===== FINAL SAFETY GATES (never trade below hard rules) =====
            market_volume = float(getattr(signal.market, "volume", 0) or 0)
            market_liquidity = float(getattr(signal.market, "liquidity", 0) or 0)
            if abs(signal.edge) < self.min_edge:
                return {
                    "status": "skipped",
                    "reason": f"Edge {signal.edge:.1%} < min edge {self.min_edge:.1%}",
                }
            if market_volume < self.min_volume:
                return {
                    "status": "skipped",
                    "reason": f"Volume ${market_volume:,.0f} < min volume ${self.min_volume:,.0f}",
                }
            if market_liquidity < self.min_liquidity:
                return {
                    "status": "skipped",
                    "reason": f"Liquidity ${market_liquidity:,.0f} < min liquidity ${self.min_liquidity:,.0f}",
                }

            # ===== TIMING ANALYSIS =====
            timing_signal = await entry_timing_optimizer.analyze_timing(
                signal.market,
                signal.direction,
                self.max_position  # Use max position as size estimate
            )

            if not timing_signal.should_trade_now:
                logger.info(f"⏳ Delaying trade: {timing_signal.wait_reason}")
                return {
                    "status": "delayed",
                    "reason": timing_signal.wait_reason,
                    "suggested_wait_minutes": timing_signal.suggested_wait_minutes,
                    "timing_score": timing_signal.overall_timing_score
                }

            # ===== RISK MANAGEMENT =====
            # Calculate Kelly-optimal position size with advanced risk factors
            kelly_size = self._calculate_kelly_position(signal)

            if kelly_size <= 0:
                logger.warning(f"Kelly size too small (${kelly_size:.2f}) - skipping trade")
                return {"status": "skipped", "reason": "Kelly size too small"}

            # Apply timing size multiplier
            kelly_size *= timing_signal.size_multiplier

            # Cap at maximum position size
            kelly_size = min(kelly_size, self.max_position)

            # Determine token and price
            if signal.direction == "YES":
                token_id = signal.market.yes_token_id
                price = signal.market.yes_price
            else:
                token_id = signal.market.no_token_id
                price = signal.market.no_price

            # Calculate shares (accounting for fees)
            # Size includes fees, so shares = (size / (1 + fee_rate)) / price
            effective_size = kelly_size / (1 + self.fee_rate)
            shares = effective_size / price

            logger.info("💰 Executing {} trade (Kelly-sized, timing-adjusted)", signal.direction)
            logger.info("📊 Market: {} (vol=${}, liq=${})", signal.market.question[:45], market_volume, market_liquidity)
            logger.info("🎲 Probability: {:.1%} | Market: {:.1%} | Timing: {:.1%}",
                       signal.probability, signal.market_probability, timing_signal.overall_timing_score)
            logger.info("💎 EV: {:.1%} | Confidence: {:.1%}", signal.expected_value, signal.confidence)
            logger.info("💵 Size: ${:.0f} (${:.0f} after timing adjustment)",
                       kelly_size / timing_signal.size_multiplier, kelly_size)
            logger.info("📊 Shares: {:.0f} @ {:.1%} (fee-adjusted)", shares, price)

            if self.paper_trading:
                logger.info("📝 PAPER TRADE - No real transaction")
                self.trades_executed += 1
                self.position_log.append({
                    'signal': signal,
                    'size': kelly_size,
                    'shares': shares,
                    'direction': signal.direction,
                    'timing_score': timing_signal.overall_timing_score,
                    'timestamp': datetime.now()
                })
                # Add to position manager for active management
                await self.position_manager.add_position(signal, {
                    "status": "executed",
                    "mode": "paper",
                    "size": kelly_size,
                    "shares": shares,
                    "direction": signal.direction,
                    "timing_score": timing_signal.overall_timing_score
                })

                return {
                    "status": "executed",
                    "mode": "paper",
                    "size": kelly_size,
                    "shares": shares,
                    "direction": signal.direction,
                    "timing_score": timing_signal.overall_timing_score
                }

            # Execute real trade using the new polymarket client
            response = await polymarket.place_order_for_market(
                market=signal.market,
                direction=signal.direction,
                size_usd=kelly_size,  # Use dollar amount, not shares
                order_type="limit"    # Use limit order for better control
            )

            if response and "orderID" in response:
                logger.info("✅ Trade executed: {}", response['orderID'])
                self.trades_executed += 1
                self.position_log.append({
                    'signal': signal,
                    'size': kelly_size,
                    'shares': shares,
                    'order_id': response['orderID'],
                    'direction': signal.direction,
                    'timing_score': timing_signal.overall_timing_score,
                    'timestamp': datetime.now()
                })
                # Add to position manager for active management
                await self.position_manager.add_position(signal, {
                    "status": "executed",
                    "mode": "live",
                    "order_id": response['orderID'],
                    "size": kelly_size,
                    "shares": shares,
                    "direction": signal.direction,
                    "timing_score": timing_signal.overall_timing_score
                })

                return {
                    "status": "executed",
                    "mode": "live",
                    "order_id": response['orderID'],
                    "size": kelly_size,
                    "shares": shares,
                    "direction": signal.direction,
                    "timing_score": timing_signal.overall_timing_score
                }
            else:
                logger.error("❌ Trade execution failed: {}", response)
                return {"status": "failed", "reason": "API error", "response": response}

        except Exception as e:
            logger.error("Trade execution failed: {}", e)
            return {"status": "error", "reason": str(e)}

    def _calculate_kelly_position(self, signal: TradingSignal) -> float:
        """
        Calculate Kelly criterion optimal position size.

        Kelly formula: f = (bp - q) / b
        Where:
        - b = odds received (decimal)
        - p = probability of winning
        - q = probability of losing (1-p)

        Simplified for prediction markets.
        """
        try:
            # For prediction markets, the "odds" are implicit
            # Use expected value and confidence to determine position size

            if signal.expected_value <= 0:
                return 0.0

            # Use advanced risk management for position sizing
            # This includes Kelly with uncertainty, correlation adjustments, drawdown protection
            bankroll = getattr(self, 'current_bankroll', self.max_position)
            position_size = self.risk_manager.calculate_position_size_advanced(
                signal, signal.market, self.risk_manager.state.positions
            )

            # Ensure minimum and maximum bounds
            position_size = max(position_size, 10.0)  # Minimum $10 trade
            position_size = min(position_size, self.max_position)

            return position_size

        except Exception as e:
            logger.warning(f"Kelly calculation failed: {e}")
            return 0.0


def run_trading_bot(scan_interval_minutes: int = 30) -> None:
    """Synchronous wrapper for the async trading bot."""
    import asyncio
    asyncio.run(run_trading_bot_async(scan_interval_minutes))

async def run_trading_bot_async(scan_interval_minutes: int = 30):
    """
    Run the advanced trading bot with daily P&L tracking and risk management.

    Features:
    - Daily loss limits with auto-pause
    - Comprehensive logging to file
    - P&L tracking and reporting
    - Smart position sizing with Kelly criterion
    - Enhanced market filtering

    Args:
        scan_interval_minutes: How often to scan markets (minutes)
    """
    print(colored("🚀 Starting Advanced Polymarket AI Trading Bot", "cyan", attrs=["bold"]))

    # Setup logging to file
    log_filename = f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logger.add(log_filename, rotation="1 day", retention="7 days")

    try:
        # Initialize trading swarm
        swarm = TradingSwarm()

        # Create scheduler
        scheduler = BlockingScheduler()

        async         def scan_and_trade():
            """Advanced scheduled task with P&L tracking and risk management."""
            # Create event loop for async operations
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                current_time = datetime.now()
                print(colored(f"\n⏰ Market scan started at {current_time}", "blue"))

                # Check daily loss limit
                if swarm.daily_pnl <= -swarm.daily_loss_limit:
                    print(colored(f"🚨 Daily loss limit (${swarm.daily_loss_limit:.0f}) reached. Pausing trading.", "red", attrs=["bold"]))
                    logger.warning(f"Daily loss limit reached: ${swarm.daily_pnl:.2f}")
                    return

                # Reset daily P&L if new day
                if current_time.date() != swarm.daily_start_time:
                    swarm.daily_start_time = current_time.date()
                    swarm.daily_pnl = 0.0
                    print(colored("📅 New trading day started - P&L reset", "green"))

                # Find opportunities with smart filtering
                opportunities = loop.run_until_complete(swarm.find_opportunities(limit=15))  # Analyze top 15 markets

                # Sort by expected value and execute top opportunities
                opportunities.sort(key=lambda x: x.expected_value, reverse=True)
                executed = 0

                for signal in opportunities[:3]:  # Execute top 3 by EV
                    result = loop.run_until_complete(swarm.execute_signal(signal))
                    if result["status"] == "executed":
                        executed += 1

                        # Log trade details
                        logger.info("Trade executed: {}... | {} | Size: ${:.0f} | EV: {:.3f}",
                                  signal.market.question[:50], signal.direction,
                                  result["size"], signal.expected_value)

                # Log session summary
                session_pnl = sum(pos.get('size', 0) for pos in swarm.position_log[-executed:]) if swarm.paper_trading else 0
                swarm.daily_pnl += session_pnl

                print(colored(f"📈 Session complete: {len(opportunities)} opportunities, {executed} trades", "green"))
                print(colored(f"💰 Daily P&L: ${swarm.daily_pnl:.2f} | Signals generated: {swarm.signals_generated}", "white"))

                # Log comprehensive session data
                logger.info(f"Session complete - Opportunities: {len(opportunities)}, "
                          f"Trades: {executed}, Daily P&L: ${swarm.daily_pnl:.2f}, "
                          f"Total signals: {swarm.signals_generated}")

            except Exception as e:
                logger.error(f"Scheduled task failed: {e}")
                print(colored(f"❌ Scheduled task failed: {e}", "red"))
            finally:
                loop.close()

        # Schedule market scans
        scheduler.add_job(
            scan_and_trade,
            'interval',
            minutes=scan_interval_minutes,
            id='market_scan',
            max_instances=1  # Prevent overlapping scans
        )

        # Run initial scan
        scan_and_trade()

        print(colored(f"📅 Scheduled market scans every {scan_interval_minutes} minutes. Press Ctrl+C to stop.", "white"))
        print(colored(f"📊 Logs saved to: {log_filename}", "white"))
        print(colored(f"🚨 Daily loss limit: ${swarm.daily_loss_limit:.0f}", "yellow"))

        scheduler.start()

    except KeyboardInterrupt:
        # Final P&L report
        final_pnl = sum(pos.get('size', 0) for pos in swarm.position_log) if swarm.paper_trading else 0
        print(colored(f"\n👋 Trading bot stopped by user", "yellow"))
        print(colored(f"📊 Final session: {swarm.signals_generated} signals, {swarm.trades_executed} trades", "white"))
        print(colored(f"💰 Total P&L: ${final_pnl:.2f}", "green" if final_pnl >= 0 else "red"))
        logger.info(f"Bot stopped - Signals: {swarm.signals_generated}, Trades: {swarm.trades_executed}, P&L: ${final_pnl:.2f}")

    except Exception as e:
        logger.error(f"Trading bot failed: {e}")
        print(colored(f"❌ Trading bot failed: {e}", "red"))
        raise

    finally:
        logger.remove(file_handler)
