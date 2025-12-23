"""
Contrarian Detector - Find when the crowd is wrong

This module identifies contrarian trading opportunities by detecting when:
- Social sentiment diverges from market prices
- Sharp money (professional bettors) differs from public money
- Markets overreact to news without fundamental justification
- Consensus is too extreme (likely missing tail risks)

All signals are for Polymarket trading only.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from collections import defaultdict

from loguru import logger

from src.connectors import Market

try:
    from src.services.news_monitor import news_aggregator
    from src.services.external_odds import odds_aggregator
    from src.connectors import Market
except ImportError:
    # For testing
    class NewsAggregator:
        async def get_breaking_news(self, categories): return []

    class ConsensusDetector:
        async def get_external_consensus(self, market):
            return {"sources": {}, "weighted_average": None, "sources_count": 0}


    news_aggregator = NewsAggregator()
    consensus_detector = ConsensusDetector()


@dataclass
class ContrarianSignal:
    """Signal indicating the crowd might be wrong."""
    market: Any  # Polymarket market
    signal_type: str  # "sentiment_divergence", "sharp_vs_public", "overreaction", "consensus_trap"

    current_price: float
    suggested_direction: str  # "YES" or "NO"

    contrarian_score: float  # 0-1, higher = stronger contrarian signal

    evidence: List[str]  # Reasons for contrarian view
    risks: List[str]  # Why the crowd might be right

    confidence: float  # 0-1 overall confidence in the signal
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentAnalysis:
    """Social sentiment analysis result."""
    probability: float  # 0-1 sentiment-derived probability
    confidence: float  # 0-1 confidence in sentiment analysis
    sample_size: int  # Number of posts analyzed
    sources: List[str]  # Twitter, Reddit, etc.


class ContrarianDetector:
    """
    Detect contrarian trading opportunities.

    Finds when the crowd is likely wrong by analyzing:
    - Social sentiment vs market prices
    - Sharp money vs public money divergence
    - Overreactions to minor news
    - Extreme consensus (likely missing risks)
    """

    def __init__(self, news_aggregator: Optional[Any] = None, consensus_detector: Optional[Any] = None) -> None:
        """Initialize contrarian detector."""
        self.news = news_aggregator or globals().get('news_aggregator')
        self.consensus = consensus_detector or globals().get('consensus_detector')

        logger.info("🔄 Contrarian detector initialized")

    async def detect_sentiment_divergence(self, market: Market) -> Optional[ContrarianSignal]:
        """
        Find when social sentiment diverges significantly from market price.

        When Twitter/Reddit sentiment strongly disagrees with market pricing,
        it could indicate:
        - Market knows something social media doesn't (follow market)
        - Social media knows something market hasn't priced (contrarian opportunity)
        """
        try:
            sentiment = await self._analyze_social_sentiment(market)

            if not sentiment or sentiment.confidence < 0.6:
                return None

            divergence = abs(sentiment.probability - market.yes_price)

            if divergence > 0.20:  # 20%+ divergence threshold
                suggested_direction = "YES" if sentiment.probability > market.yes_price else "NO"

                return ContrarianSignal(
                    market=market,
                    signal_type="sentiment_divergence",
                    current_price=market.yes_price,
                    suggested_direction=suggested_direction,
                    contrarian_score=min(divergence * 2, 1.0),  # Scale divergence to score
                    evidence=[
                        f"Social sentiment: {sentiment.probability*100:.0f}% ({sentiment.sources})",
                        f"Market price: {market.yes_price*100:.0f}%",
                        f"Divergence: {divergence*100:.0f}%",
                        f"Sample size: {sentiment.sample_size} posts"
                    ],
                    risks=[
                        "Social media can be echo chamber",
                        "Market may have insider information",
                        "Sentiment analysis may be inaccurate"
                    ],
                    confidence=min(sentiment.confidence, 0.6)  # Conservative confidence
                )

        except Exception as e:
            logger.debug(f"Sentiment divergence detection failed: {e}")

        return None

    async def detect_sharp_money_divergence(self, market: Market) -> Optional[ContrarianSignal]:
        """
        Detect when sharp money (Betfair) differs from public money (PredictIt).

        Professional bettors (sharp money) are historically more accurate than
        retail bettors (public money). When they disagree, follow the sharp money.
        """
        try:
            external = await self.consensus.get_external_consensus(market)

            if external.sources_count < 2:
                return None

            # Look for sharp vs public divergence
            sharp_sources = ["betfair"]  # Professional betting
            public_sources = ["predictit", "kalshi"]  # Retail platforms

            sharp_probs = []
            public_probs = []

            for source, prob in external.probabilities.items():
                if source in sharp_sources:
                    sharp_probs.append(prob)
                elif source in public_sources:
                    public_probs.append(prob)

            if not sharp_probs or not public_probs:
                return None

            sharp_avg = sum(sharp_probs) / len(sharp_probs)
            public_avg = sum(public_probs) / len(public_probs)

            divergence = sharp_avg - public_avg

            if abs(divergence) > 0.08:  # 8%+ difference threshold
                suggested_direction = "YES" if divergence > 0 else "NO"

                return ContrarianSignal(
                    market=market,
                    signal_type="sharp_vs_public",
                    current_price=market.yes_price,
                    suggested_direction=suggested_direction,
                    contrarian_score=min(abs(divergence) * 3, 1.0),  # Amplify divergence
                    evidence=[
                        f"Sharp money: {sharp_avg*100:.0f}% ({len(sharp_probs)} sources)",
                        f"Public money: {public_avg*100:.0f}% ({len(public_probs)} sources)",
                        f"Divergence: {divergence*100:+.0f}%",
                        "Sharp bettors historically more accurate"
                    ],
                    risks=[
                        "Sharp money isn't always right",
                        "Liquidity differences between platforms",
                        "Sharp money can be wrong in manipulated markets"
                    ],
                    confidence=0.7  # Higher confidence for sharp vs public
                )

        except Exception as e:
            logger.debug(f"Sharp money divergence detection failed: {e}")

        return None

    async def detect_overreaction(self, market: Market, price_history: Optional[List] = None) -> Optional[ContrarianSignal]:
        """
        Detect when market overreacted to news without fundamental justification.

        Pattern:
        - Large price move (15%+) in short time
        - No major news to justify the move
        - Historically, such moves often revert

        Strategy: Fade the overreaction (bet against the move).
        """
        try:
            # For now, simulate price history analysis
            # In real implementation, would use historical price data
            if not price_history:
                # Mock analysis - check for large recent moves
                # This would be replaced with actual price history analysis
                recent_volatility = await self._analyze_recent_volatility(market)
            else:
                # Calculate actual price movement
                if len(price_history) < 5:
                    return None

                recent_prices = [getattr(p, 'yes_price', 0.5) for p in price_history[-5:]]
                price_change = recent_prices[-1] - recent_prices[0]
                recent_volatility = abs(price_change)

            if recent_volatility > 0.15:  # 15%+ move threshold
                # Check if there's major news to justify the move
                news = await self.news.get_breaking_news([market.category])
                has_major_news = any(
                    getattr(n, 'relevance_score', 0) > 0.9
                    for n in news
                )

                if not has_major_news:
                    suggested_direction = "NO" if recent_volatility > 0 else "YES"  # Fade the move

                    return ContrarianSignal(
                        market=market,
                        signal_type="overreaction",
                        current_price=market.yes_price,
                        suggested_direction=suggested_direction,
                        contrarian_score=min(recent_volatility * 2, 1.0),
                        evidence=[
                            f"Recent volatility: {recent_volatility*100:.0f}% move",
                            "No major news to justify price movement",
                            "Likely overreaction to minor news/hype",
                            "Historical pattern: large moves without news often revert"
                        ],
                        risks=[
                            "Move might be informed trading we don't see",
                            "Major news might be developing",
                            "Could be start of larger trend"
                        ],
                        confidence=0.5  # Medium confidence - overreactions are risky
                    )

        except Exception as e:
            logger.debug(f"Overreaction detection failed: {e}")

        return None

    async def detect_consensus_trap(self, market: Market) -> Optional[ContrarianSignal]:
        """
        Detect when consensus is too extreme (likely missing tail risks).

        "When everyone agrees, someone is usually wrong."

        Markets priced at 88%+ or 12%- are "sure things" but often
        underestimate black swan risks or hidden information.
        """
        try:
            price = market.yes_price

            # Check for extreme consensus
            if price > 0.88 or price < 0.12:
                is_extreme_yes = price > 0.88
                suggested_direction = "NO" if is_extreme_yes else "YES"

                # Calculate how extreme it is (further from 50% = more extreme)
                extremity = abs(price - 0.5) * 2  # 0-1 scale

                return ContrarianSignal(
                    market=market,
                    signal_type="consensus_trap",
                    current_price=market.yes_price,
                    suggested_direction=suggested_direction,
                    contrarian_score=min(extremity * 0.5, 0.6),  # Capped at 0.6 (conservative)
                    evidence=[
                        f"Market at {price*100:.0f}% - extreme consensus",
                        f"Extremity score: {extremity:.1%}",
                        "Extreme prices often underestimate tail risks",
                        "Small edge but asymmetric risk/reward"
                    ],
                    risks=[
                        "Consensus is usually right at extremes",
                        "Edge is small if market is correct",
                        "May be correct - don't fight the trend"
                    ],
                    confidence=0.4  # Low confidence - these are high-risk signals
                )

        except Exception as e:
            logger.debug(f"Consensus trap detection failed: {e}")

        return None

    async def scan_all_contrarian(
        self,
        markets: List[Market],
        signal_types: Optional[List[str]] = None
    ) -> List[ContrarianSignal]:
        """
        Scan all markets for contrarian opportunities.

        Args:
            markets: List of Polymarket markets to analyze
            signal_types: Optional filter for specific signal types
        """
        try:
            signals = []
            detector_methods = [
                ("sentiment_divergence", self.detect_sentiment_divergence),
                ("sharp_vs_public", self.detect_sharp_money_divergence),
                ("overreaction", self.detect_overreaction),
                ("consensus_trap", self.detect_consensus_trap),
            ]

            # Filter to requested signal types
            if signal_types:
                detector_methods = [
                    (name, method) for name, method in detector_methods
                    if name in signal_types
                ]

            for market in markets:
                for signal_name, detector_method in detector_methods:
                    try:
                        signal = await detector_method(market)
                        if signal:
                            signals.append(signal)
                            logger.debug(f"Contrarian signal found: {signal.signal_type} for {market.question[:50]}...")
                    except Exception as e:
                        logger.debug(f"Failed to check {signal_name} for market {market.question[:30]}...: {e}")

            # Sort by contrarian score (highest first)
            signals.sort(key=lambda s: s.contrarian_score, reverse=True)

            total_volume = sum(getattr(s.market, 'volume', 0) for s in signals)
            logger.info(f"🔄 Found {len(signals)} contrarian signals across {len(markets)} markets (Total Volume: ${total_volume:,.0f})")

            return signals

        except Exception as e:
            logger.error(f"Contrarian scanning failed: {e}")
            return []

    async def _analyze_social_sentiment(self, market: Market) -> Optional[SentimentAnalysis]:
        """
        Analyze social media sentiment for a market.

        This is a placeholder implementation. In production, would:
        - Query Twitter API for recent tweets about the market
        - Analyze Reddit posts in relevant subreddits
        - Use NLP to extract sentiment and probability estimates
        - Aggregate across multiple sources
        """
        try:
            # Placeholder implementation
            # In real implementation, would integrate with Twitter/Reddit APIs

            # Mock sentiment analysis based on market question
            question_lower = market.question.lower()

            # Simple keyword-based sentiment simulation
            positive_keywords = ['win', 'success', 'yes', 'victory', 'beat']
            negative_keywords = ['lose', 'fail', 'no', 'defeat', 'crash']

            positive_count = sum(1 for word in positive_keywords if word in question_lower)
            negative_count = sum(1 for word in negative_keywords if word in question_lower)

            if positive_count > negative_count:
                sentiment_prob = 0.7
            elif negative_count > positive_count:
                sentiment_prob = 0.3
            else:
                sentiment_prob = 0.5

            return SentimentAnalysis(
                probability=sentiment_prob,
                confidence=0.6,  # Placeholder confidence
                sample_size=50,  # Mock sample size
                sources=["twitter", "reddit"]
            )

        except Exception as e:
            logger.debug(f"Social sentiment analysis failed: {e}")
            return None

    async def _analyze_recent_volatility(self, market: Market) -> float:
        """
        Analyze recent price volatility for overreaction detection.

        This is a placeholder. In production, would analyze actual price history.
        """
        try:
            # Placeholder: simulate some volatility analysis
            # In real implementation, would query price history database
            import random
            return random.uniform(0.05, 0.25)  # Random volatility for testing

        except Exception as e:
            logger.debug(f"Volatility analysis failed: {e}")
            return 0.0


# Global instance
contrarian_detector = ContrarianDetector()
