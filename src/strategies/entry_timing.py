"""
Entry Timing Optimization for Polymarket Trading

Analyzes optimal entry times based on spread, momentum, time-of-day,
volatility, and market conditions to maximize trade execution quality.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pytz

from loguru import logger


@dataclass
class TimingSignal:
    """Analysis of optimal entry timing for a trade."""
    should_trade_now: bool
    wait_reason: Optional[str] = None
    suggested_wait_minutes: int = 0

    # Individual scores (0-1, higher = better to trade now)
    liquidity_score: float = 0.0
    spread_score: float = 0.0
    momentum_score: float = 0.0
    time_of_day_score: float = 0.0
    volatility_score: float = 0.0
    expiry_score: float = 0.0
    order_book_score: float = 0.0

    overall_timing_score: float = 0.0

    size_multiplier: float = 1.0  # Reduce size if timing not ideal

    # Analysis details
    analysis_details: Dict[str, str] = field(default_factory=dict)


class EntryTimingOptimizer:
    """
    Optimizes trade entry timing based on multiple market factors.

    Analyzes spread, momentum, time-of-day, volatility, and market conditions
    to determine optimal entry times and position sizes.
    """

    # Optimal trading hours (EST - Eastern Standard Time)
    OPTIMAL_HOURS = (9, 21)  # 9 AM - 9 PM EST (best liquidity)

    # Time zones
    EST = pytz.timezone('US/Eastern')
    UTC = pytz.timezone('UTC')

    def __init__(self) -> None:
        """Initialize timing optimizer."""
        self.price_cache: Dict[str, List[Tuple[datetime, float]]] = {}
        self.last_analysis: Dict[str, TimingSignal] = {}

        logger.info("⏰ Entry timing optimizer initialized")

    async def analyze_timing(self, market: Any, direction: str, size: float = 100) -> TimingSignal:
        """
        Comprehensive timing analysis for trade entry.

        Args:
            market: Market object with price data
            direction: "YES" or "NO"
            size: Intended trade size in USD

        Returns:
            TimingSignal with recommendation and scores
        """
        signal = TimingSignal(should_trade_now=True)

        try:
            # Analyze individual factors
            spread_score, spread_reason = self._check_spread(market)
            momentum_score, momentum_reason = self._check_momentum(market, direction)
            time_score, time_reason = self._check_time_of_day()
            volatility_score, volatility_reason = self._check_volatility(market)
            expiry_score, expiry_reason = self._check_time_to_expiry(market)
            order_book_score, order_book_reason = self._check_order_book(market, direction, size)

            # Update signal with individual scores
            signal.spread_score = spread_score
            signal.momentum_score = momentum_score
            signal.time_of_day_score = time_score
            signal.volatility_score = volatility_score
            signal.expiry_score = expiry_score
            signal.order_book_score = order_book_score

            # Store analysis details
            signal.analysis_details = {
                'spread': spread_reason,
                'momentum': momentum_reason,
                'time_of_day': time_reason,
                'volatility': volatility_reason,
                'expiry': expiry_reason,
                'order_book': order_book_reason
            }

            # Calculate overall score (weighted average)
            weights = {
                'spread': 0.2,
                'momentum': 0.25,
                'time_of_day': 0.15,
                'volatility': 0.15,
                'expiry': 0.15,
                'order_book': 0.1
            }

            overall_score = (
                spread_score * weights['spread'] +
                momentum_score * weights['momentum'] +
                time_score * weights['time_of_day'] +
                volatility_score * weights['volatility'] +
                expiry_score * weights['expiry'] +
                order_book_score * weights['order_book']
            )

            signal.overall_timing_score = overall_score

            # Determine if we should trade now
            if expiry_score < 0.3:
                # Too close to expiry - reject
                signal.should_trade_now = False
                signal.wait_reason = f"Too close to expiry: {expiry_reason}"
            elif overall_score >= 0.7:
                # Good timing - proceed
                signal.should_trade_now = True
                signal.size_multiplier = 1.0
            elif overall_score >= 0.5:
                # Acceptable timing - reduce size slightly
                signal.should_trade_now = True
                signal.size_multiplier = 0.8
            elif overall_score >= 0.3:
                # Poor timing - reduce size significantly
                signal.should_trade_now = True
                signal.size_multiplier = 0.5
            else:
                # Very poor timing - wait
                signal.should_trade_now = False
                signal.wait_reason = f"Poor timing (score: {overall_score:.2f})"
                signal.suggested_wait_minutes = 60  # Wait 1 hour

            # Cache analysis
            market_id = getattr(market, 'condition_id', getattr(market, 'slug', 'unknown'))
            self.last_analysis[market_id] = signal

            # Log analysis
            self._log_timing_analysis(market, signal)

        except Exception as e:
            logger.error(f"Timing analysis failed: {e}")
            # Default to proceed with caution
            signal.should_trade_now = True
            signal.size_multiplier = 0.7
            signal.wait_reason = f"Analysis failed: {e}"

        return signal

    def _check_spread(self, market: Any) -> Tuple[float, str]:
        """
        Analyze bid-ask spread quality.

        Better spreads = lower slippage = better entry timing.
        """
        try:
            yes_price = getattr(market, 'yes_price', 0.5)
            no_price = getattr(market, 'no_price', 0.5)

            # Calculate spread as percentage
            spread_pct = abs(yes_price + no_price - 1.0) * 100

            if spread_pct < 2.0:
                return 1.0, f"{spread_pct:.1f}% spread - excellent"
            elif spread_pct < 4.0:
                return 0.7, f"{spread_pct:.1f}% spread - acceptable"
            elif spread_pct < 6.0:
                return 0.4, f"{spread_pct:.1f}% spread - wide, reduce size"
            else:
                return 0.1, f"{spread_pct:.1f}% spread - very wide"
        except Exception:
            return 0.5, "Could not calculate spread"

    def _check_momentum(self, market: Any, direction: str) -> Tuple[float, str]:
        """
        Analyze recent price momentum.

        Don't chase price movements - buy weakness, sell strength.
        """
        try:
            market_id = getattr(market, 'condition_id', getattr(market, 'slug', 'unknown'))
            current_price = getattr(market, 'yes_price', 0.5) if direction == "YES" else getattr(market, 'no_price', 0.5)

            # Get recent prices (would need price history in real implementation)
            # For now, simulate based on market data
            recent_prices = self.price_cache.get(market_id, [])

            if len(recent_prices) < 2:
                return 0.7, "Insufficient price history for momentum analysis"

            # Calculate 1-hour price change
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_prices_filtered = [(ts, price) for ts, price in recent_prices if ts > one_hour_ago]

            if len(recent_prices_filtered) < 2:
                return 0.7, "Not enough recent price data"

            old_price = recent_prices_filtered[0][1]
            price_change_pct = (current_price - old_price) / old_price * 100

            # For buying: negative momentum is good (price falling = good entry)
            # For selling: positive momentum is good (price rising = good entry)
            if direction == "YES":
                # Buying YES: want price to be falling
                if price_change_pct < -3:
                    return 0.9, f"{price_change_pct:+.1f}% recent change - strong downtrend, good entry"
                elif price_change_pct < -1:
                    return 0.7, f"{price_change_pct:+.1f}% recent change - moderate pullback, decent entry"
                elif price_change_pct > 3:
                    return 0.3, f"{price_change_pct:+.1f}% recent change - uptrend, wait for pullback"
                else:
                    return 0.6, f"{price_change_pct:+.1f}% recent change - stable, neutral"
            else:  # Buying NO
                # Buying NO: want price to be rising
                if price_change_pct > 3:
                    return 0.9, f"{price_change_pct:+.1f}% recent change - strong uptrend, good entry"
                elif price_change_pct > 1:
                    return 0.7, f"{price_change_pct:+.1f}% recent change - moderate rise, decent entry"
                elif price_change_pct < -3:
                    return 0.3, f"{price_change_pct:+.1f}% recent change - downtrend, wait for bounce"
                else:
                    return 0.6, f"{price_change_pct:+.1f}% recent change - stable, neutral"
        except Exception as e:
            return 0.5, f"Momentum analysis failed: {e}"

    def _check_time_of_day(self) -> Tuple[float, str]:
        """
        Analyze current time of day for liquidity.

        Markets are most liquid during business hours.
        """
        try:
            now_est = datetime.now(self.EST)
            current_hour = now_est.hour

            if 9 <= current_hour < 21:  # 9 AM - 9 PM EST
                return 1.0, f"{current_hour}:00 EST - peak liquidity hours"
            elif 6 <= current_hour < 9:  # 6 AM - 9 AM EST (opening)
                return 0.7, f"{current_hour}:00 EST - market opening, moderate liquidity"
            elif 21 <= current_hour < 24:  # 9 PM - 12 AM EST
                return 0.5, f"{current_hour}:00 EST - evening, lower liquidity"
            else:  # 12 AM - 6 AM EST
                return 0.2, f"{current_hour}:00 EST - overnight, very low liquidity"
        except Exception as e:
            return 0.5, f"Time analysis failed: {e}"

    def _check_volatility(self, market: Any) -> Tuple[float, str]:
        """
        Analyze recent market volatility.

        High volatility = higher risk = wait for stabilization.
        """
        try:
            # Simple volatility proxy: distance from 0.5
            yes_price = getattr(market, 'yes_price', 0.5)
            volatility_proxy = abs(yes_price - 0.5) * 2  # Scale to 0-1

            if volatility_proxy < 0.1:
                return 0.8, f"Low volatility ({volatility_proxy:.2f}) - stable, safe entry"
            elif volatility_proxy < 0.2:
                return 1.0, f"Normal volatility ({volatility_proxy:.2f}) - good"
            elif volatility_proxy < 0.3:
                return 0.6, f"High volatility ({volatility_proxy:.2f}) - wait for stabilization"
            else:
                return 0.3, f"Very high volatility ({volatility_proxy:.2f}) - avoid"
        except Exception:
            return 0.5, "Could not assess volatility"

    def _check_time_to_expiry(self, market: Any) -> Tuple[float, str]:
        """
        Analyze time until market resolution.

        Too close = high risk, too far = capital inefficiency.
        """
        try:
            end_date = getattr(market, 'end_date', None)
            if not end_date:
                return 0.5, "No expiry date available"

            # Convert to datetime if needed
            if isinstance(end_date, str):
                try:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except ValueError as e:
                    logger.debug(f"Could not parse expiry date in entry timing: {e}")
                    return 0.5, "Could not parse expiry date"

            now = datetime.now(self.UTC)
            hours_until_expiry = (end_date - now).total_seconds() / 3600

            if hours_until_expiry < 24:
                return 0.1, f"{hours_until_expiry:.0f} hours until expiry - too risky, avoid"
            elif hours_until_expiry < 168:  # 7 days
                return 0.7, f"{hours_until_expiry:.0f} hours until expiry - short-term, need strong conviction"
            elif hours_until_expiry < 720:  # 30 days
                return 1.0, f"{hours_until_expiry:.0f} hours until expiry - sweet spot"
            elif hours_until_expiry < 2160:  # 90 days
                return 0.8, f"{hours_until_expiry:.0f} hours until expiry - longer term, capital tied up"
            else:
                return 0.5, f"{hours_until_expiry:.0f} hours until expiry - very long term, opportunity cost"
        except Exception as e:
            return 0.5, f"Expiry analysis failed: {e}"

    def _check_order_book(self, market: Any, direction: str, size: float) -> Tuple[float, str]:
        """
        Analyze order book depth for slippage risk.

        Large orders need to be split to avoid moving the market.
        """
        try:
            # Estimate order book depth from market data
            volume = getattr(market, 'volume', 0)
            liquidity = getattr(market, 'liquidity', volume * 0.1)  # Rough estimate

            if liquidity <= 0:
                return 0.5, "No liquidity data available"

            # Calculate what percentage of available liquidity our order represents
            liquidity_ratio = size / liquidity

            if liquidity_ratio < 0.1:  # < 10% of liquidity
                return 1.0, f"Small order ({liquidity_ratio:.1%} of liquidity) - no slippage"
            elif liquidity_ratio < 0.25:  # 10-25%
                return 0.8, f"Medium order ({liquidity_ratio:.1%} of liquidity) - minor slippage"
            elif liquidity_ratio < 0.5:   # 25-50%
                return 0.5, f"Large order ({liquidity_ratio:.1%} of liquidity) - noticeable slippage, reduce size"
            else:  # > 50%
                return 0.2, f"Very large order ({liquidity_ratio:.1%} of liquidity) - major slippage, split order"
        except Exception as e:
            return 0.5, f"Order book analysis failed: {e}"

    def get_optimal_entry_time(self, market: Any) -> datetime:
        """
        Suggest optimal entry time in next 24 hours.

        Considers expected liquidity patterns and market hours.
        """
        try:
            now_est = datetime.now(self.EST)

            # Find next optimal trading hour
            optimal_start = now_est.replace(hour=self.OPTIMAL_HOURS[0], minute=0, second=0, microsecond=0)
            optimal_end = now_est.replace(hour=self.OPTIMAL_HOURS[1], minute=0, second=0, microsecond=0)

            # If we're already in optimal hours, return current time
            if optimal_start <= now_est <= optimal_end:
                return now_est

            # If before optimal hours, return start of optimal period
            if now_est < optimal_start:
                return optimal_start

            # If after optimal hours, return start of optimal period tomorrow
            tomorrow = now_est + timedelta(days=1)
            return tomorrow.replace(hour=self.OPTIMAL_HOURS[0], minute=0, second=0, microsecond=0)

        except Exception as e:
            logger.warning(f"Could not determine optimal entry time: {e}")
            return datetime.now(self.UTC)

    def should_split_order(self, market: Any, size: float) -> List[Tuple[float, int]]:
        """
        Determine if large order should be split to minimize slippage.

        Returns list of (size, delay_minutes) tuples.
        """
        try:
            # Check order book impact
            _, order_book_reason = self._check_order_book(market, "YES", size)  # Direction doesn't matter for size check

            # If significant impact, split the order
            liquidity = getattr(market, 'liquidity', getattr(market, 'volume', 0) * 0.1)
            if liquidity <= 0:
                return [(size, 0)]  # Can't assess, don't split

            impact_ratio = size / liquidity

            if impact_ratio < 0.25:
                # Small order, no need to split
                return [(size, 0)]
            elif impact_ratio < 0.5:
                # Medium order, split in 2
                half_size = size / 2
                return [(half_size, 0), (half_size, 30)]  # 30 minute delay
            else:
                # Large order, split in 3
                third_size = size / 3
                return [
                    (third_size, 0),
                    (third_size, 45),
                    (third_size, 90)
                ]

        except Exception as e:
            logger.warning(f"Order splitting analysis failed: {e}")
            return [(size, 0)]  # Default: don't split

    def _log_timing_analysis(self, market: Any, signal: TimingSignal):
        """Log detailed timing analysis."""
        market_name = getattr(market, 'question', 'Unknown Market')[:50]

        logger.info("⏰ [TIMING] Market: \"{}\"", market_name)
        logger.info("   Spread: {:.1f}% (score: {:.1f}) - {}",
                   getattr(market, 'yes_price', 0.5) * 100, signal.spread_score,
                   signal.analysis_details.get('spread', 'N/A'))
        logger.info("   Momentum: {} (score: {:.1f}) - {}",
                   signal.analysis_details.get('momentum', 'N/A'), signal.momentum_score,
                   signal.analysis_details.get('momentum', 'N/A'))
        logger.info("   Time: {} (score: {:.1f}) - {}",
                   signal.analysis_details.get('time_of_day', 'N/A'), signal.time_of_day_score,
                   signal.analysis_details.get('time_of_day', 'N/A'))
        logger.info("   Volatility: {} (score: {:.1f}) - {}",
                   signal.analysis_details.get('volatility', 'N/A'), signal.volatility_score,
                   signal.analysis_details.get('volatility', 'N/A'))
        logger.info("   Expiry: {} (score: {:.1f}) - {}",
                   signal.analysis_details.get('expiry', 'N/A'), signal.expiry_score,
                   signal.analysis_details.get('expiry', 'N/A'))
        logger.info("   Order Book: {} (score: {:.1f}) - {}",
                   signal.analysis_details.get('order_book', 'N/A'), signal.order_book_score,
                   signal.analysis_details.get('order_book', 'N/A'))

        action = "✅ PROCEED" if signal.should_trade_now else "⏳ WAIT"
        logger.info("   Overall: {:.2f} - {} with {:.0f}% size",
                   signal.overall_timing_score, action, signal.size_multiplier * 100)

        if not signal.should_trade_now and signal.wait_reason:
            logger.info("   Reason: {}", signal.wait_reason)

    def update_price_cache(self, market_id: str, price: float):
        """Update price cache for momentum calculations."""
        if market_id not in self.price_cache:
            self.price_cache[market_id] = []

        self.price_cache[market_id].append((datetime.now(self.UTC), price))

        # Keep only last 24 hours of data
        cutoff = datetime.now(self.UTC) - timedelta(hours=24)
        self.price_cache[market_id] = [
            (ts, p) for ts, p in self.price_cache[market_id] if ts > cutoff
        ]


# Global instance for easy access
entry_timing_optimizer = EntryTimingOptimizer()
