"""
Advanced Risk Management for Polymarket AI Trading Bot

Implements sophisticated risk controls including Kelly Criterion with uncertainty,
correlation-adjusted sizing, drawdown protection, Sharpe ratio tracking,
and comprehensive bankroll management.

Author: Polymarket Trading Bot
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import math
import numpy as np

from loguru import logger


@dataclass
class RiskLimits:
    """Comprehensive risk management parameters."""
    max_daily_loss: float = 200.0  # Maximum daily loss in USD
    max_position_size: float = 100.0  # Maximum size per position in USD
    max_positions: int = 10  # Maximum concurrent positions
    max_exposure: float = 500.0  # Maximum total exposure in USD
    min_edge: float = 0.08  # 8% minimum edge required
    min_confidence: float = 0.6  # Minimum confidence score (0-1)
    min_expected_value: float = 5.0  # $5 minimum expected value
    max_correlation: float = 0.7  # Maximum allowed correlation (0-1)
    cooldown_after_loss: int = 300  # 5 minutes cooldown after loss (seconds)
    kelly_fraction: float = 0.25  # Conservative Kelly fraction (0.25 = 1/4 Kelly)
    drawdown_threshold_1: float = 0.10  # 10% drawdown triggers size reduction
    drawdown_threshold_2: float = 0.20  # 20% drawdown triggers trading halt
    weekend_reduction: float = 0.5  # 50% size reduction on weekends
    min_market_hours: int = 24  # Minimum hours until market expiry

    # Volume/Liquidity requirements
    min_market_volume: float = 50000       # $50k minimum
    min_market_liquidity: float = 1.0       # $1.00 minimum (realistic for Polymarket)
    max_order_pct_of_liquidity: float = 0.02  # 2% max


@dataclass
class LiquidityRequirements:
    """Minimum liquidity/volume requirements to avoid slippage"""

    # Minimum market requirements
    min_volume_24h: float = 50000          # $50k minimum 24h volume
    min_liquidity: float = 1.0             # $1.00 minimum liquidity (realistic for Polymarket)
    min_volume_lifetime: float = 100000    # $100k lifetime volume

    # Order size limits (relative to market)
    max_order_pct_of_liquidity: float = 0.02   # Max 2% of liquidity per order
    max_order_pct_of_volume: float = 0.01      # Max 1% of 24h volume per order

    # Absolute limits
    max_order_size: float = 100            # Never more than $100 per order
    min_order_size: float = 5              # Never less than $5 per order

    # Spread limits
    max_spread_pct: float = 0.05           # Max 5% bid-ask spread

    # Slippage tolerance
    max_slippage_pct: float = 0.02         # Max 2% slippage allowed

    # Market depth requirements
    min_orderbook_depth: float = 1000      # Min $1000 on each side of book


@dataclass
class SlippageEstimate:
    """Estimated slippage for an order"""
    estimated_slippage_pct: float
    estimated_slippage_usd: float
    effective_price: float
    is_acceptable: bool
    reason: str
    recommended_size: float  # Size that would have acceptable slippage


@dataclass
class Position:
    """Represents an active trading position."""
    market_id: str
    direction: str  # "YES" or "NO"
    size: float  # Position size in USD
    entry_price: float  # Entry price (probability)
    timestamp: datetime
    correlation_group: str = ""  # Group for correlation tracking
    expected_value: float = 0.0
    market_question: str = ""  # For correlation analysis
    category: str = ""  # Market category


@dataclass
class RiskState:
    """Current risk management state."""
    daily_pnl: float = 0.0
    peak_portfolio_value: float = 0.0
    current_drawdown: float = 0.0
    positions: List[Position] = field(default_factory=list)
    last_loss_time: Optional[datetime] = None
    total_exposure: float = 0.0
    daily_start_time: datetime = field(default_factory=lambda: datetime.now().date())
    trading_paused: bool = False
    pause_reason: str = ""


class DrawdownProtector:
    """Manages portfolio drawdown protection."""

    def __init__(self) -> None:
        self.peak_value = 0.0
        self.current_value = 0.0
        self.drawdown_history: List[Tuple[datetime, float]] = []

    def update(self, portfolio_value: float, timestamp: Optional[datetime] = None) -> None:
        """Update with current portfolio value."""
        if timestamp is None:
            timestamp = datetime.now()

        self.current_value = portfolio_value

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Record drawdown history
        current_dd = self.current_drawdown
        self.drawdown_history.append((timestamp, current_dd))

        # Keep only last 30 days
        cutoff = timestamp - timedelta(days=30)
        self.drawdown_history = [
            (ts, dd) for ts, dd in self.drawdown_history if ts > cutoff
        ]

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as percentage."""
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - self.current_value) / self.peak_value

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown in history."""
        if not self.drawdown_history:
            return 0.0
        return max(dd for _, dd in self.drawdown_history)

    def get_size_multiplier(self) -> float:
        """
        Reduce position sizes during drawdown.

        Drawdown < 10%: Full size (1.0)
        Drawdown 10-15%: 75% size
        Drawdown 15-20%: 50% size
        Drawdown > 20%: 25% size or pause
        """
        dd = self.current_drawdown

        if dd < 0.10:
            return 1.0
        elif dd < 0.15:
            return 0.75
        elif dd < 0.20:
            return 0.50
        else:
            return 0.25

    def should_pause_trading(self) -> bool:
        """Return True if should stop trading due to drawdown."""
        return self.current_drawdown > 0.25  # 25% drawdown = pause


class PerformanceTracker:
    """Tracks trading performance metrics."""

    def __init__(self) -> None:
        self.daily_returns: List[float] = []
        self.daily_timestamps: List[datetime] = []
        self.trade_returns: List[float] = []

    def record_daily_return(self, return_pct: float, timestamp: Optional[datetime] = None) -> None:
        """Record daily return percentage."""
        if timestamp is None:
            timestamp = datetime.now()

        self.daily_returns.append(return_pct)
        self.daily_timestamps.append(timestamp)

        # Keep only last 365 days
        if len(self.daily_returns) > 365:
            self.daily_returns = self.daily_returns[-365:]
            self.daily_timestamps = self.daily_timestamps[-365:]

    def record_trade(self, trade_return: float) -> None:
        """Record individual trade return."""
        self.trade_returns.append(trade_return)

    def get_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (avg_return - risk_free) / std_dev
        Annualized assuming 365 trading days.
        """
        if len(self.daily_returns) < 7:
            return 0.0

        avg_return = np.mean(self.daily_returns) * 365
        std_return = np.std(self.daily_returns) * np.sqrt(365)

        if std_return == 0:
            return 0.0

        return (avg_return - risk_free_rate) / std_return

    def get_sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Sortino ratio - like Sharpe but only penalizes downside volatility.
        """
        if len(self.daily_returns) < 7:
            return 0.0

        avg_return = np.mean(self.daily_returns) * 365
        downside_returns = np.array([r for r in self.daily_returns if r < 0])

        if len(downside_returns) == 0:
            return float('inf')  # No downside!

        downside_std = np.std(downside_returns) * np.sqrt(365)

        return (avg_return - risk_free_rate) / downside_std

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from daily returns."""
        if len(self.daily_returns) < 2:
            return 0.0

        cumulative = np.cumsum(self.daily_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / np.maximum(peak, 0.01)  # Avoid div by zero

        return float(np.max(drawdown))

    def get_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if not self.trade_returns:
            return 0.0
        return float(np.mean(np.array(self.trade_returns) > 0))

    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.trade_returns:
            return 0.0

        profits = [r for r in self.trade_returns if r > 0]
        losses = [abs(r) for r in self.trade_returns if r < 0]

        if not profits or not losses:
            return 0.0

        return sum(profits) / sum(losses)

    def generate_performance_report(self) -> Dict[str, float]:
        """Generate comprehensive performance report."""
        if not self.daily_returns:
            return {"error": "No performance data available"}

        returns = np.array(self.daily_returns)

        return {
            "total_return": float(np.sum(returns)),
            "avg_daily_return": float(np.mean(returns)),
            "std_dev": float(np.std(returns)),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "max_drawdown": self.get_max_drawdown(),
            "win_rate": self.get_win_rate(),
            "profit_factor": self.get_profit_factor(),
            "best_day": float(np.max(returns)) if len(returns) > 0 else 0.0,
            "worst_day": float(np.min(returns)) if len(returns) > 0 else 0.0,
            "days_tracked": len(returns),
            "total_trades": len(self.trade_returns),
        }


class BankrollManager:
    """Manages bankroll and position sizing rules."""

    def __init__(self, initial_bankroll: float) -> None:
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.daily_start_bankroll = initial_bankroll

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update current bankroll."""
        self.current_bankroll = new_bankroll

    def reset_daily(self) -> None:
        """Reset for new trading day."""
        self.daily_start_bankroll = self.current_bankroll

    def get_risk_per_trade(self) -> float:
        """
        Maximum risk per trade based on bankroll.

        Rules:
        - Never risk more than 5% of current bankroll per trade
        - Reduce to 2% if in drawdown
        - Scale with bankroll growth
        """
        base_risk = 0.05  # 5%

        # Adjust for drawdown from initial bankroll
        drawdown = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        if drawdown > 0.10:
            base_risk = 0.02  # Reduce to 2%
        elif drawdown > 0.20:
            base_risk = 0.01  # Reduce to 1%

        return self.current_bankroll * base_risk

    def get_max_total_risk(self) -> float:
        """
        Maximum total risk across all positions.

        Never have more than 30% of bankroll at risk.
        """
        return self.current_bankroll * 0.30

    def can_afford_trade(self, size: float) -> bool:
        """Check if we can afford this trade."""
        return size <= self.get_risk_per_trade()

    def get_bankroll_health(self) -> Dict[str, float]:
        """Get bankroll health metrics."""
        return {
            "current_bankroll": self.current_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "total_return": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll,
            "daily_risk_limit": self.get_risk_per_trade(),
            "max_total_risk": self.get_max_total_risk(),
            "drawdown": (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll,
        }


class SlippageProtector:
    """
    Protects against slippage by:
    1. Requiring minimum volume/liquidity
    2. Limiting order size relative to market
    3. Estimating slippage before trading
    4. Splitting large orders
    """

    def __init__(self, requirements: Optional[LiquidityRequirements] = None) -> None:
        self.req = requirements or LiquidityRequirements()
        logger.info(f"🛡️ SlippageProtector initialized")
        logger.info(f"   Min Volume: ${self.req.min_volume_24h:,.0f}")
        logger.info(f"   Min Liquidity: ${self.req.min_liquidity:,.0f}")
        logger.info(f"   Max Order % of Liquidity: {self.req.max_order_pct_of_liquidity*100:.1f}%")

    def check_market_liquidity(self, market: Any) -> Tuple[bool, str]:
        """
        Check if market has sufficient liquidity for trading.

        Returns (is_ok, reason)
        """
        volume = getattr(market, 'volume', 0) or 0
        liquidity = getattr(market, 'liquidity', 0) or 0

        # Check 24h volume
        if volume < self.req.min_volume_24h:
            return False, f"Volume ${volume:,.0f} < ${self.req.min_volume_24h:,.0f} minimum"

        # Check liquidity
        if liquidity < self.req.min_liquidity:
            return False, f"Liquidity ${liquidity:,.0f} < ${self.req.min_liquidity:,.0f} minimum"

        # Check spread if available
        yes_price = getattr(market, 'yes_price', 0.5)
        no_price = getattr(market, 'no_price', 0.5)
        spread = abs((1 - yes_price - no_price))

        if spread > self.req.max_spread_pct:
            return False, f"Spread {spread*100:.1f}% > {self.req.max_spread_pct*100:.1f}% maximum"

        return True, "OK"

    def calculate_max_order_size(self, market) -> float:
        """
        Calculate maximum safe order size for a market.

        Uses the most conservative of:
        1. Absolute max ($100)
        2. % of liquidity (2%)
        3. % of 24h volume (1%)
        """
        volume = getattr(market, 'volume', 0) or 0
        liquidity = getattr(market, 'liquidity', 0) or 0

        # Calculate limits
        limit_by_liquidity = liquidity * self.req.max_order_pct_of_liquidity
        limit_by_volume = volume * self.req.max_order_pct_of_volume
        limit_absolute = self.req.max_order_size

        # Take the minimum (most conservative)
        max_size = min(limit_by_liquidity, limit_by_volume, limit_absolute)

        # Ensure at least min_order_size if market is liquid enough
        if max_size < self.req.min_order_size:
            return 0  # Market too illiquid

        return max_size

    def estimate_slippage(
        self,
        market,
        order_size: float,
        direction: str
    ) -> SlippageEstimate:
        """
        Estimate slippage for a given order.

        Simple model: slippage increases with order size relative to liquidity.
        """
        liquidity = getattr(market, 'liquidity', 0) or 1
        volume = getattr(market, 'volume', 0) or 1

        # Order size as % of liquidity (avoid division by zero)
        size_pct = order_size / liquidity if liquidity > 0 else 1.0

        # Estimate slippage (simplified model)
        # Real slippage would need order book depth
        # This assumes roughly: 0.5% slippage per 1% of liquidity
        estimated_slippage_pct = size_pct * 0.5

        # Get current price
        if direction == "YES":
            current_price = getattr(market, 'yes_price', 0.5)
        else:
            current_price = getattr(market, 'no_price', 0.5)

        # Calculate effective price after slippage
        effective_price = current_price * (1 + estimated_slippage_pct)
        estimated_slippage_usd = order_size * estimated_slippage_pct

        # Is it acceptable?
        is_acceptable = estimated_slippage_pct <= self.req.max_slippage_pct

        if is_acceptable:
            reason = f"Slippage {estimated_slippage_pct*100:.2f}% is acceptable"
        else:
            reason = f"Slippage {estimated_slippage_pct*100:.2f}% exceeds {self.req.max_slippage_pct*100:.1f}% limit"

        # Calculate recommended size for acceptable slippage
        if not is_acceptable:
            # Work backwards: what size gives max acceptable slippage?
            recommended_size = liquidity * self.req.max_slippage_pct * 2  # Inverse of slippage model
            recommended_size = min(recommended_size, self.req.max_order_size)
        else:
            recommended_size = order_size

        return SlippageEstimate(
            estimated_slippage_pct=estimated_slippage_pct,
            estimated_slippage_usd=estimated_slippage_usd,
            effective_price=effective_price,
            is_acceptable=is_acceptable,
            reason=reason,
            recommended_size=recommended_size
        )

    def get_safe_order_size(
        self,
        market,
        desired_size: float,
        direction: str
    ) -> Tuple[float, str]:
        """
        Get a safe order size that won't cause excessive slippage.

        Returns (safe_size, reason)
        """
        # First check if market is liquid enough
        is_ok, reason = self.check_market_liquidity(market)
        if not is_ok:
            return 0, reason

        # Get max order size for this market
        max_size = self.calculate_max_order_size(market)

        if max_size == 0:
            return 0, "Market too illiquid for any order"

        # Cap at max size
        safe_size = min(desired_size, max_size)

        # Estimate slippage at this size
        slippage = self.estimate_slippage(market, safe_size, direction)

        if not slippage.is_acceptable:
            safe_size = slippage.recommended_size

        # Final bounds check
        if safe_size < self.req.min_order_size:
            return 0, f"Order ${safe_size:.2f} below ${self.req.min_order_size:.2f} minimum"

        if safe_size < desired_size:
            reason = f"Reduced from ${desired_size:.2f} to ${safe_size:.2f} for slippage protection"
        else:
            reason = "OK"

        return safe_size, reason

    def should_split_order(
        self,
        market,
        total_size: float
    ) -> List[Tuple[float, int]]:
        """
        If order is too large, split into multiple smaller orders.

        Returns list of (size, delay_seconds) tuples.
        """
        max_size = self.calculate_max_order_size(market)

        if max_size == 0 or total_size <= max_size:
            return [(total_size, 0)]  # Single order, no delay

        # Split into chunks
        chunks = []
        remaining = total_size
        delay = 0

        while remaining > 0:
            chunk_size = min(remaining, max_size)
            if chunk_size < self.req.min_order_size:
                break
            chunks.append((chunk_size, delay))
            remaining -= chunk_size
            delay += 60  # 60 second delay between chunks

        return chunks


class RiskManager:
    """Advanced risk management system with multiple protection layers."""

    def __init__(self, limits: Optional[RiskLimits] = None, initial_bankroll: float = 1000.0, liquidity_req: Optional[LiquidityRequirements] = None) -> None:
        """Initialize risk manager with configurable limits."""
        self.limits = limits or RiskLimits()
        self.state = RiskState()

        # Advanced risk management components
        self.drawdown_protector = DrawdownProtector()
        self.performance_tracker = PerformanceTracker()
        self.bankroll_manager = BankrollManager(initial_bankroll)
        self.slippage = SlippageProtector(liquidity_req or LiquidityRequirements())

        # Correlation tracking
        self.market_correlations: Dict[str, List[str]] = self._load_market_correlations()
        self.correlation_cache: Dict[Tuple[str, str], float] = {}

        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Any] = {}

        logger.info("🎛️  Advanced Risk Manager initialized")
        logger.info(f"📊 Daily loss limit: ${self.limits.max_daily_loss}")
        logger.info(f"💰 Max position: ${self.limits.max_position_size}")
        logger.info(f"🎯 Min edge: {self.limits.min_edge:.1%}")
        logger.info(f"🛡️  Kelly fraction: {self.limits.kelly_fraction:.1%}")
        logger.info(f"💵 Initial bankroll: ${initial_bankroll}")

    def can_trade(self, signal: Any, market: Any) -> Tuple[bool, str]:
        """
        Comprehensive risk check before allowing a trade.

        Args:
            signal: TradingSignal object
            market: Market object

        Returns:
            (can_trade: bool, reason: str)
        """
        # Reset daily stats if new day
        if datetime.now().date() != self.state.daily_start_time:
            self._reset_daily_stats()

        # Check if trading is paused
        if self.state.trading_paused:
            return False, f"Trading paused: {self.state.pause_reason}"

        # Check daily loss limit
        if self.state.daily_pnl <= -self.limits.max_daily_loss:
            self.state.trading_paused = True
            self.state.pause_reason = f"Daily loss limit (${self.limits.max_daily_loss}) exceeded"
            logger.warning(f"🚨 Trading halted: {self.state.pause_reason}")
            return False, self.state.pause_reason

        # Check cooldown after loss
        if self.state.last_loss_time:
            cooldown_remaining = (datetime.now() - self.state.last_loss_time).seconds
            if cooldown_remaining < self.limits.cooldown_after_loss:
                return False, f"Cooldown active: {self.limits.cooldown_after_loss - cooldown_remaining}s remaining"

        # Check maximum positions
        if len(self.state.positions) >= self.limits.max_positions:
            return False, f"Maximum positions ({self.limits.max_positions}) reached"

        # Check signal quality
        if signal.edge < self.limits.min_edge:
            return False, f"Edge too low: {signal.edge:.1%} < {self.limits.min_edge:.1%}"

        if signal.confidence < self.limits.min_confidence:
            return False, f"Confidence too low: {signal.confidence:.1%} < {self.limits.min_confidence:.1%}"

        if signal.expected_value < self.limits.min_expected_value:
            return False, f"Expected value too low: ${signal.expected_value:.2f} < ${self.limits.min_expected_value:.2f}"

        # Check market expiry time
        if hasattr(market, 'end_date') and market.end_date:
            try:
                end_date = datetime.fromisoformat(market.end_date.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                hours_until_expiry = (end_date - now).total_seconds() / 3600
                if hours_until_expiry < self.limits.min_market_hours:
                    return False, f"Market expires too soon: {hours_until_expiry:.1f}h < {self.limits.min_market_hours}h"
            except Exception as e:
                logger.warning(f"Could not parse end_date '{market.end_date}': {e}")
                # Skip expiry check if parsing fails

        # Check correlation with existing positions
        correlation_issues = self._check_correlation_issues(signal, market)
        if correlation_issues:
            return False, correlation_issues

        # Check drawdown impact
        drawdown_factor = self._calculate_drawdown_factor()
        if drawdown_factor <= 0:
            self.state.trading_paused = True
            self.state.pause_reason = f"Drawdown too severe: {self.state.current_drawdown:.1%}"
            return False, self.state.pause_reason

        # Check market liquidity for slippage protection
        is_liquid, reason = self.slippage.check_market_liquidity(market)
        if not is_liquid:
            logger.warning(f"🚫 Market rejected: {reason}")
            return False, reason

        market_volume = getattr(market, 'volume', 0)
        market_liquidity = getattr(market, 'liquidity', 0)
        logger.info(f"✅ Trade approved for {market.question[:40]}...: edge={signal.edge:.1%}, conf={signal.confidence:.1%}, vol=${market_volume:,.0f}, liq=${market_liquidity:,.0f}")
        return True, "Trade approved"

    def check_trade_risk(self, signal: Any, market: Any) -> Tuple[bool, str, float]:
        """
        Full risk check including slippage protection.

        Returns (can_trade, reason, adjusted_size)
        """
        # First check all standard risk rules
        can_trade, reason = self.can_trade(signal, market)
        if not can_trade:
            return False, reason, 0

        # Calculate position size
        desired_size = self.calculate_position_size(signal, self.bankroll_manager.current_bankroll)

        # Apply slippage protection
        safe_size, slippage_reason = self.slippage.get_safe_order_size(market, desired_size, signal.direction)
        if safe_size == 0:
            return False, slippage_reason, 0

        # Estimate and log slippage
        slippage = self.slippage.estimate_slippage(market, safe_size, signal.direction)
        logger.info(f"📊 Slippage estimate: {slippage.estimated_slippage_pct*100:.2f}% (${slippage.estimated_slippage_usd:.2f})")

        if safe_size < desired_size:
            logger.info(f"📉 Size adjusted: ${desired_size:.2f} → ${safe_size:.2f} for slippage protection")

        return True, "OK", safe_size

    def calculate_position_size(self, signal: Any, bankroll: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion with risk adjustments.

        Args:
            signal: TradingSignal object
            bankroll: Current bankroll/portfolio value

        Returns:
            Optimal position size in USD
        """
        # Base Kelly calculation
        kelly_size = self.get_kelly_size(signal.edge, signal.confidence, bankroll)

        # Apply risk adjustments
        kelly_size = self._apply_risk_adjustments(kelly_size, signal)

        # Apply drawdown reduction
        drawdown_factor = self._calculate_drawdown_factor()
        kelly_size *= drawdown_factor

        # Apply weekend reduction
        weekend_factor = self._calculate_weekend_factor()
        kelly_size *= weekend_factor

        # Apply correlation reduction
        correlation_factor = self._calculate_correlation_factor(signal.market)
        kelly_size *= correlation_factor

        # Apply limits
        kelly_size = min(kelly_size, self.limits.max_position_size)

        # Check total exposure
        total_after_trade = self.state.total_exposure + kelly_size
        if total_after_trade > self.limits.max_exposure:
            max_allowed = self.limits.max_exposure - self.state.total_exposure
            kelly_size = max(0, max_allowed)

        # CRITICAL: Enforce liquidity limits to prevent slippage
        market_liquidity = getattr(signal.market, 'liquidity', 1) or 1  # Ensure non-zero
        max_safe_size = market_liquidity * 0.02  # Max 2% of liquidity
        kelly_size = min(kelly_size, max_safe_size)

        logger.info(f"💰 Position size: ${kelly_size:.2f} (Kelly: ${self.get_kelly_size(signal.edge, signal.confidence, bankroll):.2f}, Max Safe: ${max_safe_size:.2f})")
        return kelly_size

    def get_kelly_size(self, edge: float, confidence: float, bankroll: float) -> float:
        """
        Calculate Kelly Criterion optimal position size for prediction markets.

        Kelly Formula: f = (bp - q) / b
        Where for prediction markets:
        - b = (1/market_price - 1) = decimal odds received
        - p = our estimated win probability
        - q = 1 - p = loss probability

        Args:
            edge: Edge in decimal (0.05 = 5%) - unused in proper Kelly
            confidence: Confidence in prediction (0-1)
            bankroll: Current bankroll

        Returns:
            Optimal position size in USD
        """
        if confidence <= 0 or confidence >= 1:
            return 0.0

        # For prediction markets, Kelly should use the actual market odds
        # Since we don't have market_price here, use edge as approximation
        # Edge represents expected value per dollar risked
        win_prob = confidence
        loss_prob = 1 - confidence

        # Simplified Kelly approximation for prediction markets
        # f ≈ edge * confidence (since edge already factors in the odds)
        kelly_fraction = edge * win_prob

        # Apply conservative Kelly fraction (typically 1/4 to 1/2 Kelly)
        kelly_fraction *= self.limits.kelly_fraction

        # Ensure positive and reasonable bounds
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50% of bankroll

        position_size = bankroll * kelly_fraction

        return position_size

    def record_trade(self, signal: Any, size: float, market: Any) -> None:
        """
        Record a trade for position tracking and correlation analysis.

        Args:
            signal: TradingSignal object
            size: Position size in USD
            market: Market object
        """
        position = Position(
            market_id=getattr(market, 'slug', str(id(market))),
            direction=signal.direction,
            size=size,
            entry_price=signal.market_probability,
            timestamp=datetime.now(),
            correlation_group=self._get_correlation_group(market),
            expected_value=signal.expected_value
        )

        self.state.positions.append(position)
        self.state.total_exposure += size

        market_volume = getattr(market, 'volume', 0)
        logger.info(f"📝 Trade recorded: {market.question[:35]}... | {signal.direction} | ${size:.2f} | Vol: ${market_volume:,.0f}")

    def update_pnl(self, pnl: float) -> None:
        """
        Update portfolio P&L and check for drawdown/risk triggers.

        Args:
            pnl: Profit/loss amount (positive for profit, negative for loss)
        """
        old_pnl = self.state.daily_pnl
        self.state.daily_pnl += pnl

        # Update peak portfolio value
        portfolio_value = 1000 + self.state.daily_pnl  # Assuming $1000 starting capital
        self.state.peak_portfolio_value = max(self.state.peak_portfolio_value, portfolio_value)

        # Calculate drawdown
        if self.state.peak_portfolio_value > 0:
            self.state.current_drawdown = (self.state.peak_portfolio_value - portfolio_value) / self.state.peak_portfolio_value

        # Track loss for cooldown
        if pnl < 0:
            self.state.last_loss_time = datetime.now()

        logger.info(f"📊 P&L updated: ${old_pnl:.2f} → ${self.state.daily_pnl:.2f} | Drawdown: {self.state.current_drawdown:.1%}")

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get comprehensive daily trading statistics."""
        return {
            "daily_pnl": self.state.daily_pnl,
            "current_drawdown": self.state.current_drawdown,
            "total_exposure": self.state.total_exposure,
            "active_positions": len(self.state.positions),
            "trading_paused": self.state.trading_paused,
            "pause_reason": self.state.pause_reason,
            "total_trades": len(self.trade_history),
            "win_rate": self._calculate_win_rate(),
            "avg_trade_size": self._calculate_avg_trade_size(),
            "sharpe_ratio": self._calculate_sharpe_ratio()
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self.trade_history:
            return 0.0

        wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        return wins / len(self.trade_history)

    def _calculate_avg_trade_size(self) -> float:
        """Calculate average trade size."""
        if not self.trade_history:
            return 0.0

        sizes = [trade.get('size', 0) for trade in self.trade_history]
        return sum(sizes) / len(sizes)

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if not self.trade_history:
            return 0.0

        returns = [trade.get('pnl', 0) for trade in self.trade_history]
        if len(returns) < 2:
            return 0.0

        avg_return = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))

        # Assuming risk-free rate of 0 for simplicity
        return avg_return / std_dev if std_dev > 0 else 0.0

    def _check_correlation_issues(self, signal: Any, market: Any) -> Optional[str]:
        """Check for correlation issues with existing positions."""
        market_id = getattr(market, 'slug', str(id(market)))

        for position in self.state.positions:
            correlation = self.check_correlation(market_id, position.market_id)
            if correlation > self.limits.max_correlation:
                return f"High correlation ({correlation:.2f}) with existing position in {position.market_id}"

        return None

    def check_correlation(self, market1: str, market2: str) -> float:
        """
        Check correlation between two markets.

        Args:
            market1: First market identifier
            market2: Second market identifier

        Returns:
            Correlation coefficient (0-1, higher = more correlated)
        """
        # Check cache first
        cache_key = tuple(sorted([market1, market2]))
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]

        # Simple correlation based on market categories and keywords
        correlation = self._calculate_market_correlation(market1, market2)

        # Cache result
        self.correlation_cache[cache_key] = correlation
        return correlation

    def _calculate_market_correlation(self, market1: str, market2: str) -> float:
        """Calculate correlation based on market similarity."""
        if market1 == market2:
            return 1.0

        # Simple keyword-based correlation
        # In a real implementation, this would use NLP and historical price correlation
        market1_lower = market1.lower()
        market2_lower = market2.lower()

        # Check for common keywords
        common_keywords = ['bitcoin', 'crypto', 'president', 'election', 'fed', 'economy']
        matches = 0

        for keyword in common_keywords:
            if keyword in market1_lower and keyword in market2_lower:
                matches += 1

        # Return correlation based on keyword matches
        if matches >= 2:
            return 0.8  # High correlation
        elif matches == 1:
            return 0.5  # Medium correlation
        else:
            return 0.1  # Low correlation

    def _get_correlation_group(self, market: Any) -> str:
        """Assign market to correlation group."""
        market_name = getattr(market, 'question', str(market)).lower()

        if any(word in market_name for word in ['bitcoin', 'btc', 'crypto']):
            return 'crypto'
        elif any(word in market_name for word in ['president', 'election', 'biden', 'trump']):
            return 'politics'
        elif any(word in market_name for word in ['fed', 'economy', 'recession']):
            return 'economy'
        else:
            return 'other'

    def _apply_risk_adjustments(self, base_size: float, signal: Any) -> float:
        """Apply additional risk adjustments to position size."""
        # Reduce size for low confidence signals
        confidence_factor = min(1.0, signal.confidence / 0.5)  # Scale down below 50% confidence

        # Reduce size for very short-term markets
        # (Already handled in can_trade, but additional reduction here)

        return base_size * confidence_factor

    def _calculate_drawdown_factor(self) -> float:
        """Calculate position size reduction factor based on drawdown."""
        if self.state.current_drawdown >= self.limits.drawdown_threshold_2:
            return 0.0  # Stop trading
        elif self.state.current_drawdown >= self.limits.drawdown_threshold_1:
            # Linear reduction from 1.0 to 0.5 as drawdown goes from 10% to 20%
            reduction = 1.0 - (self.state.current_drawdown - self.limits.drawdown_threshold_1) / self.limits.drawdown_threshold_1
            return max(0.5, reduction)  # Minimum 50% size
        else:
            return 1.0  # No reduction

    def _calculate_weekend_factor(self) -> float:
        """Calculate weekend trading reduction factor."""
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return self.limits.weekend_reduction
        return 1.0

    def _calculate_correlation_factor(self, market: Any) -> float:
        """Calculate size reduction based on correlation with existing positions."""
        market_id = getattr(market, 'slug', str(id(market)))
        max_correlation = 0.0

        for position in self.state.positions:
            correlation = self.check_correlation(market_id, position.market_id)
            max_correlation = max(max_correlation, correlation)

        # Reduce size for highly correlated positions
        if max_correlation > 0.8:
            return 0.3  # 70% reduction
        elif max_correlation > 0.6:
            return 0.5  # 50% reduction
        elif max_correlation > 0.4:
            return 0.7  # 30% reduction
        else:
            return 1.0  # No reduction

    def _load_market_correlations(self) -> Dict[str, List[str]]:
        """Load predefined market correlation groups."""
        return {
            'crypto': ['bitcoin', 'ethereum', 'crypto', 'btc', 'eth'],
            'politics': ['president', 'election', 'biden', 'trump', 'congress'],
            'economy': ['fed', 'economy', 'recession', 'inflation', 'unemployment'],
            'tech': ['apple', 'google', 'microsoft', 'tesla', 'tech'],
            'sports': ['football', 'basketball', 'baseball', 'soccer', 'sports']
        }

    def _reset_daily_stats(self):
        """Reset daily statistics at start of new day."""
        self.state.daily_start_time = datetime.now().date()
        self.state.daily_pnl = 0.0
        self.state.peak_portfolio_value = 0.0
        self.state.current_drawdown = 0.0
        self.state.trading_paused = False
        self.state.pause_reason = ""

        logger.info("📅 Daily statistics reset - new trading day started")

    # ===== ADVANCED RISK MANAGEMENT METHODS =====

    def calculate_kelly_with_uncertainty(
        self,
        estimated_probability: float,
        confidence: float,
        price: float,
        max_kelly: float = 0.25
    ) -> float:
        """
        Kelly criterion that accounts for uncertainty in estimate.

        Standard Kelly: f* = (bp - q) / b

        But we're uncertain about p, so we use fractional Kelly.
        Lower confidence = more fractional.

        Args:
            estimated_probability: AI's probability estimate (0-1)
            confidence: Confidence in the estimate (0-1)
            price: Current market price (0-1)
            max_kelly: Maximum Kelly fraction allowed

        Returns:
            Fraction of bankroll to bet
        """
        # Base Kelly calculation
        b = (1 / price) - 1  # Decimal odds - 1
        p = estimated_probability
        q = 1 - p

        if b <= 0:
            return 0

        kelly = (b * p - q) / b
        kelly = max(0, kelly)

        # Adjust for confidence uncertainty
        # confidence 1.0 = full Kelly * max_kelly
        # confidence 0.5 = Kelly * 0.5 * max_kelly
        adjusted_kelly = kelly * confidence * max_kelly

        # Also reduce if edge is uncertain (edge < confidence suggests uncertainty)
        edge = p - price
        edge_reliability = min(confidence, 0.8)  # Cap at 80%

        final_kelly = adjusted_kelly * edge_reliability

        return min(final_kelly, max_kelly)

    def calculate_correlated_position_size(
        self,
        signal: Any,
        existing_positions: List[Position]
    ) -> float:
        """
        Reduce position size if correlated with existing positions.

        Args:
            signal: TradingSignal object
            existing_positions: List of current positions

        Returns:
            Adjusted position size multiplier (0-1)
        """
        correlations = self._calculate_correlations(signal.market, existing_positions)

        # If highly correlated with existing position, reduce size
        max_correlation = max(correlations.values()) if correlations else 0

        size_multiplier = 1.0 - (max_correlation * 0.5)  # Reduce by up to 50%

        return max(size_multiplier, 0.1)  # Minimum 10% size

    def _calculate_correlations(
        self,
        market: Any,
        positions: List[Position]
    ) -> Dict[str, float]:
        """
        Estimate correlation between markets using keyword overlap.

        Args:
            market: Current market object
            positions: List of existing positions

        Returns:
            Dict of position_id -> correlation_score (0-1)
        """
        correlations = {}

        # Get market attributes
        market_question = getattr(market, 'question', '').lower()
        market_category = getattr(market, 'category', '')

        market_words = set(market_question.split()) if market_question else set()

        for pos in positions:
            # Keyword correlation
            pos_words = set(pos.market_question.lower().split()) if pos.market_question else set()
            word_overlap = len(market_words & pos_words) / max(len(market_words | pos_words), 1)

            correlation = word_overlap

            # Same category bonus
            if market_category and pos.category == market_category:
                correlation += 0.2

            # Cap at 1.0
            correlations[pos.market_id] = min(correlation, 1.0)

        return correlations

    def update_portfolio_value(self, portfolio_value: float):
        """Update portfolio value for drawdown tracking."""
        self.drawdown_protector.update(portfolio_value)

        # Update bankroll
        self.bankroll_manager.update_bankroll(portfolio_value)

        # Check if trading should be paused due to drawdown
        if self.drawdown_protector.should_pause_trading():
            self.emergency_stop("Drawdown protection triggered")

    def get_position_size_multiplier(self) -> float:
        """
        Get overall position size multiplier based on all risk factors.

        Returns:
            Multiplier to apply to position sizes (0-1)
        """
        multiplier = 1.0

        # Drawdown protection
        multiplier *= self.drawdown_protector.get_size_multiplier()

        # Weekend reduction (if applicable)
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            multiplier *= self.limits.weekend_reduction

        return max(multiplier, 0.1)  # Minimum 10%

    def calculate_position_size_advanced(
        self,
        signal: Any,
        market: Any,
        existing_positions: List[Position]
    ) -> float:
        """
        Advanced position sizing with all risk factors considered.

        Args:
            signal: TradingSignal object
            market: Market object
            existing_positions: Current open positions

        Returns:
            Recommended position size in USD
        """
        # Base Kelly sizing with uncertainty
        base_size = self.calculate_kelly_with_uncertainty(
            signal.probability,
            signal.confidence,
            signal.market_probability,
            self.limits.kelly_fraction
        )

        # Apply bankroll limits
        bankroll = self.bankroll_manager.current_bankroll
        max_risk_per_trade = self.bankroll_manager.get_risk_per_trade()
        base_size = min(base_size, max_risk_per_trade)

        # Correlation adjustment
        correlation_multiplier = self.calculate_correlated_position_size(signal, existing_positions)
        base_size *= correlation_multiplier

        # Overall risk multiplier (drawdown, weekend, etc.)
        risk_multiplier = self.get_position_size_multiplier()
        base_size *= risk_multiplier

        # Apply absolute limits
        base_size = min(base_size, self.limits.max_position_size)

        # Bankroll affordability check
        if not self.bankroll_manager.can_afford_trade(base_size):
            base_size = self.bankroll_manager.get_risk_per_trade()

        return max(base_size, 0.0)

    def record_daily_performance(self, daily_return_pct: float):
        """Record daily performance for Sharpe ratio calculation."""
        self.performance_tracker.record_daily_return(daily_return_pct)

    def record_trade_performance(self, trade_return: float):
        """Record individual trade performance."""
        self.performance_tracker.record_trade(trade_return)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = self.performance_tracker.generate_performance_report()

        # Add risk management specific metrics
        report.update({
            "current_drawdown": self.drawdown_protector.current_drawdown,
            "max_drawdown": self.drawdown_protector.max_drawdown,
            "bankroll_health": self.bankroll_manager.get_bankroll_health(),
            "trading_paused": self.state.trading_paused,
            "pause_reason": self.state.pause_reason,
            "open_positions": len(self.state.positions),
            "total_exposure": self.state.total_exposure,
        })

        return report

    def emergency_stop(self, reason: str):
        """Emergency stop all trading."""
        self.state.trading_paused = True
        self.state.pause_reason = f"EMERGENCY STOP: {reason}"
        logger.critical(f"🚨 EMERGENCY STOP: {reason}")

    def resume_trading(self):
        """Resume trading after emergency stop."""
        self.state.trading_paused = False
        self.state.pause_reason = ""
        logger.info("✅ Trading resumed")


def run_self_tests():
    """Run self-tests for critical risk management functions."""
    print("🧪 Risk Manager Self-Tests")

    # Test Kelly calculation
    limits = RiskLimits(kelly_fraction=0.25, max_position_size=1000.0)
    risk_mgr = RiskManager(limits=limits, initial_bankroll=1000.0)

    # Test basic Kelly calculation
    kelly_size = risk_mgr.get_kelly_size(0.1, 0.8, 1000.0)
    assert kelly_size > 0, f"Kelly size should be positive, got {kelly_size}"
    assert kelly_size <= 250.0, f"Kelly size should be <= 250 (25% of bankroll), got {kelly_size}"

    # Test zero edge gives zero size
    zero_kelly = risk_mgr.get_kelly_size(0.0, 0.8, 1000.0)
    assert zero_kelly == 0.0, f"Zero edge should give zero Kelly size, got {zero_kelly}"

    # Test slippage estimation
    mock_market = type('MockMarket', (), {'liquidity': 1000.0})()
    slippage_estimate = risk_mgr.slippage.estimate_slippage(mock_market, 50.0, "YES")
    assert 0 <= slippage_estimate.estimated_slippage_pct <= 1, f"Slippage should be 0-1, got {slippage_estimate.estimated_slippage_pct}"
    assert isinstance(slippage_estimate.reason, str), f"Reason should be string, got {type(slippage_estimate.reason)}"

    print("✅ Risk Manager self-tests passed")


if __name__ == "__main__":
    run_self_tests()
