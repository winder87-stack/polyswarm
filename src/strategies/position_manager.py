"""
Position Manager - Active management of open trading positions

Handles position lifecycle, scaling in/out, stop losses, and portfolio optimization
for Polymarket prediction market trading.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid

from loguru import logger

try:
    from src.agents.trading_swarm import TradingSwarm
    from src.agents.signals import TradingSignal
    from src.connectors import Market
except ImportError:
    # For testing without full imports
    class TradingSwarm:
        pass

    class TradingSignal:
        pass

    from src.connectors import Market


@dataclass
class ManagedPosition:
    """Tracks and manages an open trading position."""
    position_id: str
    market: Any  # Market object
    direction: str  # "YES" or "NO"

    # Entry details
    entry_price: float
    entry_time: datetime
    entry_size: float  # Total size in USDC
    entry_shares: float
    entry_edge: float
    entry_confidence: float

    # Current state
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

    # Management state
    scale_ins: List[Dict] = field(default_factory=list)  # Additional buys
    scale_outs: List[Dict] = field(default_factory=list)  # Partial sells
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    # Time tracking
    hours_held: float = 0.0
    hours_until_expiry: float = 0.0

    # Risk flags
    is_underwater: bool = False  # Losing position
    needs_attention: bool = False  # Something unusual
    last_updated: datetime = field(default_factory=datetime.now)

    def update(self, new_price: float, expiry_hours: Optional[float] = None):
        """Update position with new market price."""
        self.current_price = new_price
        self.current_value = self.entry_shares * new_price

        # Calculate total invested (including scale-ins)
        total_invested = self.entry_size
        for scale_in in self.scale_ins:
            total_invested += scale_in["size"]

        # Calculate total shares held
        total_shares = self.entry_shares
        for scale_in in self.scale_ins:
            total_shares += scale_in["shares"]

        # Account for scale-outs (partial sells)
        shares_sold = sum(s["shares"] for s in self.scale_outs)
        total_shares -= shares_sold

        # Recalculate current value and P&L
        self.current_value = total_shares * new_price

        # Calculate realized P&L from scale-outs
        realized_pnl = sum(
            (s["price"] - self.entry_price) * s["shares"]
            for s in self.scale_outs
        )

        # Calculate unrealized P&L
        self.unrealized_pnl = self.current_value - (total_invested - realized_pnl)
        self.unrealized_pnl_pct = (self.unrealized_pnl / total_invested) * 100 if total_invested > 0 else 0

        self.is_underwater = self.unrealized_pnl < 0

        # Update time tracking
        self.hours_held = (datetime.now() - self.entry_time).total_seconds() / 3600
        if expiry_hours is not None:
            self.hours_until_expiry = expiry_hours

        self.last_updated = datetime.now()

    def add_scale_in(self, price: float, shares: float, size: float):
        """Record a scale-in (additional purchase)."""
        self.scale_ins.append({
            "timestamp": datetime.now(),
            "price": price,
            "shares": shares,
            "size": size
        })
        logger.info(f"📈 Scaled into {self.position_id}: +${size:.0f} at ${price:.2f}")

    def add_scale_out(self, price: float, shares: float, size: float, reason: str):
        """Record a scale-out (partial sell)."""
        self.scale_outs.append({
            "timestamp": datetime.now(),
            "price": price,
            "shares": shares,
            "size": size,
            "reason": reason
        })
        logger.info(f"📉 Scaled out of {self.position_id}: -${size:.0f} at ${price:.2f} ({reason})")

    def get_remaining_shares(self) -> float:
        """Get shares still held (after scale-outs)."""
        total_shares = self.entry_shares
        for scale_in in self.scale_ins:
            total_shares += scale_in["shares"]
        for scale_out in self.scale_outs:
            total_shares -= scale_out["shares"]
        return max(0, total_shares)

    def get_total_invested(self) -> float:
        """Get total capital invested (including scale-ins)."""
        total = self.entry_size
        for scale_in in self.scale_ins:
            total += scale_in["size"]
        return total

    def get_realized_pnl(self) -> float:
        """Get realized P&L from scale-outs."""
        realized = 0
        for scale_out in self.scale_outs:
            # Approximate realized P&L (using entry price as reference)
            realized += (scale_out["price"] - self.entry_price) * scale_out["shares"]
        return realized

    def __str__(self) -> str:
        """String representation for logging."""
        market_name = getattr(self.market, 'question', 'Unknown Market')[:40]
        pnl_str = f"{self.unrealized_pnl:+.0f}" if not self.is_underwater else f"{self.unrealized_pnl:.0f}"
        return f"{market_name} ({self.direction}) - {pnl_str} ({self.unrealized_pnl_pct:+.1f}%)"


class PositionManager:
    """
    Active position management for prediction market trading.

    Handles position lifecycle from entry to exit, including scaling in/out,
    stop losses, and portfolio optimization.
    """

    def __init__(self, trading_swarm: TradingSwarm) -> None:
        """Initialize position manager."""
        self.swarm = trading_swarm
        self.positions: Dict[str, ManagedPosition] = {}

        logger.info("📊 Position manager initialized")

    async def add_position(self, signal: TradingSignal, execution: Dict):
        """Add a new position to manage."""
        try:
            position_id = str(uuid.uuid4())[:8]

            # Calculate expiry time
            end_date = getattr(signal.market, 'end_date', None)
            hours_until_expiry = 168  # Default 1 week
            if end_date:
                try:
                    if isinstance(end_date, str):
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    hours_until_expiry = (end_date - datetime.now()).total_seconds() / 3600
                except ValueError as e:
                    logger.debug(f"Could not parse expiry date in position manager: {e}")
                    pass

            position = ManagedPosition(
                position_id=position_id,
                market=signal.market,
                direction=signal.direction,
                entry_price=signal.market_probability,
                entry_time=datetime.now(),
                entry_size=execution.get("size", 0),
                entry_shares=execution.get("shares", 0),
                entry_edge=signal.edge,
                entry_confidence=signal.confidence,
                current_price=signal.market_probability,
                current_value=execution.get("size", 0),
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                hours_until_expiry=hours_until_expiry
            )

            # Set initial stop loss (30% below entry)
            position.stop_loss_price = position.entry_price * 0.7

            self.positions[position_id] = position

            logger.info(f"✅ Added position {position_id}: {position}")

        except Exception as e:
            logger.error(f"Failed to add position: {e}")

    async def update_all_positions(self):
        """Update all positions with current market prices."""
        try:
            # In a real implementation, this would fetch current prices from Polymarket
            # For now, we'll simulate price updates
            for position in self.positions.values():
                # Simulate slight price movement (real implementation would use live prices)
                price_change = (datetime.now().timestamp() % 100 - 50) / 10000  # Small random change
                new_price = position.current_price + price_change
                new_price = max(0.01, min(0.99, new_price))  # Keep in valid range

                # Update expiry time
                hours_until_expiry = max(0, position.hours_until_expiry - 1/60)  # Subtract 1 minute

                position.update(new_price, hours_until_expiry)

        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    async def check_position_actions(self, position: ManagedPosition) -> Optional[Dict]:
        """
        Check if any action is needed for this position.

        Returns action dict if action needed, None otherwise.
        """
        try:
            # Check scale-in opportunity
            should_scale_in, scale_amount, scale_reason = await self.should_scale_in(position)
            if should_scale_in:
                return {
                    "action": "scale_in",
                    "position_id": position.position_id,
                    "amount": scale_amount,
                    "reason": scale_reason
                }

            # Check scale-out opportunity
            should_scale_out, shares_to_sell, scale_out_reason = await self.should_scale_out(position)
            if should_scale_out:
                return {
                    "action": "scale_out",
                    "position_id": position.position_id,
                    "shares": shares_to_sell,
                    "reason": scale_out_reason
                }

            # Check if should close position
            should_close, close_reason = await self.should_close_position(position)
            if should_close:
                return {
                    "action": "close",
                    "position_id": position.position_id,
                    "reason": close_reason
                }

            # Check if needs reanalysis
            if await self.should_reanalyze(position):
                return {
                    "action": "reanalyze",
                    "position_id": position.position_id,
                    "reason": "Periodic reanalysis"
                }

            return None

        except Exception as e:
            logger.error(f"Failed to check position actions for {position.position_id}: {e}")
            return None

    async def should_scale_in(self, position: ManagedPosition) -> Tuple[bool, float, str]:
        """
        Check if we should add to this position (averaging down).

        Scale in when position is down but thesis still valid.
        """
        try:
            # Only scale in if position is down 15-25%
            if position.unrealized_pnl_pct > -15 or position.unrealized_pnl_pct < -35:
                return False, 0, "Position not in scale-in range"

            # Check if we've already scaled in too much (max 2x original size)
            total_invested = position.get_total_invested()
            if total_invested > position.entry_size * 2.5:
                return False, 0, "Already scaled in too much"

            # Re-analyze the market to confirm thesis still valid
            signal = await self.swarm.analyze_market(position.market)

            if not signal:
                return False, 0, "Could not re-analyze market"

            # Check if AI still agrees with our direction
            if signal.direction != position.direction:
                return False, 0, f"AI flipped direction to {signal.direction}"

            # Check if edge is still attractive
            if signal.edge < 0.08:
                return False, 0, f"Edge too small: {signal.edge:.1%}"

            # Scale in amount: 30-50% of original position
            scale_amount = position.entry_size * 0.4

            return True, scale_amount, f"Averaging down: {position.unrealized_pnl_pct:.1f}% down but edge still {signal.edge:.1%}"

        except Exception as e:
            logger.error(f"Scale-in check failed for {position.position_id}: {e}")
            return False, 0, f"Analysis failed: {e}"

    async def should_scale_out(self, position: ManagedPosition) -> Tuple[bool, float, str]:
        """
        Check if we should take partial profits.

        Scale out when position is up significantly.
        """
        try:
            # Don't scale out losing positions
            if position.unrealized_pnl_pct < 5:
                return False, 0, "Position not profitable enough"

            remaining_shares = position.get_remaining_shares()

            # Scale out at different profit levels
            if position.unrealized_pnl_pct >= 50 and remaining_shares > position.entry_shares * 0.25:
                # Take 1/3 off at +50%
                sell_shares = remaining_shares * 0.33
                return True, sell_shares, f"Major profit at +{position.unrealized_pnl_pct:.1f}%, taking 1/3 off"

            elif position.unrealized_pnl_pct >= 35 and remaining_shares > position.entry_shares * 0.33:
                # Take 1/4 off at +35%
                sell_shares = remaining_shares * 0.25
                return True, sell_shares, f"Good profit at +{position.unrealized_pnl_pct:.1f}%, taking 1/4 off"

            elif position.unrealized_pnl_pct >= 20 and remaining_shares > position.entry_shares * 0.5:
                # Take 1/5 off at +20%
                sell_shares = remaining_shares * 0.2
                return True, sell_shares, f"Solid profit at +{position.unrealized_pnl_pct:.1f}%, taking 1/5 off"

            # Time-based scale out near expiry
            if position.hours_until_expiry < 48 and position.unrealized_pnl_pct > 0 and remaining_shares > 0:
                sell_shares = remaining_shares * 0.5
                return True, sell_shares, f"Approaching expiry ({position.hours_until_expiry:.0f}h), securing 50% profits"

            return False, 0, "No scale-out conditions met"

        except Exception as e:
            logger.error(f"Scale-out check failed for {position.position_id}: {e}")
            return False, 0, f"Analysis failed: {e}"

    async def should_close_position(self, position: ManagedPosition) -> Tuple[bool, str]:
        """
        Check if we should close the entire position.
        """
        try:
            # Stop loss: down 30%
            if position.unrealized_pnl_pct < -30:
                return True, f"Stop loss triggered at {position.unrealized_pnl_pct:.1f}%"

            # Re-analyze market
            signal = await self.swarm.analyze_market(position.market)

            if not signal:
                # If we can't analyze, hold if not too bad
                if position.unrealized_pnl_pct > -20:
                    return False, "Cannot analyze but position acceptable"
                else:
                    return True, "Cannot analyze and position deteriorating"

            # AI strongly disagrees
            if signal.direction != position.direction and signal.confidence > 0.75:
                return True, f"AI strongly disagrees ({signal.confidence:.0%} confidence)"

            # Edge completely gone
            if signal.edge < 0.02:
                return True, f"Edge evaporated: {signal.edge:.1%}"

            # Held too long with no movement (1 week)
            if position.hours_held > 168 and abs(position.unrealized_pnl_pct) < 3:
                return True, "Position stagnant for 1 week"

            # Market resolved or expiring very soon
            if position.hours_until_expiry < 6:
                if position.unrealized_pnl_pct > 0:
                    return True, "Market expiring soon, lock in profits"
                elif position.unrealized_pnl_pct < -10:
                    return True, "Market expiring soon, cut losses"

            return False, "Hold position"

        except Exception as e:
            logger.error(f"Close check failed for {position.position_id}: {e}")
            return False, f"Analysis failed: {e}"

    async def should_reanalyze(self, position: ManagedPosition) -> bool:
        """
        Check if position needs reanalysis.

        Reanalyze every 12 hours or if something unusual happens.
        """
        try:
            hours_since_entry = position.hours_held
            hours_since_update = (datetime.now() - position.last_updated).total_seconds() / 3600

            # Reanalyze every 12 hours
            if hours_since_update > 12:
                return True

            # Reanalyze if position moved significantly
            if abs(position.unrealized_pnl_pct) > 15 and hours_since_update > 6:
                return True

            # Reanalyze near expiry
            if position.hours_until_expiry < 72 and hours_since_update > 12:
                return True

            return False

        except Exception:
            return False

    async def execute_position_action(self, action: Dict) -> bool:
        """
        Execute a position management action.

        Returns True if action executed successfully.
        """
        try:
            position_id = action["position_id"]
            position = self.positions.get(position_id)

            if not position:
                logger.error(f"Position {position_id} not found")
                return False

            action_type = action["action"]

            if action_type == "scale_in":
                # In a real implementation, this would execute a buy order
                amount = action["amount"]
                logger.info(f"🔄 Scaling into {position_id}: ${amount:.0f}")

                # Simulate scale-in
                scale_price = position.current_price
                scale_shares = amount / scale_price
                position.add_scale_in(scale_price, scale_shares, amount)

                return True

            elif action_type == "scale_out":
                # In a real implementation, this would execute a sell order
                shares_to_sell = action["shares"]
                sell_price = position.current_price
                sell_amount = shares_to_sell * sell_price

                logger.info(f"🔄 Scaling out of {position_id}: {shares_to_sell:.0f} shares at ${sell_price:.2f}")

                position.add_scale_out(sell_price, shares_to_sell, sell_amount, action["reason"])

                return True

            elif action_type == "close":
                # Close entire position
                remaining_shares = position.get_remaining_shares()
                if remaining_shares > 0:
                    logger.info(f"🔄 Closing {position_id}: {remaining_shares:.0f} shares")

                    # Simulate close
                    position.add_scale_out(
                        position.current_price,
                        remaining_shares,
                        remaining_shares * position.current_price,
                        action["reason"]
                    )

                # Remove from managed positions
                del self.positions[position_id]
                logger.info(f"✅ Closed position {position_id}")

                return True

            elif action_type == "reanalyze":
                # Re-run analysis (already done in check_position_actions)
                logger.info(f"🔄 Reanalyzed {position_id}")
                return True

            else:
                logger.warning(f"Unknown action type: {action_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to execute position action: {e}")
            return False

    async def run_management_loop(self, interval_minutes: int = 60):
        """
        Continuously manage all positions.

        Every interval:
        1. Update all prices
        2. Check each position for actions
        3. Execute any needed actions
        4. Log portfolio status
        """
        logger.info(f"🔄 Starting position management loop (every {interval_minutes} minutes)")

        while True:
            try:
                # Update all positions
                await self.update_all_positions()

                actions_taken = 0

                # Check each position for actions
                for position in list(self.positions.values()):
                    action = await self.check_position_actions(position)
                    if action:
                        success = await self.execute_position_action(action)
                        if success:
                            actions_taken += 1

                # Log portfolio summary
                if actions_taken > 0 or len(self.positions) > 0:
                    summary = self.get_portfolio_summary()
                    logger.info("📊 Portfolio: {} positions, ${:.0f} value, ${:.0f} P&L ({:.1f}%)".format(
                        summary["total_positions"],
                        summary["total_value"],
                        summary["total_pnl"],
                        summary["total_pnl_pct"]
                    ))

                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Position management loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.
        """
        try:
            if not self.positions:
                return {
                    "total_positions": 0,
                    "total_value": 0.0,
                    "total_pnl": 0.0,
                    "total_pnl_pct": 0.0,
                    "winners": 0,
                    "losers": 0,
                    "largest_winner": None,
                    "largest_loser": None,
                    "positions": []
                }

            total_value = 0.0
            total_invested = 0.0
            winners = 0
            losers = 0
            largest_winner = None
            largest_loser = None

            position_summaries = []

            for position in self.positions.values():
                total_value += position.current_value
                total_invested += position.get_total_invested()

                if position.unrealized_pnl > 0:
                    winners += 1
                    if not largest_winner or position.unrealized_pnl > largest_winner["pnl"]:
                        largest_winner = {
                            "position_id": position.position_id,
                            "market": str(position),
                            "pnl": position.unrealized_pnl,
                            "pnl_pct": position.unrealized_pnl_pct
                        }
                elif position.unrealized_pnl < 0:
                    losers += 1
                    if not largest_loser or position.unrealized_pnl < largest_loser["pnl"]:
                        largest_loser = {
                            "position_id": position.position_id,
                            "market": str(position),
                            "pnl": position.unrealized_pnl,
                            "pnl_pct": position.unrealized_pnl_pct
                        }

                position_summaries.append({
                    "position_id": position.position_id,
                    "market": str(position),
                    "direction": position.direction,
                    "entry_size": position.entry_size,
                    "current_value": position.current_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_pct": position.unrealized_pnl_pct,
                    "hours_held": position.hours_held,
                    "scale_ins": len(position.scale_ins),
                    "scale_outs": len(position.scale_outs)
                })

            total_pnl = total_value - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

            return {
                "total_positions": len(self.positions),
                "total_value": round(total_value, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 1),
                "winners": winners,
                "losers": losers,
                "largest_winner": largest_winner,
                "largest_loser": largest_loser,
                "positions": position_summaries
            }

        except Exception as e:
            logger.error(f"Failed to generate portfolio summary: {e}")
            return {"error": str(e)}

    def log_position_status(self, position: ManagedPosition):
        """Log detailed position status."""
        market_name = getattr(position.market, 'question', 'Unknown Market')[:50]

        status = "📊 [POSITION] \"{}\" - {}".format(
            market_name,
            "UNDERWATER" if position.is_underwater else "PROFITABLE"
        )

        logger.info(status)
        logger.info("   Entry: ${:.2f} → Current: ${:.2f} ({:+.1f}%)".format(
            position.entry_price, position.current_price, position.unrealized_pnl_pct
        ))
        logger.info("   Size: ${:.0f} ({:.0f} shares) | Value: ${:.2f}".format(
            position.entry_size, position.entry_shares, position.current_value
        ))
        logger.info("   Edge at entry: {:.1f}% | Hours held: {:.0f} | Hours to expiry: {:.0f}".format(
            position.entry_edge * 100, position.hours_held, position.hours_until_expiry
        ))

        if position.scale_ins:
            logger.info("   Scale-ins: {} (${:.0f} total)".format(
                len(position.scale_ins),
                sum(s["size"] for s in position.scale_ins)
            ))

        if position.scale_outs:
            logger.info("   Scale-outs: {} (${:.0f} realized)".format(
                len(position.scale_outs),
                sum(s["size"] for s in position.scale_outs)
            ))

        logger.info("   Status: Hold (monitoring)")

    def remove_position(self, position_id: str):
        """Remove a position from management (e.g., when manually closed)."""
        if position_id in self.positions:
            del self.positions[position_id]
            logger.info(f"🗑️ Removed position {position_id} from management")

    def __str__(self) -> str:
        """String representation."""
        summary = self.get_portfolio_summary()
        return f"PositionManager({summary['total_positions']} positions, ${summary['total_value']:.0f} value, {summary['total_pnl_pct']:+.1f}% P&L)"


# Global instance for easy access
position_manager = None

def create_position_manager(trading_swarm: TradingSwarm) -> PositionManager:
    """Create and return a position manager instance."""
    global position_manager
    position_manager = PositionManager(trading_swarm)
    return position_manager
