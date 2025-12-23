"""
Polymarket Client - Based on Official Documentation
https://docs.polymarket.com/

Two APIs:
- Gamma API (gamma-api.polymarket.com): Market data, no auth needed
- CLOB API (clob.polymarket.com): Trading, requires auth
"""

import os
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from loguru import logger

# Rate limiter for API limits
from src.utils.rate_limiter import rate_limiter

# Official Polymarket Python client
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    BookParams
)
from py_clob_client.order_builder.constants import BUY, SELL


# API Endpoints
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon Mainnet


@dataclass
class Market:
    """Market data from Polymarket Gamma API"""
    condition_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    category: str
    yes_token_id: str = ""
    no_token_id: str = ""
    tick_size: str = "0.01"
    neg_risk: bool = False
    min_order_size: float = 5.0
    end_date: str = ""

    @property
    def spread(self) -> float:
        return abs(1.0 - self.yes_price - self.no_price)


class PolymarketClient:
    """
    Polymarket API client following official documentation.

    Usage:
        client = PolymarketClient()

        # Read operations (no auth needed)
        markets = await client.get_markets(limit=10)

        # Trading (requires wallet setup)
        await client.place_limit_order(
            token_id=market.yes_token_id,
            price=0.55,
            size=10,
            side="BUY"
        )
    """

    def __init__(self):
        self.gamma_url = GAMMA_HOST
        self.clob_url = CLOB_HOST

        # Credentials from environment
        self.private_key = os.getenv("POLYGON_WALLET_PRIVATE_KEY", "")
        self.funder = os.getenv("POLYGON_FUNDER_ADDRESS", "")
        self.sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))

        # Add 0x prefix to private key if missing
        if self.private_key and not self.private_key.startswith("0x"):
            self.private_key = "0x" + self.private_key

        # Initialize clients
        self._init_read_client()
        self._init_trade_client()

        logger.info(f"📊 PolymarketClient initialized")
        logger.info(f"   Gamma API: {self.gamma_url}")
        logger.info(f"   CLOB API: {self.clob_url}")
        logger.info(f"   Trading: {'Enabled' if self.trade_client else 'Disabled (read-only)'}")

    def _init_read_client(self):
        """Initialize read-only client (no auth needed)"""
        try:
            self.read_client = ClobClient(CLOB_HOST)
            logger.debug("📖 Read-only CLOB client initialized")
        except Exception as e:
            logger.error(f"Failed to init read client: {e}")
            self.read_client = None

    def _init_trade_client(self):
        """Initialize trading client with full auth"""
        self.trade_client = None

        if not self.private_key:
            logger.warning("⚠️ No POLYGON_WALLET_PRIVATE_KEY - trading disabled")
            return

        if not self.funder:
            logger.warning("⚠️ No POLYGON_FUNDER_ADDRESS - trading disabled")
            return

        try:
            # Create client with wallet
            self.trade_client = ClobClient(
                CLOB_HOST,
                key=self.private_key,
                chain_id=CHAIN_ID,
                signature_type=self.sig_type,
                funder=self.funder
            )

            # Set up API credentials
            self._setup_api_creds()

            logger.info(f"💰 Trade client initialized")
            logger.info(f"   Funder: {self.funder[:10]}...{self.funder[-6:]}")
            logger.info(f"   Signature type: {self.sig_type} ({'EOA' if self.sig_type == 0 else 'Proxy'})")

        except Exception as e:
            logger.error(f"❌ Failed to init trade client: {e}")
            self.trade_client = None

    def _setup_api_creds(self):
        """Set up API credentials from env or derive new ones"""
        api_key = os.getenv("POLYMARKET_API_KEY")
        api_secret = os.getenv("POLYMARKET_API_SECRET")
        api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")

        if api_key and api_secret and api_passphrase:
            # Use existing credentials
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase
            )
            self.trade_client.set_api_creds(creds)
            logger.debug(f"✅ Using existing API credentials")
        else:
            # Derive new credentials (deterministic based on wallet)
            try:
                creds = self.trade_client.create_or_derive_api_creds()
                self.trade_client.set_api_creds(creds)
                logger.info(f"✅ Generated new API credentials")
                logger.info(f"   💡 Add to .env to avoid regenerating:")
                logger.info(f"   POLYMARKET_API_KEY={creds.api_key}")
                logger.info(f"   POLYMARKET_API_SECRET={creds.api_secret}")
                logger.info(f"   POLYMARKET_API_PASSPHRASE={creds.api_passphrase}")
            except Exception as e:
                logger.error(f"Failed to derive API creds: {e}")

    # =========================================
    # GAMMA API - Market Data (No Auth)
    # =========================================

    async def get_markets(
        self,
        limit: int = 50,
        min_volume: float = 50000,
        min_liquidity: float = 10000,
        active_only: bool = True,
        closed: bool = False,
    ) -> List[Market]:
        """
        Fetch markets from Gamma API with filtering.

        Args:
            limit: Max markets to return
            min_volume: Minimum 24h volume in USD
            min_liquidity: Minimum liquidity in USD
            active_only: Only return active markets
            closed: Include closed markets
        """
        try:
            # Rate limiting for Gamma /markets endpoint
            await rate_limiter.wait_for_gamma_markets()

            url = f"{self.gamma_url}/markets"
            params = {
                "limit": min(limit * 2, 100),  # Fetch extra to filter
                "active": str(active_only).lower(),
                "closed": str(closed).lower(),
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Gamma API error: {response.status}")
                        return []

                    data = await response.json()

            markets = []
            for m in data:
                try:
                    # Parse volume and liquidity
                    vol = float(m.get("volume", 0) or 0)
                    liq = float(m.get("liquidity", 0) or 0)

                    # Apply filters
                    if vol < min_volume:
                        continue
                    if liq < min_liquidity:
                        continue

                    # Parse prices
                    prices = m.get("outcomePrices", [])
                    if len(prices) >= 2:
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                    else:
                        yes_price = 0.5
                        no_price = 0.5

                    # Get token IDs (CRITICAL for order placement)
                    clob_tokens = m.get("clobTokenIds", [])
                    yes_token = clob_tokens[0] if len(clob_tokens) > 0 else ""
                    no_token = clob_tokens[1] if len(clob_tokens) > 1 else ""

                    # Skip if no token IDs (can't trade)
                    if not yes_token or not no_token:
                        continue

                    # Check expiry
                    end_date = m.get("endDate", "")
                    if end_date:
                        try:
                            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600
                            if hours_left < 24:  # Skip markets closing in <24h
                                continue
                        except:
                            pass

                    markets.append(Market(
                        condition_id=m.get("conditionId", ""),
                        question=m.get("question", ""),
                        slug=m.get("slug", ""),
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=vol,
                        liquidity=liq,
                        category=m.get("category", ""),
                        yes_token_id=yes_token,
                        no_token_id=no_token,
                        tick_size=str(m.get("minimum_tick_size", 0.01)),
                        neg_risk=m.get("negRisk", False),
                        min_order_size=float(m.get("minimum_order_size", 5)),
                        end_date=end_date,
                    ))

                except Exception as e:
                    logger.debug(f"Failed to parse market: {e}")
                    continue

            # Sort by volume (most liquid first)
            markets.sort(key=lambda m: m.volume, reverse=True)

            logger.info(f"📊 Found {len(markets)} markets with vol≥${min_volume:,.0f}, liq≥${min_liquidity:,.0f}")

            return markets[:limit]

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_market_by_slug(self, slug: str) -> Optional[Market]:
        """Fetch a single market by slug"""
        try:
            # Rate limiting for Gamma API
            await rate_limiter.wait_for_gamma_markets()

            url = f"{self.gamma_url}/markets"
            params = {"slug": slug}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()

            if not data:
                return None

            m = data[0] if isinstance(data, list) else data

            prices = m.get("outcomePrices", [0.5, 0.5])
            clob_tokens = m.get("clobTokenIds", ["", ""])

            return Market(
                condition_id=m.get("conditionId", ""),
                question=m.get("question", ""),
                slug=m.get("slug", ""),
                yes_price=float(prices[0]) if prices else 0.5,
                no_price=float(prices[1]) if len(prices) > 1 else 0.5,
                volume=float(m.get("volume", 0) or 0),
                liquidity=float(m.get("liquidity", 0) or 0),
                category=m.get("category", ""),
                yes_token_id=clob_tokens[0] if clob_tokens else "",
                no_token_id=clob_tokens[1] if len(clob_tokens) > 1 else "",
                tick_size=str(m.get("minimum_tick_size", 0.01)),
                neg_risk=m.get("negRisk", False),
                min_order_size=float(m.get("minimum_order_size", 5)),
                end_date=m.get("endDate", ""),
            )
        except Exception as e:
            logger.error(f"Failed to fetch market {slug}: {e}")
            return None

    # =========================================
    # CLOB API - Order Book Data (No Auth)
    # =========================================

    def get_order_book(self, token_id: str) -> Dict:
        """Get order book for a token"""
        if not self.read_client:
            return {}
        try:
            return self.read_client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return {}

    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get best price for a token"""
        if not self.read_client:
            return None
        try:
            return self.read_client.get_price(token_id, side=side)
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token"""
        if not self.read_client:
            return None
        try:
            return self.read_client.get_midpoint(token_id)
        except Exception as e:
            logger.error(f"Failed to get midpoint: {e}")
            return None

    def get_spread(self, token_id: str) -> Dict:
        """Get bid-ask spread for a token"""
        if not self.read_client:
            return {}
        try:
            return self.read_client.get_spread(token_id)
        except Exception as e:
            logger.error(f"Failed to get spread: {e}")
            return {}

    # =========================================
    # CLOB API - Trading (Requires Auth)
    # =========================================

    async def place_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        tick_size: str = "0.01",
        neg_risk: bool = False
    ) -> Dict:
        """
        Place a limit order (GTC - Good Till Cancelled).

        Args:
            token_id: YES or NO token ID from market.yes_token_id or market.no_token_id
            price: Price per share (0.01 to 0.99)
            size: Number of shares
            side: "BUY" or "SELL"
            tick_size: Market's tick size (from market.tick_size)
            neg_risk: Market's neg_risk flag (from market.neg_risk)

        Returns:
            Order response dict
        """
        if not self.trade_client:
            logger.error("Trading not initialized")
            return {"error": "Trading not initialized - check wallet credentials"}

        try:
            # Rate limiting for CLOB order endpoint
            await rate_limiter.wait_for_clob_order()
            # Validate price
            price = round(price, 2)
            if not 0.01 <= price <= 0.99:
                return {"error": f"Price {price} out of range (0.01-0.99)"}

            # Create order args
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY if side.upper() == "BUY" else SELL
            )

            # Create signed order
            signed_order = self.trade_client.create_order(order_args)

            # Post to CLOB
            response = self.trade_client.post_order(
                signed_order,
                OrderType.GTC,
                tick_size=tick_size,
                neg_risk=neg_risk
            )

            logger.success(f"✅ Limit order: {side} {size} shares @ ${price}")
            return response

        except Exception as e:
            logger.error(f"❌ Limit order failed: {e}")
            return {"error": str(e)}

    async def place_market_order(
        self,
        token_id: str,
        amount: float,
        side: str,
        tick_size: str = "0.01",
        neg_risk: bool = False
    ) -> Dict:
        """
        Place a market order (FOK - Fill Or Kill).

        Args:
            token_id: YES or NO token ID
            amount: Dollar amount to spend
            side: "BUY" or "SELL"
            tick_size: Market's tick size
            neg_risk: Market's neg_risk flag

        Returns:
            Order response dict
        """
        if not self.trade_client:
            logger.error("Trading not initialized")
            return {"error": "Trading not initialized - check wallet credentials"}

        try:
            # Rate limiting for CLOB order endpoint
            await rate_limiter.wait_for_clob_order()

            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=BUY if side.upper() == "BUY" else SELL
            )

            signed_order = self.trade_client.create_market_order(order_args)

            response = self.trade_client.post_order(
                signed_order,
                OrderType.FOK,
                tick_size=tick_size,
                neg_risk=neg_risk
            )

            logger.success(f"✅ Market order: {side} ${amount}")
            return response

        except Exception as e:
            logger.error(f"❌ Market order failed: {e}")
            return {"error": str(e)}

    async def place_order_for_market(
        self,
        market: Market,
        direction: str,
        size_usd: float,
        order_type: str = "limit"
    ) -> Dict:
        """
        Convenience method to place order using Market object.

        Args:
            market: Market dataclass with token IDs and settings
            direction: "YES" or "NO"
            size_usd: Dollar amount
            order_type: "limit" or "market"
        """
        # Get correct token ID
        if direction.upper() == "YES":
            token_id = market.yes_token_id
            price = market.yes_price
        else:
            token_id = market.no_token_id
            price = market.no_price

        if not token_id:
            return {"error": f"No token ID for {direction}"}

        # Calculate shares from USD amount
        shares = size_usd / price if price > 0 else 0

        if order_type == "market":
            return await self.place_market_order(
                token_id=token_id,
                amount=size_usd,
                side="BUY",
                tick_size=market.tick_size,
                neg_risk=market.neg_risk
            )
        else:
            return await self.place_limit_order(
                token_id=token_id,
                price=price,
                size=shares,
                side="BUY",
                tick_size=market.tick_size,
                neg_risk=market.neg_risk
            )

    # =========================================
    # Account & Positions
    # =========================================

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        if not self.trade_client:
            return []
        try:
            return self.trade_client.get_orders()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order by ID"""
        if not self.trade_client:
            return {"error": "Trading not initialized"}
        try:
            return self.trade_client.cancel(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return {"error": str(e)}

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        if not self.trade_client:
            return {"error": "Trading not initialized"}
        try:
            return self.trade_client.cancel_all()
        except Exception as e:
            logger.error(f"Failed to cancel all: {e}")
            return {"error": str(e)}


# Singleton instance
polymarket = PolymarketClient()


def run_self_tests():
    """Run self-tests for Polymarket client functions."""
    print("🧪 Polymarket Client Self-Tests")

    try:
        # Test Market dataclass
        market = Market(
            condition_id="test-123",
            question="Will it rain tomorrow?",
            slug="will-it-rain-tomorrow",
            yes_price=0.6,
            no_price=0.4,
            volume=1000,
            liquidity=500,
            category="weather",
            yes_token_id="yes-token",
            no_token_id="no-token"
        )

        # Test that dataclass works
        assert market.condition_id == "test-123", f"Expected condition_id 'test-123', got {market.condition_id}"
        assert market.question == "Will it rain tomorrow?", f"Unexpected question: {market.question}"
        assert market.yes_price == 0.6, f"Expected yes_price 0.6, got {market.yes_price}"
        assert market.no_price == 0.4, f"Expected no_price 0.4, got {market.no_price}"
        assert market.volume == 1000, f"Expected volume 1000, got {market.volume}"
        assert market.liquidity == 500, f"Expected liquidity 500, got {market.liquidity}"
        assert market.spread == 0.0, f"Expected spread 0.0, got {market.spread}"

        # Test price validation (should be within bounds)
        assert 0.0 <= market.yes_price <= 1.0, f"YES price should be bounded, got {market.yes_price}"
        assert 0.0 <= market.no_price <= 1.0, f"NO price should be bounded, got {market.no_price}"

        print("✅ Polymarket Client self-tests passed")

    except Exception as e:
        print(f"❌ Polymarket Client self-test failed: {e}")
        raise


if __name__ == "__main__":
    run_self_tests()
