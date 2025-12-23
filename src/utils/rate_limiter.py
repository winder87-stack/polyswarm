"""
Rate Limiter for Polymarket API
Based on official rate limits from docs.polymarket.com
"""

import asyncio
from datetime import datetime, timedelta
from collections import deque
from typing import Dict
from loguru import logger


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.
    """

    def __init__(self, max_requests: int, time_window_seconds: int, name: str = ""):
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.name = name
        self.requests = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request slot is available"""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.time_window)

            # Remove old requests outside the window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # If at limit, wait
            if len(self.requests) >= self.max_requests:
                oldest = self.requests[0]
                wait_time = (oldest + timedelta(seconds=self.time_window) - now).total_seconds()
                if wait_time > 0:
                    logger.debug(f"Rate limit ({self.name}): waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Clean up after waiting
                    now = datetime.now()
                    cutoff = now - timedelta(seconds=self.time_window)
                    while self.requests and self.requests[0] < cutoff:
                        self.requests.popleft()

            # Record this request
            self.requests.append(datetime.now())

    @property
    def current_usage(self) -> int:
        """Current number of requests in window"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        return sum(1 for r in self.requests if r >= cutoff)


class PolymarketRateLimiter:
    """
    Rate limiters for all Polymarket endpoints.
    Based on official documentation.
    """

    def __init__(self):
        # Gamma API limits
        self.gamma_general = RateLimiter(750, 10, "gamma_general")
        self.gamma_markets = RateLimiter(125, 10, "gamma_markets")
        self.gamma_events = RateLimiter(100, 10, "gamma_events")

        # CLOB API limits
        self.clob_general = RateLimiter(5000, 10, "clob_general")
        self.clob_book = RateLimiter(200, 10, "clob_book")
        self.clob_price = RateLimiter(200, 10, "clob_price")
        self.clob_order = RateLimiter(2400, 10, "clob_order")  # Burst limit

        logger.info("🚦 Rate limiters initialized")

    async def wait_for_gamma_markets(self):
        """Wait for gamma /markets endpoint"""
        await self.gamma_general.acquire()
        await self.gamma_markets.acquire()

    async def wait_for_gamma_events(self):
        """Wait for gamma /events endpoint"""
        await self.gamma_general.acquire()
        await self.gamma_events.acquire()

    async def wait_for_clob_book(self):
        """Wait for CLOB /book endpoint"""
        await self.clob_general.acquire()
        await self.clob_book.acquire()

    async def wait_for_clob_price(self):
        """Wait for CLOB /price endpoint"""
        await self.clob_general.acquire()
        await self.clob_price.acquire()

    async def wait_for_clob_order(self):
        """Wait for CLOB /order endpoint"""
        await self.clob_general.acquire()
        await self.clob_order.acquire()

    def get_status(self) -> Dict[str, int]:
        """Get current usage across all limiters"""
        return {
            "gamma_markets": self.gamma_markets.current_usage,
            "clob_book": self.clob_book.current_usage,
            "clob_order": self.clob_order.current_usage,
        }


# Singleton instance
rate_limiter = PolymarketRateLimiter()
