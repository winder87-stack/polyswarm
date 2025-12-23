"""
External Odds / Consensus Detector - FREE Sources Only

Provides real-time consensus from FREE prediction platforms.
No API keys required for any functionality.
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz

from loguru import logger


@dataclass
class ExternalProbability:
    """Probability estimate from an external source."""
    source: str  # "predictit", "metaculus"
    market_name: str
    probability: float
    last_updated: datetime
    url: Optional[str] = None


@dataclass
class AggregatedProbability:
    """Aggregated probabilities from multiple sources."""
    probabilities: Dict[str, float]  # source -> probability
    weighted_average: float
    simple_average: float
    median: float
    std_dev: float
    sources_count: int
    polymarket_vs_consensus: float  # PM price - consensus (positive = PM overvalued)
    confidence_score: float  # Overall confidence in consensus
    last_updated: datetime = field(default_factory=datetime.now)


class PredictItFetcher:
    """
    PredictIt Public API - FREE, no auth required.
    https://www.predictit.org/api/marketdata/all/
    """

    API_URL = "https://www.predictit.org/api/marketdata/all/"

    async def fetch_markets(self) -> List[Dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.API_URL, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        markets = []

                        for market in data.get("markets", []):
                            for contract in market.get("contracts", []):
                                markets.append({
                                    "question": f"{market.get('name', '')} - {contract.get('name', '')}",
                                    "probability": contract.get("lastTradePrice", 0.5),
                                    "source": "predictit",
                                    "url": market.get("url", ""),
                                })

                        logger.debug(f"PredictIt: fetched {len(markets)} contracts")
                        return markets
        except Exception as e:
            logger.warning(f"PredictIt fetch failed: {e}")

        return []


class MetaculusFetcher:
    """
    Metaculus Public API - FREE, no auth required.
    Community predictions on various topics.
    """

    API_URL = "https://www.metaculus.com/api2/questions/"

    async def fetch_markets(self) -> List[Dict]:
        try:
            params = {
                "limit": 50,
                "status": "open",
                "type": "forecast",
                "order_by": "-activity",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.API_URL, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        markets = []

                        for q in data.get("results", []):
                            prediction = q.get("community_prediction", {})
                            if prediction and prediction.get("full"):
                                prob = prediction["full"].get("q2", 0.5)  # Median
                                markets.append({
                                    "question": q.get("title", ""),
                                    "probability": prob,
                                    "source": "metaculus",
                                    "url": f"https://www.metaculus.com/questions/{q.get('id')}/",
                                })

                        logger.debug(f"Metaculus: fetched {len(markets)} questions")
                        return markets
        except Exception as e:
            logger.warning(f"Metaculus fetch failed: {e}")

        return []


class ConsensusDetector:
    """
    Find where Polymarket diverges from external consensus.
    Uses FREE sources only - no API keys needed.

    Sources:
    - PredictIt (free public API)
    - Metaculus (free public API)
    """

    SOURCE_WEIGHTS = {
        "predictit": 1.0,
        "metaculus": 0.9,
    }

    def __init__(self) -> None:
        self.fetchers = {
            "predictit": PredictItFetcher(),
            "metaculus": MetaculusFetcher(),
        }
        self._cache = {}
        self._cache_time = None
        logger.info("📊 ConsensusDetector initialized (FREE sources only)")

    async def _refresh_cache(self):
        """Refresh external markets cache"""
        now = datetime.now()

        # Cache for 5 minutes
        if self._cache_time and (now - self._cache_time).seconds < 300:
            return

        self._cache = {}

        for name, fetcher in self.fetchers.items():
            try:
                markets = await fetcher.fetch_markets()
                self._cache[name] = markets
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")

        self._cache_time = now

    async def get_external_consensus(self, market) -> Dict:
        """
        Get external probabilities for a Polymarket market.

        Returns:
        {
            "sources": {"predictit": 0.65, "metaculus": 0.70},
            "weighted_average": 0.67,
            "sources_count": 2,
            "divergence": 0.07  # vs Polymarket
        }
        """
        await self._refresh_cache()

        sources = {}

        for source_name, source_markets in self._cache.items():
            match = self._find_matching_market(market.question, source_markets)
            if match:
                sources[source_name] = match["probability"]

        if not sources:
            return {
                "sources": {},
                "weighted_average": None,
                "sources_count": 0,
                "divergence": 0,
            }

        # Calculate weighted average
        total_weight = sum(self.SOURCE_WEIGHTS.get(s, 1.0) for s in sources)
        weighted_avg = sum(
            prob * self.SOURCE_WEIGHTS.get(source, 1.0)
            for source, prob in sources.items()
        ) / total_weight

        return {
            "sources": sources,
            "weighted_average": weighted_avg,
            "sources_count": len(sources),
            "divergence": weighted_avg - market.yes_price,
        }

    def _find_matching_market(self, pm_question: str, external_markets: List[Dict]) -> Optional[Dict]:
        """Find matching market using fuzzy string matching"""
        best_match = None
        best_score = 0

        pm_clean = pm_question.lower().strip()

        for ext in external_markets:
            ext_question = ext.get("question", "").lower().strip()
            score = fuzz.token_set_ratio(pm_clean, ext_question)

            if score > best_score and score >= 60:  # 60% similarity threshold
                best_score = score
                best_match = ext

        return best_match

    async def find_divergences(
        self,
        markets: List,
        min_divergence: float = 0.08
    ) -> List[Dict]:
        """Find markets where Polymarket differs from consensus"""
        divergences = []

        for market in markets:
            consensus = await self.get_external_consensus(market)

            if consensus["sources_count"] >= 1:  # At least 1 external source
                div = abs(consensus.get("divergence", 0))

                if div >= min_divergence:
                    divergences.append({
                        "market": market,
                        "polymarket_price": market.yes_price,
                        "external_consensus": consensus["weighted_average"],
                        "divergence": consensus["divergence"],
                        "direction": "YES" if consensus["divergence"] > 0 else "NO",
                        "sources": list(consensus["sources"].keys()),
                    })

        return sorted(divergences, key=lambda x: abs(x["divergence"]), reverse=True)


