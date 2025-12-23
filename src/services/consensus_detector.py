"""
Consensus Signal Detector

This module fetches probability data from external prediction platforms
to identify when Polymarket prices diverge from broader consensus.

IMPORTANT:
- External platforms are READ-ONLY data sources
- NO trades are placed on external platforms
- ALL trading happens ONLY on Polymarket
- No accounts required for most external data (public APIs)

Data Sources:
- PredictIt: Public API, no auth
- Metaculus: Public API, no auth
- FiveThirtyEight: Web scraping, no auth
- Betfair: Requires API key (optional)

Usage:
- Find markets where Polymarket differs from external consensus
- Generate signals for POLYMARKET trading only
- Combine AI analysis with external consensus for better signals
"""

import asyncio
import aiohttp
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import re

from loguru import logger

from src.connectors import Market
from fuzzywuzzy import fuzz

try:
    from src.connectors import Market
except ImportError:
    logger.warning("Could not import aiohttp - consensus detection disabled")


@dataclass
class ExternalProbability:
    """Read-only probability from external source."""
    source: str  # "predictit", "betfair", "538", "metaculus"
    probability: float  # 0-1 probability
    confidence: float  # 0-1 reliability score
    last_updated: datetime
    url: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ConsensusSignal:
    """Signal when Polymarket differs from external consensus."""
    market: Any  # Polymarket market
    polymarket_price: float  # Current Polymarket YES price
    external_consensus: float  # Weighted average from external sources
    divergence: float  # external_consensus - polymarket_price
    direction: str  # "YES" or "NO" to buy on Polymarket
    sources_agreeing: List[str]  # Which external sources agree
    confidence: float  # 0-1 confidence in the signal
    sources_count: int
    created_at: datetime = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


class PredictItFetcher:
    """
    Fetch from PredictIt public API.
    API: https://www.predictit.org/api/marketdata/all/
    FREE, no auth required, READ-ONLY.
    """

    def __init__(self) -> None:
        self.base_url = "https://www.predictit.org/api/marketdata/all/"
        self.session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Polymarket-Trading-Bot/1.0',
                    'Accept': 'application/json'
                }
            )
        return self.session

    async def fetch_markets(self) -> List[Dict]:
        """
        Fetch all active markets from PredictIt.
        Returns list of market dictionaries with questions and probabilities.
        """
        try:
            session = await self._get_session()
            async with session.get(self.base_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"PredictIt API error: {response.status}")
                    return []

                data = await response.json()

            markets = []
            for market in data.get('markets', []):
                try:
                    parsed = self._parse_predictit_market(market)
                    if parsed:
                        markets.append(parsed)
                except Exception as e:
                    logger.debug(f"Failed to parse PredictIt market {market.get('id')}: {e}")

            logger.info(f"✅ PredictIt: Fetched {len(markets)} markets")
            return markets

        except Exception as e:
            logger.error(f"PredictIt fetch failed: {e}")
            return []

    def _parse_predictit_market(self, market: Dict) -> Optional[Dict]:
        """Parse PredictIt market into standardized format."""
        try:
            market_name = market.get('name', '').strip()
            if not market_name:
                return None

            contracts = market.get('contracts', [])
            if len(contracts) != 2:  # Should be YES/NO
                return None

            # Find YES and NO contracts
            yes_contract = None
            no_contract = None

            for contract in contracts:
                name = contract.get('name', '').lower()
                if 'yes' in name or 'will happen' in name:
                    yes_contract = contract
                elif 'no' in name or 'will not happen' in name:
                    no_contract = contract

            if not yes_contract or not no_contract:
                return None

            # Get prices (PredictIt prices are 0-1 representing $0-$1)
            yes_price = yes_contract.get('lastTradePrice', 0.5)
            no_price = no_contract.get('lastTradePrice', 0.5)

            # Validate prices
            if not (0.01 <= yes_price <= 0.99) or not (0.01 <= no_price <= 0.99):
                return None

            return {
                "source": "predictit",
                "question": market_name,
                "probability": yes_price,  # Probability is the price
                "confidence": 0.85,  # PredictIt is generally reliable for politics
                "url": f"https://www.predictit.org/markets/detail/{market.get('id')}",
                "volume": sum(c.get('totalSharesTraded', 0) for c in contracts),
                "last_updated": datetime.now()
            }

        except Exception as e:
            logger.debug(f"Failed to parse PredictIt market: {e}")
            return None

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


class MetaculusFetcher:
    """
    Fetch from Metaculus public API.
    FREE, no auth, READ-ONLY.
    """

    def __init__(self) -> None:
        self.base_url = "https://www.metaculus.com/api2"
        self.session = None

    async def fetch_markets(self) -> List[Dict]:
        """Fetch questions from Metaculus API."""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/questions/"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"Metaculus API error: {response.status}")
                    return []

                data = await response.json()

            markets = []
            for question in data.get('results', []):
                try:
                    parsed = self._parse_metaculus_question(question)
                    if parsed:
                        markets.append(parsed)
                except Exception as e:
                    logger.debug(f"Failed to parse Metaculus question {question.get('id')}: {e}")

            logger.info(f"✅ Metaculus: Fetched {len(markets)} questions")
            return markets

        except Exception as e:
            logger.error(f"Metaculus fetch failed: {e}")
            return []

    def _parse_metaculus_question(self, question: Dict) -> Optional[Dict]:
        """Parse Metaculus question into standardized format."""
        try:
            title = question.get('title', '').strip()
            if not title:
                return None

            # Get community prediction
            community_prediction = question.get('community_prediction', {})
            if not community_prediction:
                return None

            # Extract probability (Metaculus uses different formats)
            prob_data = community_prediction.get('full', {}).get('q2', {})
            probability = prob_data.get('quantile', 0.5)  # Use median as default

            # Skip if probability is invalid
            if not (0.01 <= probability <= 0.99):
                return None

            return {
                "source": "metaculus",
                "question": title,
                "probability": probability,
                "confidence": 0.8,  # Metaculus has good track record
                "url": f"https://www.metaculus.com/questions/{question.get('id')}",
                "last_updated": datetime.now()
            }

        except Exception as e:
            logger.debug(f"Failed to parse Metaculus question: {e}")
            return None

    async def _get_session(self):
        """Get aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Polymarket-Trading-Bot/1.0',
                    'Accept': 'application/json'
                }
            )
        return self.session

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


class FiveThirtyEightFetcher:
    """
    Scrape 538 election/sports forecasts.
    Public data, READ-ONLY.
    """

    def __init__(self) -> None:
        self.base_url = "https://projects.fivethirtyeight.com"
        self.session = None

    async def fetch_markets(self) -> List[Dict]:
        """Fetch forecasts from 538."""
        # Implementation would require web scraping
        # For now, return empty list as placeholder
        logger.info("ℹ️ 538 fetcher not fully implemented (requires scraping)")
        return []

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


class ConsensusDetector:
    """
    Find where Polymarket price diverges from external consensus.

    IMPORTANT: This ONLY generates signals for Polymarket trading.
    External platforms are READ-ONLY data sources.
    """

    # How much to trust each source (based on historical accuracy)
    SOURCE_WEIGHTS = {
        "predictit": 0.9,   # Retail predictions, decent accuracy
        "betfair": 1.2,     # Sharp money, highly accurate
        "metaculus": 0.8,   # Community predictions, variable quality
        "538": 1.1,         # Expert forecasters, very accurate
        "kalshi": 0.9,      # Similar to Polymarket, good reference
    }

    def __init__(self) -> None:
        """Initialize consensus detector."""
        self.fetchers = {
            "predictit": PredictItFetcher(),
            "metaculus": MetaculusFetcher(),
            "538": FiveThirtyEightFetcher(),
        }

        logger.info("🎯 Consensus detector initialized (READ-ONLY external data)")

    async def get_external_consensus(self, market: Market) -> Dict:
        """
        Fetch probabilities from external sources for a market.
        Uses fuzzy matching to find equivalent markets.

        Returns:
        {
            "sources": {"predictit": 0.68, "betfair": 0.71, ...},
            "weighted_average": 0.69,
            "simple_average": 0.68,
            "sources_count": 3,
        }
        """
        try:
            # Fetch from all sources concurrently
            external_markets = await self._fetch_all_external_markets()

            # Match this Polymarket market to external equivalents
            matched_probabilities = {}

            for source, markets in external_markets.items():
                match = self.match_market_to_external(market, markets)
                if match:
                    probability = match["probability"]
                    confidence = match["confidence"]
                    matched_probabilities[source] = {
                        "probability": probability,
                        "confidence": confidence,
                        "url": match.get("url")
                    }

            if not matched_probabilities:
                return {
                    "sources": {},
                    "weighted_average": 0.5,
                    "simple_average": 0.5,
                    "sources_count": 0
                }

            # Calculate consensus
            sources = {k: v["probability"] for k, v in matched_probabilities.items()}

            # Weighted average using source weights
            total_weight = 0
            weighted_sum = 0

            for source, prob in sources.items():
                weight = self.SOURCE_WEIGHTS.get(source, 1.0)
                weighted_sum += prob * weight
                total_weight += weight

            weighted_average = weighted_sum / total_weight if total_weight > 0 else sum(sources.values()) / len(sources)

            # Simple average
            simple_average = sum(sources.values()) / len(sources)

            return {
                "sources": sources,
                "weighted_average": weighted_average,
                "simple_average": simple_average,
                "sources_count": len(sources)
            }

        except Exception as e:
            logger.error(f"Failed to get external consensus: {e}")
            return {
                "sources": {},
                "weighted_average": 0.5,
                "simple_average": 0.5,
                "sources_count": 0
            }

    def match_market_to_external(
        self,
        pm_market: Market,
        external_markets: List[Dict]
    ) -> Optional[Dict]:
        """
        Match a Polymarket market to equivalent external market.
        Uses fuzzy string matching on question text.

        Returns matched external market dict or None.
        """
        try:
            pm_question = getattr(pm_market, 'question', '').lower().strip()

            best_match = None
            best_score = 0

            for ext_market in external_markets:
                ext_question = ext_market.get('question', '').lower().strip()

                # Multiple fuzzy matching approaches
                ratio_score = fuzz.ratio(pm_question, ext_question)
                partial_score = fuzz.partial_ratio(pm_question, ext_question)
                token_score = fuzz.token_sort_ratio(pm_question, ext_question)

                # Use the best score
                final_score = max(ratio_score, partial_score, token_score)

                if final_score > best_score and final_score > 70:  # 70% similarity threshold
                    best_score = final_score
                    best_match = ext_market

            if best_match:
                logger.debug(f"🎯 Matched PM '{pm_question[:50]}...' to {best_match['source']} (score: {best_score})")

            return best_match

        except Exception as e:
            logger.error(f"Market matching failed: {e}")
            return None

    async def find_divergences(
        self,
        markets: List[Market],
        min_divergence: float = 0.08  # 8% minimum divergence
    ) -> List[ConsensusSignal]:
        """
        Find markets where Polymarket differs from external consensus.

        Returns list of signals for POLYMARKET trading only.
        """
        try:
            signals = []

            # Process markets in batches to avoid overwhelming APIs
            batch_size = 10
            for i in range(0, len(markets), batch_size):
                batch = markets[i:i + batch_size]

                # Get consensus for this batch
                tasks = [self.get_external_consensus(market) for market in batch]
                consensuses = await asyncio.gather(*tasks)

                for market, consensus in zip(batch, consensuses):
                    if consensus["sources_count"] < 2:
                        continue  # Need at least 2 external sources for reliable signal

                    pm_price = getattr(market, 'yes_price', 0.5)
                    external_consensus = consensus["weighted_average"]

                    divergence = external_consensus - pm_price

                    if abs(divergence) >= min_divergence:
                        signal = ConsensusSignal(
                            market=market,
                            polymarket_price=pm_price,
                            external_consensus=external_consensus,
                            divergence=divergence,
                            direction="YES" if divergence > 0 else "NO",
                            sources_agreeing=list(consensus["sources"].keys()),
                            confidence=min(consensus["sources_count"] / 5, 1.0),  # More sources = higher confidence
                            sources_count=consensus["sources_count"]
                        )
                        signals.append(signal)

                # Small delay between batches to be respectful to APIs
                await asyncio.sleep(0.5)

            # Sort by divergence magnitude (largest first)
            signals.sort(key=lambda s: abs(s.divergence), reverse=True)

            total_volume = sum(getattr(s.market, 'volume', 0) for s in signals)
            logger.info(f"🎯 Found {len(signals)} consensus divergence signals (Polymarket trading only) - Total Volume: ${total_volume:,.0f}")

            return signals

        except Exception as e:
            logger.error(f"Failed to find divergences: {e}")
            return []

    async def _fetch_all_external_markets(self) -> Dict[str, List[Dict]]:
        """Fetch markets from all external sources concurrently."""
        try:
            # Run all fetchers concurrently
            tasks = [fetcher.fetch_markets() for fetcher in self.fetchers.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            external_markets = {}
            for source_name, result in zip(self.fetchers.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to fetch from {source_name}: {result}")
                    external_markets[source_name] = []
                else:
                    external_markets[source_name] = result

            return external_markets

        except Exception as e:
            logger.error(f"Failed to fetch external markets: {e}")
            return {}

    async def close(self):
        """Close all fetcher sessions."""
        for fetcher in self.fetchers.values():
            if hasattr(fetcher, 'close'):
                await fetcher.close()


# Global instance
consensus_detector = ConsensusDetector()
