"""
News Monitor - FREE Sources Only

Provides real-time news monitoring using only FREE sources.
No API keys required for any functionality.
"""

import asyncio
import aiohttp
import feedparser
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from abc import ABC, abstractmethod

from loguru import logger


@dataclass
class NewsItem:
    """Represents a news item from any source."""
    headline: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    category: str
    relevance_score: float = 0.5


class NewsSource(ABC):
    """Abstract base class for news sources."""
    @abstractmethod
    async def fetch_latest(self, keywords: List[str] = None) -> List[NewsItem]:
        """Fetch latest news items from this source."""
        pass


class GoogleNewsSource(NewsSource):
    """Google News RSS - FREE, no API key needed"""

    BASE_URL = "https://news.google.com/rss"

    CATEGORY_FEEDS = {
        "politics": "https://news.google.com/rss/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNRFZ4ZERBU0FtVnVLQUFQAQ",
        "business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        "sports": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB",
        "technology": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
        "science": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB",
    }

    async def fetch_latest(self, keywords: List[str] = None, category: str = None) -> List[NewsItem]:
        items = []

        try:
            if keywords:
                # Search by keywords
                query = "+".join(keywords)
                url = f"{self.BASE_URL}/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            elif category and category.lower() in self.CATEGORY_FEEDS:
                url = self.CATEGORY_FEEDS[category.lower()]
            else:
                url = f"{self.BASE_URL}?hl=en-US&gl=US&ceid=US:en"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)

                        for entry in feed.entries[:20]:
                            items.append(NewsItem(
                                headline=entry.get("title", ""),
                                summary=entry.get("summary", "")[:500],
                                source="Google News",
                                url=entry.get("link", ""),
                                timestamp=datetime.now(),
                                category=category or "general",
                            ))
        except Exception as e:
            logger.warning(f"Google News fetch failed: {e}")

        return items


class RSSFeedSource(NewsSource):
    """Generic RSS Feed Source - FREE"""

    # Free RSS feeds for different categories
    FEEDS = {
        "politics": [
            "https://feeds.npr.org/1014/rss.xml",  # NPR Politics
            "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",  # NYT Politics
        ],
        "sports": [
            "https://www.espn.com/espn/rss/news",  # ESPN
        ],
        "crypto": [
            "https://cointelegraph.com/rss",  # CoinTelegraph
        ],
        "business": [
            "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg
        ],
    }

    async def fetch_latest(self, keywords: List[str] = None, category: str = None) -> List[NewsItem]:
        items = []
        feeds = self.FEEDS.get(category, []) if category else []

        # Add all feeds if no category specified
        if not feeds:
            for feed_list in self.FEEDS.values():
                feeds.extend(feed_list)

        async with aiohttp.ClientSession() as session:
            for feed_url in feeds[:5]:  # Limit to 5 feeds
                try:
                    async with session.get(feed_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)

                            for entry in feed.entries[:10]:
                                items.append(NewsItem(
                                    headline=entry.get("title", ""),
                                    summary=entry.get("summary", "")[:500],
                                    source=feed.feed.get("title", "RSS"),
                                    url=entry.get("link", ""),
                                    timestamp=datetime.now(),
                                    category=category or "general",
                                ))
                except Exception as e:
                    logger.debug(f"RSS fetch failed for {feed_url}: {e}")

        return items


class NewsAggregator:
    """
    Aggregates news from FREE sources only.
    No API keys required.
    """

    def __init__(self) -> None:
        self.sources = [
            GoogleNewsSource(),
            RSSFeedSource(),
        ]
        self.seen_headlines = set()
        logger.info("📰 NewsAggregator initialized (FREE sources only)")

    async def get_recent_news(
        self,
        market=None,
        category: str = None,
        keywords: List[str] = None,
        limit: int = 20
    ) -> List[NewsItem]:
        """Fetch news from all sources"""

        # Extract keywords from market if provided
        if market and not keywords:
            keywords = self._extract_keywords(market.question)
            category = category or market.category

        all_items = []

        for source in self.sources:
            try:
                items = await source.fetch_latest(keywords=keywords, category=category)
                all_items.extend(items)
            except Exception as e:
                logger.warning(f"Source failed: {e}")

        # Deduplicate
        unique_items = []
        for item in all_items:
            if item.headline not in self.seen_headlines:
                self.seen_headlines.add(item.headline)
                unique_items.append(item)

        # Sort by relevance if we have keywords
        if keywords:
            unique_items = self._rank_by_relevance(unique_items, keywords)

        return unique_items[:limit]

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from market question"""
        # Remove common words
        stopwords = {"will", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "be", "is", "are", "was", "were", "?", "by"}
        words = question.lower().split()
        keywords = [w.strip("?.,!") for w in words if w.lower() not in stopwords and len(w) > 2]
        return keywords[:5]  # Top 5 keywords

    def _rank_by_relevance(self, items: List[NewsItem], keywords: List[str]) -> List[NewsItem]:
        """Rank items by keyword relevance"""
        def score(item):
            text = (item.headline + " " + item.summary).lower()
            return sum(1 for kw in keywords if kw.lower() in text)

        return sorted(items, key=score, reverse=True)
