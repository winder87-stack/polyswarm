"""
News and External Services Package

Provides real-time news monitoring and external data integration
for the Polymarket trading bot.
"""

from .news_monitor import (
    NewsItem,
    NewsSource,
    GoogleNewsSource,
    RSSFeedSource,
    NewsAggregator,
)

__all__ = [
    "NewsItem",
    "NewsSource",
    "GoogleNewsSource",
    "RSSFeedSource",
    "NewsAggregator",
]
