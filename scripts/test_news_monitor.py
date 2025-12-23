#!/usr/bin/env python3
"""
Test News Monitoring System

Demonstrates the real-time news monitoring capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from termcolor import colored, cprint
import logging

from src.services import create_news_aggregator, NEWS_CONFIG
from src.agents import SwarmAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_news_monitoring():
    """Test the news monitoring system."""
    cprint("📰 Testing News Monitoring System", "cyan", attrs=["bold"])
    cprint("=" * 50, "cyan")

    # Create news aggregator
    try:
        aggregator = create_news_aggregator()
        cprint("✅ News aggregator created", "green")
    except Exception as e:
        cprint(f"❌ Failed to create news aggregator: {e}", "red")
        return

    # Test keywords for different categories
    test_keywords = {
        "crypto": ["bitcoin", "ethereum", "crypto", "blockchain"],
        "politics": ["election", "president", "government", "policy"],
        "sports": ["game", "score", "championship", "player"],
        "business": ["market", "economy", "company", "stock"]
    }

    cprint("\n🔍 Testing News Fetching", "yellow", attrs=["bold"])

    for category, keywords in test_keywords.items():
        cprint(f"\n📊 Testing {category.upper()} news:", "blue")
        try:
            news_items = await aggregator.get_breaking_news(keywords, categories=[category])
            cprint(f"  ✅ Fetched {len(news_items)} relevant items", "green")

            # Show top 2 items
            for i, item in enumerate(news_items[:2]):
                cprint(f"  {i+1}. {item.headline[:60]}...", "white")
                cprint(f"     Source: {item.source} | Relevance: {item.relevance_score:.2f}", "white")

        except Exception as e:
            cprint(f"  ❌ Error fetching {category} news: {e}", "red")

    cprint("\n🤖 Testing AI News Analysis", "yellow", attrs=["bold"])

    # Create mock market and news for testing
    class MockMarket:
        def __init__(self) -> None:
            self.question = "Will Bitcoin reach $100,000 by end of 2025?"
            self.yes_price = 0.35
            self.no_price = 0.65
            self.liquidity = 50000
            self.volume = 25000
            self.slug = "bitcoin-100k-2025"

    class MockNews:
        def __init__(self) -> None:
            self.headline = "Bitcoin surges past $90,000 on ETF approval news"
            self.summary = "Major financial institutions announce Bitcoin ETF approvals, causing price to jump 15% in hours."
            self.source = "Crypto News"
            self.timestamp = __import__('datetime').datetime.now()
            self.category = "crypto"
            self.relevance_score = 0.95

    try:
        from src.agents.trading_swarm import TradingSwarm
        swarm = TradingSwarm()
        market = MockMarket()
        news = MockNews()

        cprint("  📈 Analyzing market with breaking news...", "blue")
        analysis = await swarm.analyze_market_with_news(market, news)

        cprint("  ✅ News analysis completed", "green")
        cprint(f"     Edge: {analysis['edge']:+.1%}", "white")
        cprint(f"     Confidence: {analysis['confidence']:.1%}", "white")
        cprint(f"     Fast Trade: {'YES' if analysis['fast_trade_recommended'] else 'NO'}", "white")

    except Exception as e:
        cprint(f"  ❌ AI news analysis failed: {e}", "red")

    cprint("\n📋 News Monitoring Configuration", "yellow", attrs=["bold"])
    cprint(f"  Check Interval: {NEWS_CONFIG['check_interval_seconds']} seconds", "white")
    cprint(f"  Breaking News Threshold: {NEWS_CONFIG['breaking_news_threshold']:.1%}", "white")
    cprint(f"  Fast Trade Min Edge: {NEWS_CONFIG['fast_trade_min_edge']:.1%}", "white")
    cprint(f"  Max News Age: {NEWS_CONFIG['max_news_age_minutes']} minutes", "white")

    cprint("\n🎯 News Sources Configured:", "yellow", attrs=["bold"])
    for source in aggregator.sources:
        cprint(f"  • {source.name} ({', '.join(source.categories)})", "white")

    cprint("\n✅ News Monitoring System Test Complete!", "green", attrs=["bold"])

def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(test_news_monitoring())
    except KeyboardInterrupt:
        cprint("\n👋 Test interrupted by user", "yellow")
    except Exception as e:
        cprint(f"❌ Test failed: {e}", "red")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
