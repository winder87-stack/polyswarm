"""
Category Specialist - Specialized trading strategies per market category

Handles category-specific model weights, thresholds, news sources, and trading rules
for different types of prediction markets (politics, sports, crypto, etc.)
"""

import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from loguru import logger

from src.connectors import Market
import pandas as pd

from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.connectors import Market
except ImportError:
    # For testing without full imports
    class Market:
        pass


@dataclass
class CategoryConfig:
    """Configuration for a specific market category."""
    name: str
    enabled: bool

    # Model weights override for this category
    model_weights: Dict[str, float]

    # Thresholds override
    min_edge: float
    min_confidence: float
    max_position_size: float

    # Category-specific data sources
    news_sources: List[str]
    external_odds_sources: List[str]

    # Trading rules
    max_exposure_pct: float  # Max % of bankroll in this category
    preferred_time_to_expiry: Tuple[int, int]  # (min_days, max_days)

    # Performance tracking
    historical_win_rate: float = 0.0
    historical_roi: float = 0.0
    notes: str = ""


# Default category configurations
CATEGORY_CONFIGS = {
    "default": CategoryConfig(
        name="Default",
        enabled=True,
        model_weights={
            "claude": 1.0,
            "gemini": 1.0,
            "gpt": 1.0,
            "perplexity": 1.0,
            "deepseek": 1.0,
        },
        min_edge=0.08,
        min_confidence=0.5,
        max_position_size=100,
        news_sources=["google_news", "twitter"],
        external_odds_sources=["predictit"],
        max_exposure_pct=50,
        preferred_time_to_expiry=(1, 365),
        notes="Fallback configuration for unrecognized categories"
    ),

    "politics": CategoryConfig(
        name="Politics",
        enabled=True,
        model_weights={
            "claude": 1.3,      # Best reasoning for complex political analysis
            "gemini": 1.2,      # Strong analytical capabilities
            "gpt": 1.2,         # Good at political discourse analysis
            "perplexity": 1.5,  # Real-time news crucial for politics
            "deepseek": 0.8,    # Reduce weight - politics needs current info
        },
        min_edge=0.08,
        min_confidence=0.6,     # Higher confidence needed due to volatility
        max_position_size=75,
        news_sources=["google_news", "twitter", "fivethirtyeight", "realclearpolitics", "politico"],
        external_odds_sources=["predictit", "betfair", "kalshi", "metaculus"],
        max_exposure_pct=30,
        preferred_time_to_expiry=(7, 60),  # 1 week to 2 months
        notes="Political markets often mispriced during news events. High news impact."
    ),

    "sports": CategoryConfig(
        name="Sports",
        enabled=True,
        model_weights={
            "claude": 1.0,
            "gemini": 1.0,
            "gpt": 1.3,         # Good at sports statistics and analysis
            "perplexity": 1.3,  # Injury reports and breaking news
            "deepseek": 1.0,
        },
        min_edge=0.10,  # Higher edge needed - sports markets are efficient
        min_confidence=0.7,  # Higher confidence threshold
        max_position_size=50,
        news_sources=["espn", "sports_twitter", "injury_reports", "team_news", "odds_sharks"],
        external_odds_sources=["betfair", "draftkings", "fanduel", "pinnacle", "williamhill"],
        max_exposure_pct=25,
        preferred_time_to_expiry=(1, 14),  # 1 day to 2 weeks (shorter for sports)
        notes="Sports markets are highly efficient. Focus on statistical analysis and injury news."
    ),

    "crypto": CategoryConfig(
        name="Crypto",
        enabled=True,
        model_weights={
            "claude": 1.2,
            "gemini": 1.0,
            "gpt": 1.0,
            "perplexity": 1.4,  # Real-time crypto news and whale movements
            "deepseek": 1.1,    # Good at technical analysis
        },
        min_edge=0.12,  # Higher edge needed due to extreme volatility
        min_confidence=0.65,
        max_position_size=40,
        news_sources=["crypto_twitter", "coingecko", "whale_alert", "on_chain_metrics", "glassnode"],
        external_odds_sources=["kalshi", "polymarket_crypto"],
        max_exposure_pct=20,
        preferred_time_to_expiry=(3, 30),  # 3 days to 1 month
        notes="Crypto markets are highly volatile. Focus on on-chain metrics and whale movements."
    ),

    "entertainment": CategoryConfig(
        name="Entertainment",
        enabled=False,  # Disabled by default - hard to predict
        model_weights={
            "claude": 1.0,
            "gemini": 1.0,
            "gpt": 1.2,         # Good at pop culture knowledge
            "perplexity": 1.2,  # Current entertainment news
            "deepseek": 1.0,
        },
        min_edge=0.15,  # Very high edge needed - entertainment is unpredictable
        min_confidence=0.75,  # Very high confidence threshold
        max_position_size=30,
        news_sources=["variety", "hollywood_reporter", "entertainment_twitter", "box_office_mojo"],
        external_odds_sources=["predictit"],
        max_exposure_pct=10,
        preferred_time_to_expiry=(1, 90),
        notes="Entertainment markets are highly unpredictable. Use extreme caution."
    ),

    "science": CategoryConfig(
        name="Science",
        enabled=True,
        model_weights={
            "claude": 1.4,      # Best at technical and scientific analysis
            "gemini": 1.3,      # Strong analytical capabilities
            "gpt": 1.2,         # Good at scientific reasoning
            "perplexity": 1.1,  # Research paper updates
            "deepseek": 1.0,
        },
        min_edge=0.10,
        min_confidence=0.7,     # High confidence needed for scientific predictions
        max_position_size=60,
        news_sources=["arxiv", "nature", "science_news", "pubmed", "scientific_american"],
        external_odds_sources=["metaculus", "predictit"],
        max_exposure_pct=20,
        preferred_time_to_expiry=(14, 90),  # 2 weeks to 3 months (science takes time)
        notes="Science markets require deep technical understanding. Focus on peer-reviewed research."
    ),

    "finance": CategoryConfig(
        name="Finance",
        enabled=True,
        model_weights={
            "claude": 1.3,      # Strong at financial analysis
            "gemini": 1.2,
            "gpt": 1.3,         # Good at market analysis
            "perplexity": 1.2,  # Real-time economic news
            "deepseek": 1.0,
        },
        min_edge=0.12,  # Higher edge needed - finance markets are efficient
        min_confidence=0.7,
        max_position_size=45,
        news_sources=["bloomberg", "reuters", "wsj", "cnbc", "federal_reserve"],
        external_odds_sources=["predictit", "kalshi"],
        max_exposure_pct=25,
        preferred_time_to_expiry=(1, 180),  # 1 day to 6 months
        notes="Finance markets are highly efficient. Focus on fundamental analysis and economic indicators."
    ),

    "weather": CategoryConfig(
        name="Weather",
        enabled=True,
        model_weights={
            "claude": 1.2,
            "gemini": 1.0,
            "gpt": 1.0,
            "perplexity": 1.4,  # Real-time weather data and forecasts
            "deepseek": 1.0,
        },
        min_edge=0.08,
        min_confidence=0.6,
        max_position_size=80,
        news_sources=["noaa", "weather_channel", "accuweather", "storm_tracker"],
        external_odds_sources=["predictit"],
        max_exposure_pct=15,
        preferred_time_to_expiry=(1, 30),  # Short-term weather predictions
        notes="Weather markets can be predicted with meteorological data. Focus on forecast accuracy."
    ),

    "technology": CategoryConfig(
        name="Technology",
        enabled=True,
        model_weights={
            "claude": 1.3,
            "gemini": 1.2,
            "gpt": 1.2,
            "perplexity": 1.4,  # Tech news and product launches
            "deepseek": 1.0,
        },
        min_edge=0.10,
        min_confidence=0.65,
        max_position_size=55,
        news_sources=["techcrunch", "wired", "the_verge", "ars_technica", "github"],
        external_odds_sources=["predictit", "metaculus"],
        max_exposure_pct=25,
        preferred_time_to_expiry=(7, 90),
        notes="Technology markets move fast. Monitor product launches and industry trends."
    )
}


class CategorySpecialist:
    """
    Manages category-specific trading strategies and configurations.

    Provides specialized model weights, thresholds, and trading rules
    for different market categories (politics, sports, crypto, etc.)
    """

    def __init__(self, configs: Optional[Dict[str, CategoryConfig]] = None, config_file: str = "config/categories.yaml") -> None:
        """Initialize category specialist with configurations."""
        self.configs = configs or CATEGORY_CONFIGS.copy()
        self.config_file = config_file
        self.category_stats = {}  # Track performance per category

        # Load from YAML if it exists
        self._load_from_yaml()

        logger.info(f"🎯 Category specialist initialized with {len(self.configs)} categories")

    def _load_from_yaml(self):
        """Load category configurations from YAML file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)

                if yaml_data:
                    for category_name, config_data in yaml_data.items():
                        if category_name in self.configs:
                            # Update existing config with YAML values
                            config = self.configs[category_name]
                            for key, value in config_data.items():
                                if hasattr(config, key):
                                    if key == 'preferred_time_to_expiry':
                                        # Handle nested dict format
                                        if isinstance(value, dict):
                                            min_days = value.get('min_days', 1)
                                            max_days = value.get('max_days', 365)
                                            setattr(config, key, (min_days, max_days))
                                        else:
                                            setattr(config, key, value)
                                    else:
                                        setattr(config, key, value)
                    logger.info(f"✅ Loaded category configs from {self.config_file}")
                else:
                    logger.warning(f"Empty or invalid YAML file: {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not load category config from {self.config_file}: {e}")

    def save_to_yaml(self):
        """Save current configurations to YAML file."""
        try:
            # Convert configs to dict for YAML serialization
            yaml_data = {}
            for category_name, config in self.configs.items():
                yaml_data[category_name] = {
                    "enabled": config.enabled,
                    "model_weights": config.model_weights,
                    "min_edge": config.min_edge,
                    "min_confidence": config.min_confidence,
                    "max_position_size": config.max_position_size,
                    "news_sources": config.news_sources,
                    "external_odds_sources": config.external_odds_sources,
                    "max_exposure_pct": config.max_exposure_pct,
                    "preferred_time_to_expiry": list(config.preferred_time_to_expiry),
                    "notes": config.notes
                }

            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            with open(self.config_file, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            logger.info(f"💾 Saved category configs to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save category configs: {e}")

    def get_config(self, market: Market) -> CategoryConfig:
        """Get category config for a market, with fallback to default."""
        category = getattr(market, 'category', 'unknown')
        if category:
            category = category.lower().strip()

        # Try exact match first
        if category in self.configs:
            return self.configs[category]

        # Try partial matches for common variations
        for config_name, config in self.configs.items():
            if config_name != "default":
                if config_name in category or category in config_name:
                    logger.info(f"🎯 Matched '{category}' to category '{config_name}'")
                    return config

        # Fallback to default
        logger.info(f"🎯 Using default config for category '{category}'")
        return self.configs["default"]

    def should_trade_category(self, market: Market, current_exposure: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """
        Check if we should trade this category at all.

        Args:
            market: Market to check
            current_exposure: Dict of category -> current exposure %

        Returns:
            (should_trade, reason)
        """
        config = self.get_config(market)

        if not config.enabled:
            return False, f"Category '{config.name}' is disabled"

        # Check category exposure limits
        if current_exposure:
            category_key = getattr(market, 'category', 'unknown').lower()
            current_pct = current_exposure.get(category_key, 0.0)
            if current_pct >= config.max_exposure_pct:
                return False, f"Category exposure limit reached ({current_pct:.1f}% >= {config.max_exposure_pct}%)"

        # Check historical performance (optional filter)
        if hasattr(config, 'historical_win_rate') and config.historical_win_rate < 0.3:
            logger.warning(f"⚠️ Low historical win rate for {config.name}: {config.historical_win_rate:.1%}")

        return True, "OK"

    def get_adjusted_weights(self, market: Market) -> Dict[str, float]:
        """Get model weights adjusted for this category."""
        config = self.get_config(market)
        return config.model_weights.copy()

    def get_adjusted_thresholds(self, market: Market) -> Dict[str, float]:
        """Get trading thresholds adjusted for this category."""
        config = self.get_config(market)
        return {
            "min_edge": config.min_edge,
            "min_confidence": config.min_confidence,
            "max_position_size": config.max_position_size,
        }

    def get_news_sources(self, market: Market) -> List[str]:
        """Get relevant news sources for this category."""
        config = self.get_config(market)
        return config.news_sources.copy()

    def get_external_odds_sources(self, market: Market) -> List[str]:
        """Get relevant external odds sources for this category."""
        config = self.get_config(market)
        return config.external_odds_sources.copy()

    def check_time_to_expiry(self, market: Market) -> Tuple[bool, str]:
        """
        Check if market expiry time fits category preferences.

        Returns:
            (is_preferred, reason)
        """
        config = self.get_config(market)

        try:
            end_date = getattr(market, 'end_date', None)
            if not end_date:
                return True, "No expiry date available"

            # Convert to datetime if needed
            if isinstance(end_date, str):
                try:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except ValueError as e:
                    logger.debug(f"Could not parse expiry date: {e}")
                    return True, "Could not parse expiry date"

            now = datetime.now()
            days_until_expiry = (end_date - now).days

            min_days, max_days = config.preferred_time_to_expiry

            if days_until_expiry < min_days:
                return False, f"Too soon to expiry ({days_until_expiry} < {min_days} days)"
            elif days_until_expiry > max_days:
                return False, f"Too far from expiry ({days_until_expiry} > {max_days} days)"
            else:
                return True, f"Preferred expiry window ({min_days}-{max_days} days)"

        except Exception as e:
            return True, f"Expiry check failed: {e}"

    def update_category_stats(self, market: Market, pnl: float, position_size: float):
        """Track performance per category."""
        category = getattr(market, 'category', 'unknown').lower()

        if category not in self.category_stats:
            self.category_stats[category] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "total_size": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            }

        stats = self.category_stats[category]
        stats["trades"] += 1
        stats["total_pnl"] += pnl
        stats["total_size"] += position_size

        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        # Recalculate derived stats
        if stats["trades"] > 0:
            stats["win_rate"] = stats["wins"] / stats["trades"]

        if stats["wins"] > 0:
            total_wins = sum(pnl for pnl in [pnl] if pnl > 0)  # This is simplified
            stats["avg_win"] = total_wins / stats["wins"]

        if stats["losses"] > 0:
            total_losses = sum(abs(pnl) for pnl in [pnl] if pnl < 0)
            stats["avg_loss"] = total_losses / stats["losses"]

        if stats["avg_loss"] > 0:
            stats["profit_factor"] = (stats["avg_win"] * stats["win_rate"]) / (stats["avg_loss"] * (1 - stats["win_rate"]))

    def get_category_report(self) -> pd.DataFrame:
        """Get performance breakdown by category."""
        if not self.category_stats:
            return pd.DataFrame()

        data = []
        for category, stats in self.category_stats.items():
            data.append({
                "category": category,
                "trades": stats["trades"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": stats["win_rate"],
                "total_pnl": stats["total_pnl"],
                "total_size": stats["total_size"],
                "avg_win": stats["avg_win"],
                "avg_loss": stats["avg_loss"],
                "profit_factor": stats["profit_factor"]
            })

        df = pd.DataFrame(data)
        df = df.sort_values("total_pnl", ascending=False)
        return df

    def get_enabled_categories(self) -> List[str]:
        """Get list of enabled categories."""
        return [name for name, config in self.configs.items() if config.enabled and name != "default"]

    def disable_category(self, category: str):
        """Disable trading for a category."""
        if category in self.configs:
            self.configs[category].enabled = False
            logger.info(f"🚫 Disabled category: {category}")
        else:
            logger.warning(f"Category not found: {category}")

    def enable_category(self, category: str):
        """Enable trading for a category."""
        if category in self.configs:
            self.configs[category].enabled = True
            logger.info(f"✅ Enabled category: {category}")
        else:
            logger.warning(f"Category not found: {category}")

    def update_config(self, category: str, **kwargs):
        """Update configuration for a category."""
        if category not in self.configs:
            logger.warning(f"Category not found: {category}")
            return

        config = self.configs[category]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"🔧 Updated {category}.{key} = {value}")
            else:
                logger.warning(f"Invalid config key: {key}")

    def __str__(self) -> str:
        """String representation of category specialist."""
        enabled = [name for name, config in self.configs.items() if config.enabled]
        return f"CategorySpecialist(categories={len(self.configs)}, enabled={len(enabled)})"


# Global instance
category_specialist = CategorySpecialist()
