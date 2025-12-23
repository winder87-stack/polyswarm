"""
Trading Signals - Shared signal definitions to avoid circular imports
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class TradingSignal:
    """Represents a trading signal with all analysis results."""

    # Core market data
    market: Any  # Market object (avoiding circular import)

    # Signal direction and strength
    direction: str  # "YES" or "NO"
    confidence: float  # 0-1, how confident we are
    probability: float  # AI-estimated probability of YES outcome
    market_probability: float  # Current market-implied probability
    edge: float  # Expected edge (probability difference)
    expected_value: float  # Expected value per dollar risked

    # Analysis details
    reasoning: str  # Why we're taking this position
    model_votes: Dict[str, float]  # Individual model predictions
    model_weights: Dict[str, float]  # Model confidence weights
    consensus_summary: str  # Summary of model consensus

    # News and external data
    news_context: str = ""  # Relevant news context
    news_items: Optional[List[Any]] = None  # Raw news items
    news_impact_score: float = 0.0  # Impact of news on probability

    # Contrarian signals
    contrarian_signals: Optional[List[Any]] = None  # Contrarian analysis results

    # Breaking news flag
    is_breaking_news_trade: bool = False  # Whether this is triggered by breaking news

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional analysis details (may be added by different analysis methods)
    analysis_details: Optional[Dict[str, Any]] = None
    analysis_methods: Optional[List[str]] = None

    @property
    def is_actionable(self) -> bool:
        """Check if this signal is actionable (meets minimum thresholds)."""
        return (
            abs(self.edge) >= 0.08 and  # Minimum 8% edge
            self.confidence >= 0.6 and  # Minimum 60% confidence
            self.expected_value >= 0.05  # Minimum $0.05 expected value
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "market_id": getattr(self.market, 'condition_id', 'unknown') if self.market else 'unknown',
            "direction": self.direction,
            "confidence": self.confidence,
            "probability": self.probability,
            "market_probability": self.market_probability,
            "edge": self.edge,
            "expected_value": self.expected_value,
            "reasoning": self.reasoning,
            "model_votes": self.model_votes,
            "model_weights": self.model_weights,
            "consensus_summary": self.consensus_summary,
            "news_context": self.news_context,
            "news_impact_score": self.news_impact_score,
            "is_breaking_news_trade": self.is_breaking_news_trade,
            "timestamp": self.timestamp.isoformat(),
        }
