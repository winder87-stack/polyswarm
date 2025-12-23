"""
Analysis Package

Provides historical pattern analysis and market insights
for improved trading strategies.
"""

from .pattern_analyzer import (
    PatternAnalyzer,
    create_pattern_analyzer,
)

__all__ = [
    "PatternAnalyzer",
    "create_pattern_analyzer",
]
