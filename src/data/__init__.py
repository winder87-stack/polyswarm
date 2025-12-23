"""
Data Management Package

Handles historical data collection, storage, and retrieval
for backtesting and analysis.
"""

from .historical_collector import (
    HistoricalMarket,
    PriceSnapshot,
    TradeRecord,
    HistoricalDataCollector,
    DataCollectionScheduler,
    create_historical_collector,
    create_collection_scheduler,
)

__all__ = [
    "HistoricalMarket",
    "PriceSnapshot",
    "TradeRecord",
    "HistoricalDataCollector",
    "DataCollectionScheduler",
    "create_historical_collector",
    "create_collection_scheduler",
]
