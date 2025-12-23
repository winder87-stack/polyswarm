"""
Historical Data Collector for Polymarket

Collects and stores historical market data, price histories, and trade records
for backtesting and analysis purposes.
"""

import asyncio
import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import aiohttp

from loguru import logger

# Data Models
@dataclass
class HistoricalMarket:
    """Represents a resolved historical market."""
    condition_id: str
    question: str
    category: str
    created_at: datetime
    resolved_at: datetime
    resolution: str  # "YES", "NO", "INVALID"
    final_yes_price: float
    final_no_price: float
    total_volume: float
    total_liquidity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalMarket':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return cls(**data)


@dataclass
class PriceSnapshot:
    """Represents a historical price snapshot."""
    condition_id: str
    timestamp: datetime
    yes_price: float
    no_price: float
    volume_24h: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PriceSnapshot':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class TradeRecord:
    """Represents a historical trade record."""
    condition_id: str
    timestamp: datetime
    side: str  # "BUY", "SELL"
    outcome: str  # "YES", "NO"
    price: float
    size: float
    is_taker: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class HistoricalDataCollector:
    """Collects and stores historical Polymarket data in SQLite."""

    def __init__(self, db_path: str = "data/historical.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Markets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS markets (
                    condition_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolution TEXT,
                    final_yes_price REAL,
                    final_no_price REAL,
                    total_volume REAL,
                    total_liquidity REAL,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Price snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    yes_price REAL,
                    no_price REAL,
                    volume_24h REAL,
                    FOREIGN KEY (condition_id) REFERENCES markets (condition_id),
                    UNIQUE(condition_id, timestamp)
                )
            ''')

            # Trade records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    condition_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    side TEXT,
                    outcome TEXT,
                    price REAL,
                    size REAL,
                    is_taker BOOLEAN,
                    FOREIGN KEY (condition_id) REFERENCES markets (condition_id)
                )
            ''')

            # Collection metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            logger.info(f"✅ Initialized historical database at {self.db_path}")

    async def collect_resolved_markets(self, days_back: int = 365) -> int:
        """
        Fetch resolved markets from Polymarket API.

        Args:
            days_back: Number of days to look back for resolved markets

        Returns:
            Number of new markets collected
        """
        logger.info(f"🔍 Collecting resolved markets from last {days_back} days")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        collected_count = 0

        try:
            # Fetch resolved markets from Gamma API
            async with aiohttp.ClientSession() as session:
                # Note: This is a simplified implementation
                # In reality, you'd need to paginate through the Gamma API
                # and handle rate limiting

                url = "https://gamma-api.polymarket.com/query"
                query = """
                query GetResolvedMarkets($endDate: String!) {
                    markets(where: {closed: true, endDate_lt: $endDate}, first: 1000) {
                        id
                        question
                        category
                        createdAt
                        endDate
                        resolution
                        volume
                        liquidity
                        yesPrice
                        noPrice
                    }
                }
                """

                variables = {"endDate": end_date.isoformat()}

                async with session.post(url, json={"query": query, "variables": variables}, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        data = await response.json()

                        markets_data = data.get('data', {}).get('markets', [])

                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()

                            for market_data in markets_data:
                                try:
                                    # Convert timestamps
                                    created_at = datetime.fromisoformat(market_data['createdAt'].replace('Z', '+00:00'))
                                    resolved_at = datetime.fromisoformat(market_data['endDate'].replace('Z', '+00:00'))

                                    # Only collect if resolved within our date range
                                    if resolved_at >= start_date:
                                        market = HistoricalMarket(
                                            condition_id=market_data['id'],
                                            question=market_data['question'],
                                            category=market_data.get('category', 'unknown'),
                                            created_at=created_at,
                                            resolved_at=resolved_at,
                                            resolution=market_data.get('resolution', 'UNKNOWN'),
                                            final_yes_price=float(market_data.get('yesPrice', 0)),
                                            final_no_price=float(market_data.get('noPrice', 0)),
                                            total_volume=float(market_data.get('volume', 0)),
                                            total_liquidity=float(market_data.get('liquidity', 0))
                                        )

                                        # Insert or update market
                                        cursor.execute('''
                                            INSERT OR REPLACE INTO markets
                                            (condition_id, question, category, created_at, resolved_at,
                                             resolution, final_yes_price, final_no_price, total_volume, total_liquidity)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (
                                            market.condition_id, market.question, market.category,
                                            market.created_at, market.resolved_at, market.resolution,
                                            market.final_yes_price, market.final_no_price,
                                            market.total_volume, market.total_liquidity
                                        ))

                                        collected_count += 1

                                        # Collect price history for this market
                                        await self._collect_market_price_history(market.condition_id, market.created_at, market.resolved_at)

                                except Exception as e:
                                    logger.warning(f"Error processing market {market_data.get('id', 'unknown')}: {e}")
                                    continue

                            conn.commit()

                        logger.info(f"✅ Collected {collected_count} resolved markets")
                        return collected_count

                    else:
                        logger.error(f"❌ Failed to fetch resolved markets: HTTP {response.status}")
                        return 0

        except Exception as e:
            logger.error(f"❌ Error collecting resolved markets: {e}")
            return 0

    async def _collect_market_price_history(self, condition_id: str, start_date: datetime, end_date: datetime):
        """Collect price history for a specific market."""
        try:
            logger.debug(f"📊 Collecting price history for {condition_id}")

            # This is a simplified implementation
            # In reality, you'd need to fetch price data from appropriate APIs
            # For now, we'll create placeholder price snapshots

            # Generate hourly snapshots from creation to resolution
            current_time = start_date
            snapshots = []

            while current_time <= end_date:
                # Placeholder: in real implementation, fetch actual price data
                # For demo purposes, we'll create synthetic data
                snapshot = PriceSnapshot(
                    condition_id=condition_id,
                    timestamp=current_time,
                    yes_price=0.5,  # Placeholder
                    no_price=0.5,   # Placeholder
                    volume_24h=1000.0  # Placeholder
                )
                snapshots.append(snapshot)

                # Next hour
                current_time += timedelta(hours=1)

            # Store snapshots in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for snapshot in snapshots:
                    cursor.execute('''
                        INSERT OR IGNORE INTO price_snapshots
                        (condition_id, timestamp, yes_price, no_price, volume_24h)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        snapshot.condition_id, snapshot.timestamp,
                        snapshot.yes_price, snapshot.no_price, snapshot.volume_24h
                    ))

                conn.commit()

            logger.debug(f"✅ Stored {len(snapshots)} price snapshots for {condition_id}")

        except Exception as e:
            logger.error(f"❌ Error collecting price history for {condition_id}: {e}")

    async def backfill_all_data(self) -> Dict[str, int]:
        """
        Perform full historical data backfill.

        Returns:
            Dictionary with collection statistics
        """
        logger.info("🚀 Starting full historical data backfill")

        stats = {
            "markets_collected": 0,
            "price_snapshots": 0,
            "start_time": datetime.now()
        }

        try:
            # Collect resolved markets from the past year
            markets_count = await self.collect_resolved_markets(days_back=365)
            stats["markets_collected"] = markets_count

            # Update metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO collection_metadata (key, value)
                    VALUES (?, ?)
                ''', ("last_full_backfill", datetime.now().isoformat()))

                # Count total price snapshots
                cursor.execute("SELECT COUNT(*) FROM price_snapshots")
                stats["price_snapshots"] = cursor.fetchone()[0]

                conn.commit()

            duration = datetime.now() - stats["start_time"]
            logger.info("✅ Full backfill completed!")
            logger.info(f"   Markets: {stats['markets_collected']}")
            logger.info(f"   Price snapshots: {stats['price_snapshots']}")
            logger.info(f"   Duration: {duration.total_seconds():.2f}s")
            return stats

        except Exception as e:
            logger.error(f"❌ Full backfill failed: {e}")
            return stats

    def get_resolved_markets(
        self,
        category: Optional[str] = None,
        min_volume: float = 0,
        resolution: Optional[str] = None,
        limit: int = 1000
    ) -> List[HistoricalMarket]:
        """
        Query resolved markets with filters.

        Args:
            category: Filter by category (optional)
            min_volume: Minimum volume threshold
            resolution: Filter by resolution outcome (optional)
            limit: Maximum number of results

        Returns:
            List of HistoricalMarket objects
        """
        query = """
            SELECT condition_id, question, category, created_at, resolved_at,
                   resolution, final_yes_price, final_no_price, total_volume, total_liquidity
            FROM markets
            WHERE total_volume >= ?
        """

        params = [min_volume]

        if category:
            query += " AND category = ?"
            params.append(category)

        if resolution:
            query += " AND resolution = ?"
            params.append(resolution)

        query += " ORDER BY resolved_at DESC LIMIT ?"
        params.append(limit)

        markets = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                for row in cursor.fetchall():
                    market = HistoricalMarket(
                        condition_id=row[0],
                        question=row[1],
                        category=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        resolved_at=datetime.fromisoformat(row[4]),
                        resolution=row[5],
                        final_yes_price=row[6],
                        final_no_price=row[7],
                        total_volume=row[8],
                        total_liquidity=row[9]
                    )
                    markets.append(market)

        except Exception as e:
            logger.error(f"❌ Error querying resolved markets: {e}")

        return markets

    def get_price_history(self, condition_id: str) -> List[PriceSnapshot]:
        """
        Get price history for a specific market.

        Args:
            condition_id: Market condition ID

        Returns:
            List of PriceSnapshot objects
        """
        snapshots = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT condition_id, timestamp, yes_price, no_price, volume_24h
                    FROM price_snapshots
                    WHERE condition_id = ?
                    ORDER BY timestamp ASC
                ''', (condition_id,))

                for row in cursor.fetchall():
                    snapshot = PriceSnapshot(
                        condition_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        yes_price=row[2],
                        no_price=row[3],
                        volume_24h=row[4]
                    )
                    snapshots.append(snapshot)

        except Exception as e:
            logger.error(f"❌ Error getting price history for {condition_id}: {e}")

        return snapshots

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get database collection statistics."""
        stats = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Count markets
                cursor.execute("SELECT COUNT(*) FROM markets")
                stats["total_markets"] = cursor.fetchone()[0]

                # Count price snapshots
                cursor.execute("SELECT COUNT(*) FROM price_snapshots")
                stats["total_price_snapshots"] = cursor.fetchone()[0]

                # Count trade records
                cursor.execute("SELECT COUNT(*) FROM trade_records")
                stats["total_trade_records"] = cursor.fetchone()[0]

                # Get latest collection times
                cursor.execute("SELECT value FROM collection_metadata WHERE key = 'last_full_backfill'")
                result = cursor.fetchone()
                stats["last_full_backfill"] = result[0] if result else None

                # Get market categories
                cursor.execute("SELECT category, COUNT(*) FROM markets GROUP BY category")
                stats["markets_by_category"] = dict(cursor.fetchall())

                # Get resolution outcomes
                cursor.execute("SELECT resolution, COUNT(*) FROM markets GROUP BY resolution")
                stats["resolutions"] = dict(cursor.fetchall())

        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")

        return stats


class DataCollectionScheduler:
    """Scheduler for ongoing historical data collection."""

    def __init__(self, collector: HistoricalDataCollector) -> None:
        self.collector = collector

    async def daily_collection_job(self) -> Dict[str, int]:
        """
        Daily job to collect newly resolved markets and update data.

        Returns:
            Dictionary with collection statistics
        """
        logger.info("📅 Running daily historical data collection")

        stats = {
            "new_markets": 0,
            "start_time": datetime.now()
        }

        try:
            # Collect markets resolved in the last 24 hours
            new_markets = await self.collector.collect_resolved_markets(days_back=1)
            stats["new_markets"] = new_markets

            # Update metadata
            with sqlite3.connect(self.collector.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO collection_metadata (key, value)
                    VALUES (?, ?)
                ''', ("last_daily_collection", datetime.now().isoformat()))
                conn.commit()

            duration = datetime.now() - stats["start_time"]
            logger.info("✅ Daily collection completed!")
            logger.info(f"   New markets: {stats['new_markets']}")
            logger.info(f"   Duration: {duration.total_seconds():.2f}s")
            return stats

        except Exception as e:
            logger.error(f"❌ Daily collection failed: {e}")
            return stats


# Convenience functions
def create_historical_collector(db_path: str = "data/historical.db") -> HistoricalDataCollector:
    """Create and return a HistoricalDataCollector instance."""
    return HistoricalDataCollector(db_path)

def create_collection_scheduler(collector: HistoricalDataCollector) -> DataCollectionScheduler:
    """Create and return a DataCollectionScheduler instance."""
    return DataCollectionScheduler(collector)
