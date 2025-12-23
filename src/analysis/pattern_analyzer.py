"""
Pattern Analyzer for Historical Trading Data

Analyzes resolved market data to identify profitable trading patterns,
calibration issues, and market inefficiencies.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from loguru import logger

from src.data.historical_collector import HistoricalDataCollector, HistoricalMarket


class PatternAnalyzer:
    """
    Analyzes historical market data to identify trading patterns and edges.

    Uses resolved market data to find statistically significant patterns
    that can be exploited for profitable trading strategies.
    """

    def __init__(self, db_path: str = "data/historical.db") -> None:
        """Initialize pattern analyzer with historical data connection."""
        self.db = HistoricalDataCollector(db_path)
        self._load_data()

    def _load_data(self):
        """Load and cache historical market data."""
        logger.info("Loading historical market data for analysis...")
        self.markets = self.db.get_resolved_markets()
        self.price_data = {}

        # Load price histories for significant markets
        significant_markets = [m for m in self.markets if m.total_volume > 10000]

        for market in significant_markets[:100]:  # Limit for performance
            prices = self.db.get_price_history(market.condition_id)
            if prices:
                self.price_data[market.condition_id] = prices

        logger.info(f"Loaded {len(self.markets)} markets and {len(self.price_data)} price histories")

    def analyze_category_accuracy(self) -> pd.DataFrame:
        """
        Analyze accuracy and profitability by category.

        Returns DataFrame with:
        - Category name
        - Total markets
        - YES win rate (%)
        - Average volume
        - Average final price
        - Market efficiency score
        - Historical edge opportunities
        """
        logger.info("Analyzing category accuracy patterns...")

        if not self.markets:
            return pd.DataFrame()

        # Group markets by category
        categories = {}
        for market in self.markets:
            cat = market.category or 'unknown'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(market)

        results = []
        for category, markets in categories.items():
            if len(markets) < 5:  # Skip categories with too few markets
                continue

            # Calculate YES win rate
            yes_markets = [m for m in markets if m.resolution == 'YES']
            yes_win_rate = len(yes_markets) / len(markets)

            # Average metrics
            avg_volume = np.mean([m.total_volume for m in markets])
            avg_final_price = np.mean([m.final_yes_price for m in markets])

            # Market efficiency score (how well prices predicted outcomes)
            # Lower score = more efficient (prices closer to 0/1 for clear outcomes)
            final_prices = [m.final_yes_price for m in markets]
            efficiency_score = np.mean([
                abs(price - (1 if resolution == 'YES' else 0))
                for price, resolution in zip(final_prices, [m.resolution for m in markets])
            ])

            # Edge opportunities (markets where final price was far from 0.5)
            edge_opportunities = len([m for m in markets
                                    if abs(m.final_yes_price - 0.5) > 0.3])

            results.append({
                'category': category,
                'total_markets': len(markets),
                'yes_win_rate': yes_win_rate,
                'avg_volume': avg_volume,
                'avg_final_price': avg_final_price,
                'efficiency_score': efficiency_score,
                'edge_opportunities': edge_opportunities,
                'edge_opportunity_rate': edge_opportunities / len(markets)
            })

        df = pd.DataFrame(results).sort_values('total_markets', ascending=False)
        logger.info(f"Analyzed {len(df)} categories")
        return df

    def analyze_price_patterns(self) -> Dict[str, Any]:
        """
        Analyze price trajectory patterns in resolved markets.

        Returns dictionary with pattern statistics about:
        - Price movement patterns
        - Final day volatility
        - Overshoot and reversion patterns
        """
        logger.info("Analyzing price trajectory patterns...")

        patterns = {
            'total_markets_with_price_data': len(self.price_data),
            'price_flip_analysis': {},
            'final_day_movements': {},
            'overshoot_patterns': {}
        }

        if not self.price_data:
            logger.warning("No price history data available for pattern analysis")
            return patterns

        # Analyze price flips (crossing 50% threshold)
        flip_analysis = {
            'started_below_50': 0,
            'ended_above_50': 0,
            'flipped_to_yes': 0,
            'flipped_to_no': 0
        }

        # Analyze final 24h movements
        final_movements = []

        for condition_id, prices in self.price_data.items():
            if len(prices) < 2:
                continue

            # Get market resolution
            market = next((m for m in self.markets if m.condition_id == condition_id), None)
            if not market:
                continue

            # Sort prices by time
            sorted_prices = sorted(prices, key=lambda x: x.timestamp)
            start_price = sorted_prices[0].yes_price
            end_price = sorted_prices[-1].yes_price

            # Price flip analysis
            started_below_50 = start_price < 0.5
            ended_above_50 = end_price >= 0.5

            if started_below_50 and ended_above_50:
                flip_analysis['flipped_to_yes'] += 1
            elif not started_below_50 and not ended_above_50:
                flip_analysis['flipped_to_no'] += 1

            flip_analysis['started_below_50'] += int(started_below_50)

            # Final 24h movement analysis
            if len(sorted_prices) > 1:
                # Get prices from last 24h
                cutoff_time = sorted_prices[-1].timestamp - timedelta(hours=24)
                recent_prices = [p for p in sorted_prices if p.timestamp >= cutoff_time]

                if len(recent_prices) >= 2:
                    final_movement = abs(recent_prices[-1].yes_price - recent_prices[0].yes_price)
                    final_movements.append(final_movement)

        patterns['price_flip_analysis'] = flip_analysis
        patterns['final_day_movements'] = {
            'count': len(final_movements),
            'avg_movement': np.mean(final_movements) if final_movements else 0,
            'max_movement': max(final_movements) if final_movements else 0,
            'large_moves_10pct': len([m for m in final_movements if m > 0.1])
        }

        logger.info(f"Analyzed price patterns for {patterns['total_markets_with_price_data']} markets")
        return patterns

    def analyze_volume_signals(self) -> Dict[str, Any]:
        """
        Analyze volume patterns and their relationship to market outcomes.

        Returns dictionary with volume pattern insights.
        """
        logger.info("Analyzing volume signal patterns...")

        volume_analysis = {
            'total_markets': len(self.markets),
            'volume_distribution': {},
            'outcome_by_volume_quartile': {},
            'volume_vs_accuracy': {}
        }

        if not self.markets:
            return volume_analysis

        # Volume distribution analysis
        volumes = [m.total_volume for m in self.markets]
        if volumes:
            volume_analysis['volume_distribution'] = {
                'min': min(volumes),
                'max': max(volumes),
                'median': np.median(volumes),
                'mean': np.mean(volumes),
                'quartiles': {
                    'q1': np.percentile(volumes, 25),
                    'q2': np.percentile(volumes, 50),
                    'q3': np.percentile(volumes, 75)
                }
            }

            # Analyze outcomes by volume quartile
            df = pd.DataFrame({
                'volume': volumes,
                'outcome': [m.resolution for m in self.markets]
            })

            df['volume_quartile'] = pd.qcut(df['volume'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

            quartile_outcomes = {}
            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                quartile_data = df[df['volume_quartile'] == quartile]
                yes_rate = (quartile_data['outcome'] == 'YES').mean()
                quartile_outcomes[quartile] = {
                    'count': len(quartile_data),
                    'yes_rate': yes_rate,
                    'avg_volume': quartile_data['volume'].mean()
                }

            volume_analysis['outcome_by_volume_quartile'] = quartile_outcomes

        logger.info("Completed volume pattern analysis")
        return volume_analysis

    def calculate_calibration_curve(self) -> pd.DataFrame:
        """
        Create calibration curve data to identify market mispricing.

        Shows for each price bucket (0-10%, 10-20%, etc.) what percentage
        of markets actually resolved YES, revealing calibration quality.

        Returns DataFrame with calibration analysis.
        """
        logger.info("Calculating market calibration curve...")

        if not self.markets:
            return pd.DataFrame()

        # Create price buckets
        price_buckets = []
        for market in self.markets:
            bucket = int(market.final_yes_price * 10) * 10  # Round to nearest 10%
            bucket = min(bucket, 90)  # Cap at 90%
            price_buckets.append({
                'bucket_start': bucket,
                'bucket_end': bucket + 10,
                'final_price': market.final_yes_price,
                'outcome': market.resolution,
                'volume': market.total_volume,
                'category': market.category
            })

        df = pd.DataFrame(price_buckets)

        # Calculate calibration by bucket
        calibration_results = []
        for bucket_start in range(0, 100, 10):
            bucket_end = bucket_start + 10
            bucket_data = df[(df['bucket_start'] == bucket_start)]

            if len(bucket_data) >= 5:  # Minimum sample size
                actual_yes_rate = (bucket_data['outcome'] == 'YES').mean()
                expected_yes_rate = (bucket_start + bucket_end) / 200  # Midpoint of bucket

                calibration_results.append({
                    'price_bucket': f'{bucket_start}-{bucket_end}%',
                    'markets_count': len(bucket_data),
                    'actual_yes_rate': actual_yes_rate,
                    'expected_yes_rate': expected_yes_rate,
                    'calibration_error': actual_yes_rate - expected_yes_rate,
                    'abs_error': abs(actual_yes_rate - expected_yes_rate),
                    'avg_volume': bucket_data['volume'].mean()
                })

        calibration_df = pd.DataFrame(calibration_results)
        logger.info(f"Generated calibration curve with {len(calibration_df)} data points")
        return calibration_df

    def find_profitable_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify historically profitable trading patterns.

        Analyzes historical data to find patterns that would have been
        profitable if traded consistently.

        Returns list of pattern dictionaries with profit metrics.
        """
        logger.info("Searching for profitable historical patterns...")

        patterns = []

        if not self.markets:
            return patterns

        # Pattern 1: Category performance
        category_df = self.analyze_category_accuracy()
        if not category_df.empty:
            for _, row in category_df.iterrows():
                if row['total_markets'] >= 10:
                    # Calculate hypothetical edge
                    yes_rate = row['yes_win_rate']
                    if yes_rate > 0.6:  # Strong YES bias
                        edge = yes_rate - 0.5  # Assuming we'd bet YES when >50%
                        patterns.append({
                            'pattern': f"Bet YES in {row['category']} markets",
                            'description': f"{row['category']} markets resolve YES {yes_rate:.1%} of the time",
                            'edge': edge,
                            'confidence': min(row['total_markets'] / 100, 1.0),  # Higher confidence with more data
                            'sample_size': row['total_markets'],
                            'category': 'category_bias'
                        })

        # Pattern 2: Volume-based edge
        volume_analysis = self.analyze_volume_signals()
        quartile_data = volume_analysis.get('outcome_by_volume_quartile', {})

        for quartile, data in quartile_data.items():
            if data['count'] >= 10:
                yes_rate = data['yes_rate']
                if abs(yes_rate - 0.5) > 0.05:  # Significant deviation
                    direction = "YES" if yes_rate > 0.5 else "NO"
                    edge = abs(yes_rate - 0.5)
                    patterns.append({
                        'pattern': f"Bet {direction} in {quartile} volume quartile markets",
                        'description': f"{quartile} volume markets ({data['avg_volume']:,.0f} avg) resolve {direction} {yes_rate:.1%}",
                        'edge': edge,
                        'confidence': min(data['count'] / 50, 1.0),
                        'sample_size': data['count'],
                        'category': 'volume_bias'
                    })

        # Pattern 3: Calibration-based opportunities
        calibration_df = self.calculate_calibration_curve()
        if not calibration_df.empty:
            for _, row in calibration_df.iterrows():
                if row['markets_count'] >= 10 and abs(row['calibration_error']) > 0.05:
                    direction = "UNDER" if row['calibration_error'] > 0 else "OVER"
                    confidence = min(row['markets_count'] / 50, 1.0)

                    if row['calibration_error'] > 0:  # Markets underpriced for YES
                        pattern_desc = f"Markets priced {row['price_bucket']} actually resolve YES {row['actual_yes_rate']:.1%}"
                        edge = row['calibration_error']
                    else:  # Markets overpriced for YES
                        pattern_desc = f"Markets priced {row['price_bucket']} actually resolve YES {row['actual_yes_rate']:.1%}"
                        edge = abs(row['calibration_error'])

                    patterns.append({
                        'pattern': f"Exploit {direction}confidence in {row['price_bucket']} markets",
                        'description': pattern_desc,
                        'edge': edge,
                        'confidence': confidence,
                        'sample_size': row['markets_count'],
                        'category': 'calibration'
                    })

        # Sort by edge * confidence for prioritization
        patterns.sort(key=lambda x: x['edge'] * x['confidence'], reverse=True)

        logger.info(f"Found {len(patterns)} potentially profitable patterns")
        return patterns

    def get_historical_edge_adjustment(self, market: Any, ai_probability: float) -> float:
        """
        Adjust AI probability based on historical calibration patterns.

        Args:
            market: Market object with category and current pricing
            ai_probability: AI-predicted probability

        Returns:
            Adjusted probability based on historical calibration
        """
        try:
            # Find similar historical markets
            category = getattr(market, 'category', 'unknown')
            current_price = getattr(market, 'yes_price', 0.5)

            # Get calibration data
            calibration_df = self.calculate_calibration_curve()
            if calibration_df.empty:
                return ai_probability

            # Find the calibration bucket for current price
            price_pct = int(current_price * 100)
            bucket_start = (price_pct // 10) * 10

            calibration_row = calibration_df[calibration_df['price_bucket'].str.startswith(f'{bucket_start}-')]
            if calibration_row.empty:
                return ai_probability

            # Apply calibration adjustment
            actual_yes_rate = calibration_row['actual_yes_rate'].iloc[0]
            expected_yes_rate = calibration_row['expected_yes_rate'].iloc[0]

            # Adjust AI probability towards historical actual rate
            adjustment_factor = actual_yes_rate / expected_yes_rate if expected_yes_rate > 0 else 1.0
            adjusted_probability = ai_probability * adjustment_factor

            # Ensure bounds
            adjusted_probability = max(0.01, min(0.99, adjusted_probability))

            logger.debug(".2f")
            return adjusted_probability

        except Exception as e:
            logger.warning(f"Failed to calculate historical edge adjustment: {e}")
            return ai_probability

    def should_trade_market_type(self, market: Any) -> Tuple[bool, str]:
        """
        Decide whether to trade a market type based on historical performance.

        Args:
            market: Market object

        Returns:
            Tuple of (should_trade, reason)
        """
        try:
            category = getattr(market, 'category', 'unknown')
            volume = getattr(market, 'volume', 0)

            # Get category analysis
            category_df = self.analyze_category_accuracy()
            if category_df.empty:
                return True, "Insufficient historical data for analysis"

            category_row = category_df[category_df['category'] == category]
            if category_row.empty:
                return True, f"No historical data for {category} category"

            # Check if category has enough data and reasonable efficiency
            markets_count = category_row['total_markets'].iloc[0]
            efficiency_score = category_row['efficiency_score'].iloc[0]
            edge_opportunities = category_row['edge_opportunity_rate'].iloc[0]

            if markets_count < 10:
                return False, f"Insufficient data: only {markets_count} historical {category} markets"

            if efficiency_score > 0.4:  # Very inefficient markets
                return False, f"{category} markets historically inefficient (efficiency: {efficiency_score:.2f})"

            if edge_opportunities < 0.1:  # Few edge opportunities
                return False, f"{category} markets rarely offer edges ({edge_opportunities:.1%})"

            # Check volume
            if volume < 1000:
                return False, f"Volume too low (${volume:,.0f}) for reliable pricing"

            return True, f"{category} markets historically tradable with {edge_opportunities:.1%} edge opportunities"

        except Exception as e:
            logger.error(f"Error in market type analysis: {e}")
            return True, "Analysis failed, proceeding with trade"

    def generate_insights_report(self) -> str:
        """
        Generate comprehensive markdown report with historical analysis insights.

        Returns:
            Markdown report string
        """
        logger.info("Generating historical insights report...")

        report_lines = [
            "# Historical Market Analysis Report\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Markets Analyzed:** {len(self.markets)}\n",
            "---\n"
        ]

        # Category Analysis
        report_lines.append("## Category Performance Analysis\n\n")
        category_df = self.analyze_category_accuracy()
        if not category_df.empty:
            report_lines.append("| Category | Markets | YES Rate | Efficiency | Edge Opportunities |\n")
            report_lines.append("|----------|---------|----------|------------|-------------------|\n")

            for _, row in category_df.head(10).iterrows():
                report_lines.append(".1%")

            report_lines.append("\n")
        else:
            report_lines.append("No category data available.\n\n")

        # Calibration Analysis
        report_lines.append("## Market Calibration Analysis\n\n")
        calibration_df = self.calculate_calibration_curve()
        if not calibration_df.empty:
            report_lines.append("Shows how well market prices predict outcomes. Perfect calibration would show actual YES rates matching price buckets.\n\n")
            report_lines.append("| Price Bucket | Markets | Actual YES Rate | Expected | Calibration Error |\n")
            report_lines.append("|--------------|---------|-----------------|----------|------------------|\n")

            for _, row in calibration_df.iterrows():
                report_lines.append(".1%")

            # Summary stats
            avg_error = calibration_df['abs_error'].mean()
            max_error = calibration_df['abs_error'].max()
            report_lines.append(f"**Average Calibration Error:** {avg_error:.1f}%\n")
            report_lines.append(f"**Maximum Calibration Error:** {max_error:.1f}%\n\n")
        else:
            report_lines.append("No calibration data available.\n\n")

        # Profitable Patterns
        report_lines.append("## Profitable Trading Patterns\n\n")
        patterns = self.find_profitable_patterns()
        if patterns:
            report_lines.append("Top patterns identified from historical data:\n\n")
            for i, pattern in enumerate(patterns[:10], 1):
                report_lines.append(f"{i}. **{pattern['pattern']}** (Edge: {pattern['edge']:.1%})\n")
                report_lines.append(f"   - **Sample Size:** {pattern['sample_size']} markets\n")
                report_lines.append(f"   - **{pattern['description']}**\n\n")
        else:
            report_lines.append("No profitable patterns identified.\n\n")

        # Price Pattern Insights
        report_lines.append("## Price Movement Patterns\n\n")
        price_patterns = self.analyze_price_patterns()
        if price_patterns.get('total_markets_with_price_data', 0) > 0:
            flips = price_patterns.get('price_flip_analysis', {})
            movements = price_patterns.get('final_day_movements', {})

            report_lines.append("### Price Flip Analysis\n")
            report_lines.append(f"- Markets starting below 50%: {flips.get('started_below_50', 0)}\n")
            report_lines.append(f"- Flipped to YES resolution: {flips.get('flipped_to_yes', 0)}\n")
            report_lines.append(f"- Large final-day moves (>10%): {movements.get('large_moves_10pct', 0)}\n\n")

            if movements.get('count', 0) > 0:
                report_lines.append("### Final 24-Hour Movements\n")
                report_lines.append(".1%\n")
                report_lines.append(".1%\n\n")
        else:
            report_lines.append("No price history data available for pattern analysis.\n\n")

        # Recommendations
        report_lines.append("## Trading Recommendations\n\n")
        if patterns:
            top_pattern = patterns[0]
            report_lines.append(f"**Top Pattern:** {top_pattern['pattern']}\n")
            report_lines.append(".1%\n\n")

        # Categories to trade/avoid
        if not category_df.empty:
            tradable_categories = category_df[
                (category_df['total_markets'] >= 10) &
                (category_df['efficiency_score'] <= 0.4) &
                (category_df['edge_opportunity_rate'] >= 0.1)
            ]

            if not tradable_categories.empty:
                report_lines.append("**Recommended Categories:**\n")
                for _, row in tradable_categories.iterrows():
                    report_lines.append(f"- {row['category']}: {row['edge_opportunity_rate']:.1%} edge opportunities\n")
                report_lines.append("\n")

        report_lines.append("---\n")
        report_lines.append("*Report generated by Polymarket AI Pattern Analyzer*")

        report = "".join(report_lines)

        # Save to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"historical_analysis_{timestamp}.md"

        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"✅ Historical insights report saved to {report_file}")
        return report


def create_pattern_analyzer(db_path: str = "data/historical.db") -> PatternAnalyzer:
    """Create and return a PatternAnalyzer instance."""
    return PatternAnalyzer(db_path)
