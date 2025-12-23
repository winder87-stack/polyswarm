"""
AI Accuracy Tracker for Polymarket Trading

Tracks and analyzes the accuracy, calibration, and performance of AI models
over time to optimize trading strategies and model weights.
"""

import os
import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from loguru import logger

# Import for plotting (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - plotting features disabled")


@dataclass
class AIPrediction:
    """Tracks a single AI prediction for later accuracy analysis."""
    prediction_id: str
    timestamp: datetime
    condition_id: str
    question: str

    # AI predictions
    ai_probability: float  # Ensemble probability
    model_probabilities: Dict[str, float]  # Individual model predictions
    confidence: float  # Ensemble confidence score

    # Market state at prediction time
    market_probability: float
    volume: float
    hours_until_close: float
    category: str

    # Outcome (filled later when market resolves)
    resolution: Optional[str] = None  # "YES", "NO", "INVALID"
    resolved_at: Optional[datetime] = None
    was_correct: Optional[bool] = None  # Whether ensemble was directionally correct
    brier_score: Optional[float] = None  # Calibration score
    profit_if_traded: Optional[float] = None  # Hypothetical profit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIPrediction':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return cls(**data)


class AIAccuracyTracker:
    """
    Tracks AI model predictions and analyzes their accuracy over time.

    Provides insights into model performance, calibration, and optimal weighting
    to improve trading strategy effectiveness.
    """

    def __init__(self, db_path: str = "data/ai_predictions.db") -> None:
        """Initialize the accuracy tracker with SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    condition_id TEXT NOT NULL,
                    question TEXT,
                    ai_probability REAL,
                    model_probabilities TEXT,  -- JSON string
                    confidence REAL,
                    market_probability REAL,
                    volume REAL,
                    hours_until_close REAL,
                    category TEXT,
                    resolution TEXT,
                    resolved_at TIMESTAMP,
                    was_correct BOOLEAN,
                    brier_score REAL,
                    profit_if_traded REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Model performance cache (for faster queries)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT,
                    date DATE,
                    predictions_count INTEGER,
                    accuracy REAL,
                    brier_score REAL,
                    avg_confidence REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_name, date)
                )
            ''')

            conn.commit()
            logger.info(f"✅ Initialized AI predictions database at {self.db_path}")

    def record_prediction(self, signal: Any) -> str:
        """
        Record a trading signal prediction for later accuracy analysis.

        Args:
            signal: TradingSignal object from TradingSwarm

        Returns:
            Prediction ID for tracking
        """
        try:
            # Generate unique prediction ID
            prediction_id = f"{signal.market.slug}_{int(signal.timestamp.timestamp())}_{id(signal)}"

            # Extract market info
            market_info = getattr(signal.market, '_market_info', {})
            hours_until_close = getattr(signal.market, 'hours_until_close', 24.0)
            category = getattr(signal.market, 'category', 'unknown')

            # Create prediction record
            prediction = AIPrediction(
                prediction_id=prediction_id,
                timestamp=signal.timestamp,
                condition_id=getattr(signal.market, 'slug', 'unknown'),
                question=signal.market.question,
                ai_probability=signal.probability,
                model_probabilities=signal.model_votes,
                confidence=signal.confidence,
                market_probability=signal.market_probability,
                volume=getattr(signal.market, 'volume', 0),
                hours_until_close=hours_until_close,
                category=category
            )

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO predictions
                    (prediction_id, timestamp, condition_id, question, ai_probability,
                     model_probabilities, confidence, market_probability, volume,
                     hours_until_close, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.prediction_id,
                    prediction.timestamp,
                    prediction.condition_id,
                    prediction.question,
                    prediction.ai_probability,
                    json.dumps(prediction.model_probabilities),
                    prediction.confidence,
                    prediction.market_probability,
                    prediction.volume,
                    prediction.hours_until_close,
                    prediction.category
                ))
                conn.commit()

            logger.debug(f"✅ Recorded prediction: {prediction_id}")
            return prediction_id

        except Exception as e:
            logger.error(f"❌ Failed to record prediction: {e}")
            return ""

    async def update_resolutions(self) -> int:
        """
        Check for resolved markets and update prediction outcomes.

        Returns:
            Number of predictions updated
        """
        try:
            from src.data import create_historical_collector
            collector = create_historical_collector()

            # Get recently resolved markets (last 30 days)
            resolved_markets = collector.get_resolved_markets(
                resolution=None,  # Get all resolutions
                limit=1000
            )

            # Filter to recent resolutions
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_resolutions = [
                m for m in resolved_markets
                if m.resolved_at and m.resolved_at > cutoff_date
            ]

            logger.info(f"Checking {len(recent_resolutions)} recently resolved markets")

            updated_count = 0

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for market in recent_resolutions:
                    # Find predictions for this market that haven't been resolved yet
                    cursor.execute('''
                        SELECT prediction_id, ai_probability, market_probability, model_probabilities
                        FROM predictions
                        WHERE condition_id = ? AND resolution IS NULL
                    ''', (market.condition_id,))

                    predictions = cursor.fetchall()

                    for pred_row in predictions:
                        prediction_id = pred_row[0]
                        ai_probability = pred_row[1]
                        market_probability = pred_row[2]
                        model_probs = json.loads(pred_row[3]) if pred_row[3] else {}

                        # Determine if prediction was directionally correct
                        ai_direction = "YES" if ai_probability > 0.5 else "NO"
                        actual_outcome = market.resolution

                        was_correct = ai_direction == actual_outcome

                        # Calculate Brier score (calibration metric)
                        # Brier score = (prediction - outcome)^2
                        # Lower is better (0 = perfect calibration)
                        outcome_binary = 1.0 if actual_outcome == "YES" else 0.0
                        brier_score = (ai_probability - outcome_binary) ** 2

                        # Hypothetical profit if traded
                        # Simplified: assume we bet $100 on our predicted direction
                        stake = 100.0
                        if ai_direction == "YES":
                            # YES bet: profit = stake * (1/market_prob - 1) if correct
                            if actual_outcome == "YES":
                                profit = stake * ((1.0 / market_probability) - 1.0)
                            else:
                                profit = -stake
                        else:
                            # NO bet: profit = stake * (1/(1-market_prob) - 1) if correct
                            if actual_outcome == "NO":
                                profit = stake * ((1.0 / (1.0 - market_probability)) - 1.0)
                            else:
                                profit = -stake

                        # Update the prediction record
                        cursor.execute('''
                            UPDATE predictions
                            SET resolution = ?, resolved_at = ?, was_correct = ?,
                                brier_score = ?, profit_if_traded = ?
                            WHERE prediction_id = ?
                        ''', (
                            actual_outcome,
                            market.resolved_at,
                            was_correct,
                            brier_score,
                            profit,
                            prediction_id
                        ))

                        updated_count += 1
                        logger.debug(f"✅ Updated prediction {prediction_id}: {actual_outcome}, correct={was_correct}")

                conn.commit()

            logger.info(f"✅ Updated {updated_count} prediction outcomes")
            return updated_count

        except Exception as e:
            logger.error(f"❌ Failed to update resolutions: {e}")
            return 0

    def calculate_model_accuracy(self) -> pd.DataFrame:
        """
        Calculate accuracy metrics for each individual model.

        Returns DataFrame with model performance metrics:
        - Model name
        - Predictions count
        - Accuracy (% correct directional predictions)
        - Brier score (calibration quality)
        - Average confidence
        - Bias (over/under confident)
        """
        logger.info("Calculating individual model accuracy...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all resolved predictions
                df = pd.read_sql_query('''
                    SELECT model_probabilities, resolution, was_correct
                    FROM predictions
                    WHERE resolution IS NOT NULL
                ''', conn)

            if df.empty:
                logger.warning("No resolved predictions found for model analysis")
                return pd.DataFrame()

            model_metrics = []

            for prediction in df.itertuples():
                if not prediction.model_probabilities:
                    continue

                try:
                    model_probs = json.loads(prediction.model_probabilities)
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse model probabilities for prediction {prediction.id}")
                    continue

                actual_outcome = prediction.resolution
                outcome_binary = 1.0 if actual_outcome == "YES" else 0.0

                for model_name, model_prob in model_probs.items():
                    # Calculate directional accuracy
                    model_direction = "YES" if model_prob > 0.5 else "NO"
                    direction_correct = model_direction == actual_outcome

                    # Calculate Brier score for this model
                    brier = (model_prob - outcome_binary) ** 2

                    model_metrics.append({
                        'model': model_name,
                        'probability': model_prob,
                        'direction_correct': direction_correct,
                        'brier_score': brier,
                        'outcome': outcome_binary
                    })

            if not model_metrics:
                return pd.DataFrame()

            metrics_df = pd.DataFrame(model_metrics)

            # Aggregate by model
            summary = []
            for model in metrics_df['model'].unique():
                model_data = metrics_df[metrics_df['model'] == model]

                accuracy = model_data['direction_correct'].mean()
                avg_brier = model_data['brier_score'].mean()
                avg_confidence = model_data['probability'].abs().mean()  # Rough confidence proxy
                prediction_count = len(model_data)

                # Calculate bias: average (predicted - actual)
                bias = (model_data['probability'] - model_data['outcome']).mean()

                summary.append({
                    'model': model,
                    'predictions': prediction_count,
                    'accuracy': accuracy,
                    'brier_score': avg_brier,
                    'avg_confidence': avg_confidence,
                    'bias': bias
                })

            result_df = pd.DataFrame(summary).sort_values('accuracy', ascending=False)
            logger.info(f"Calculated accuracy for {len(result_df)} models")
            return result_df

        except Exception as e:
            logger.error(f"❌ Failed to calculate model accuracy: {e}")
            return pd.DataFrame()

    def calculate_ensemble_accuracy(self) -> Dict[str, Any]:
        """
        Calculate how the ensemble performs vs individual models.

        Returns dictionary with ensemble vs individual model comparison.
        """
        logger.info("Calculating ensemble accuracy...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT ai_probability, resolution, confidence, was_correct, brier_score
                    FROM predictions
                    WHERE resolution IS NOT NULL
                ''', conn)

            if df.empty:
                return {"error": "No resolved predictions found"}

            ensemble_metrics = {
                'ensemble_predictions': len(df),
                'ensemble_accuracy': df['was_correct'].mean(),
                'ensemble_avg_brier': df['brier_score'].mean(),
                'ensemble_avg_confidence': df['confidence'].mean(),
                'individual_models': {}
            }

            # Compare to individual model performance
            individual_df = self.calculate_model_accuracy()
            if not individual_df.empty:
                best_individual = individual_df.iloc[0]
                ensemble_metrics.update({
                    'best_individual_model': best_individual['model'],
                    'best_individual_accuracy': best_individual['accuracy'],
                    'ensemble_vs_best': ensemble_metrics['ensemble_accuracy'] - best_individual['accuracy'],
                    'ensemble_improvement_pct': (
                        (ensemble_metrics['ensemble_accuracy'] - best_individual['accuracy']) /
                        best_individual['accuracy'] * 100
                    ) if best_individual['accuracy'] > 0 else 0
                })

                # Store individual model details
                for _, row in individual_df.iterrows():
                    ensemble_metrics['individual_models'][row['model']] = {
                        'accuracy': row['accuracy'],
                        'brier_score': row['brier_score'],
                        'predictions': row['predictions']
                    }

            logger.info(f"Calculated ensemble accuracy: {ensemble_metrics['ensemble_accuracy']:.1%}")
            return ensemble_metrics

        except Exception as e:
            logger.error(f"❌ Failed to calculate ensemble accuracy: {e}")
            return {"error": str(e)}

    def get_optimal_weights(self) -> Dict[str, float]:
        """
        Calculate optimal model weights based on historical accuracy.

        Uses recent prediction accuracy to determine the best weighting scheme.

        Returns dictionary of model weights.
        """
        logger.info("Calculating optimal model weights...")

        try:
            # Get recent model performance (last 100 predictions)
            accuracy_df = self.calculate_model_accuracy()

            if accuracy_df.empty:
                logger.warning("No accuracy data available for weight optimization")
                # Return default weights
                return {
                    "claude": 1.3,
                    "gemini": 1.3,
                    "gpt": 1.2,
                    "perplexity": 1.2,
                    "deepseek": 1.0
                }

            # Calculate weights based on accuracy and calibration
            # Higher weight for better accuracy and lower Brier score
            weights = {}
            max_accuracy = accuracy_df['accuracy'].max()
            min_brier = accuracy_df['brier_score'].min()

            for _, row in accuracy_df.iterrows():
                # Normalize accuracy (0-1 scale)
                norm_accuracy = row['accuracy'] / max_accuracy if max_accuracy > 0 else 0.5

                # Normalize Brier score (lower is better, so invert)
                # Brier scores typically range from 0-0.25, so use 0.25 as baseline
                norm_calibration = max(0, (0.25 - row['brier_score']) / 0.25)

                # Combined score: 70% accuracy, 30% calibration
                combined_score = 0.7 * norm_accuracy + 0.3 * norm_calibration

                # Scale to reasonable weight range (0.5 to 2.0)
                weight = 0.5 + (combined_score * 1.5)

                weights[row['model']] = round(weight, 2)

            # Ensure we have weights for all expected models
            expected_models = ["claude", "gemini", "gpt", "perplexity", "deepseek"]
            for model in expected_models:
                if model not in weights:
                    weights[model] = 1.0  # Default weight

            logger.info(f"Calculated optimal weights: {weights}")
            return weights

        except Exception as e:
            logger.error(f"❌ Failed to calculate optimal weights: {e}")
            return {"claude": 1.3, "gemini": 1.3, "gpt": 1.2, "perplexity": 1.2, "deepseek": 1.0}

    def get_calibration_adjustment(self, model: str, probability: float) -> float:
        """
        Get calibration adjustment for a specific model and probability.

        Args:
            model: Model name (e.g., 'claude', 'gpt')
            probability: Raw probability prediction

        Returns:
            Calibrated probability adjusted for historical bias
        """
        try:
            # Get historical predictions for this model
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT model_probabilities, resolution
                    FROM predictions
                    WHERE resolution IS NOT NULL AND model_probabilities LIKE ?
                    LIMIT 1000
                ''', conn, params=[f'%{model}%'])

            if df.empty:
                return probability

            # Extract this model's predictions
            model_predictions = []
            for _, row in df.iterrows():
                try:
                    model_probs = json.loads(row['model_probabilities'])
                    if model in model_probs:
                        outcome_binary = 1.0 if row['resolution'] == "YES" else 0.0
                        model_predictions.append({
                            'prediction': model_probs[model],
                            'outcome': outcome_binary
                        })
                except (KeyError, TypeError) as e:
                    logger.debug(f"Error processing model {model} data: {e}")
                    continue

            if len(model_predictions) < 10:
                return probability

            # Create calibration bins
            pred_df = pd.DataFrame(model_predictions)
            pred_df['bin'] = pd.cut(pred_df['prediction'], bins=10, labels=False)

            # Find the bin for our probability
            bin_idx = min(9, int(probability * 10))

            # Get actual outcomes for this bin
            bin_data = pred_df[pred_df['bin'] == bin_idx]
            if len(bin_data) >= 5:
                actual_avg = bin_data['outcome'].mean()
                # Adjust towards historical average
                adjustment_factor = actual_avg / probability if probability > 0 else 1.0
                adjusted = probability * adjustment_factor
                adjusted = max(0.01, min(0.99, adjusted))
                return adjusted

            return probability

        except Exception as e:
            logger.warning(f"Failed to get calibration adjustment for {model}: {e}")
            return probability

    def suggest_weight_updates(self) -> Dict[str, Any]:
        """
        Suggest weight updates based on recent performance.

        Analyzes recent predictions to suggest optimal model weights.

        Returns dictionary with current/suggested weights and reasoning.
        """
        logger.info("Generating weight update suggestions...")

        try:
            optimal_weights = self.get_optimal_weights()
            current_weights = {
                "claude": 1.3,
                "gemini": 1.3,
                "gpt": 1.2,
                "perplexity": 1.2,
                "deepseek": 1.0
            }

            # Calculate changes
            changes = {}
            reasoning_parts = []

            for model in current_weights:
                current = current_weights[model]
                suggested = optimal_weights.get(model, current)
                change = suggested - current
                changes[model] = {
                    'current': current,
                    'suggested': suggested,
                    'change': change,
                    'change_pct': (change / current * 100) if current > 0 else 0
                }

                if abs(change) > 0.1:
                    direction = "increase" if change > 0 else "decrease"
                    reasoning_parts.append(f"{model} {direction} by {abs(change):.1f}")

            reasoning = f"Analysis suggests: {', '.join(reasoning_parts)}" if reasoning_parts else "No significant changes recommended"

            return {
                'current_weights': current_weights,
                'suggested_weights': optimal_weights,
                'changes': changes,
                'reasoning': reasoning,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to generate weight suggestions: {e}")
            return {
                'error': str(e),
                'current_weights': {"claude": 1.3, "gemini": 1.3, "gpt": 1.2, "perplexity": 1.2, "deepseek": 1.0},
                'suggested_weights': {"claude": 1.3, "gemini": 1.3, "gpt": 1.2, "perplexity": 1.2, "deepseek": 1.0}
            }

    def plot_calibration_curves(self, save_path: str = "reports/calibration.png"):
        """
        Plot calibration curves for each model.

        Shows how well each model's confidence matches actual outcomes.
        Perfect calibration = diagonal line.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available - skipping calibration plot")
            return

        try:
            logger.info("Generating calibration curve plots...")

            # Get model accuracy data
            accuracy_df = self.calculate_model_accuracy()
            if accuracy_df.empty:
                logger.warning("No accuracy data available for plotting")
                return

            # Create subplots for each model
            models = accuracy_df['model'].tolist()
            n_models = len(models)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Get detailed prediction data for calibration
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT model_probabilities, resolution
                    FROM predictions
                    WHERE resolution IS NOT NULL
                    LIMIT 1000
                ''', conn)

            for i, model in enumerate(models):
                ax = axes[i]

                # Extract this model's predictions
                model_predictions = []
                for _, row in df.iterrows():
                    try:
                        model_probs = json.loads(row['model_probabilities'] or '{}')
                        if model in model_probs:
                            outcome_binary = 1.0 if row['resolution'] == "YES" else 0.0
                            model_predictions.append((model_probs[model], outcome_binary))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Error processing calibration data for model {model}: {e}")
                        continue

                if len(model_predictions) < 10:
                    ax.text(0.5, 0.5, f'Insufficient data\nfor {model}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model} Calibration')
                    continue

                # Create calibration bins
                probs, outcomes = zip(*model_predictions)
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                actual_rates = []
                for j in range(len(bins) - 1):
                    mask = (np.array(probs) >= bins[j]) & (np.array(probs) < bins[j + 1])
                    if np.any(mask):
                        actual_rate = np.mean(np.array(outcomes)[mask])
                        actual_rates.append(actual_rate)
                    else:
                        actual_rates.append(np.nan)

                # Plot calibration curve
                ax.plot(bin_centers, bin_centers, 'k--', alpha=0.5, label='Perfect calibration')
                ax.plot(bin_centers, actual_rates, 'bo-', linewidth=2, label=f'{model} actual')

                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Actual Outcome Rate')
                ax.set_title(f'{model} Calibration Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved calibration curves to {save_path}")

        except Exception as e:
            logger.error(f"❌ Failed to generate calibration plots: {e}")

    def plot_accuracy_over_time(self, save_path: str = "reports/accuracy_trend.png"):
        """
        Plot rolling accuracy over time.

        Shows how model performance changes over time.
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available - skipping accuracy trend plot")
            return

        try:
            logger.info("Generating accuracy trend plots...")

            # Get prediction data over time
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT timestamp, was_correct, confidence, ai_probability
                    FROM predictions
                    WHERE resolution IS NOT NULL
                    ORDER BY timestamp
                ''', conn)

            if df.empty:
                logger.warning("No time-series data available for plotting")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Calculate rolling metrics
            rolling_window = min(50, len(df) // 4)  # At least 50 predictions or 1/4 of data
            if rolling_window < 10:
                logger.warning("Insufficient data for rolling accuracy calculation")
                return

            df['rolling_accuracy'] = df['was_correct'].rolling(window=rolling_window).mean()
            df['rolling_confidence'] = df['confidence'].rolling(window=rolling_window).mean()

            # Plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Accuracy over time
            ax1.plot(df.index, df['rolling_accuracy'], 'b-', linewidth=2, label='Rolling Accuracy')
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
            ax1.set_ylabel('Rolling Accuracy')
            ax1.set_title('AI Model Accuracy Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Confidence over time
            ax2.plot(df.index, df['rolling_confidence'], 'g-', linewidth=2, label='Rolling Confidence')
            ax2.set_ylabel('Rolling Confidence')
            ax2.set_xlabel('Date')
            ax2.set_title('AI Model Confidence Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Format dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ Saved accuracy trend plot to {save_path}")

        except Exception as e:
            logger.error(f"❌ Failed to generate accuracy trend plot: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total_predictions = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM predictions WHERE resolution IS NOT NULL")
                resolved_predictions = cursor.fetchone()[0]

                summary = {
                    'total_predictions': total_predictions,
                    'resolved_predictions': resolved_predictions,
                    'unresolved_predictions': total_predictions - resolved_predictions,
                    'resolution_rate': resolved_predictions / total_predictions if total_predictions > 0 else 0,
                    'model_accuracy': {},
                    'ensemble_performance': {},
                    'generated_at': datetime.now().isoformat()
                }

                # Model accuracy summary
                model_df = self.calculate_model_accuracy()
                if not model_df.empty:
                    summary['model_accuracy'] = {
                        row['model']: {
                            'accuracy': row['accuracy'],
                            'brier_score': row['brier_score'],
                            'predictions': row['predictions']
                        }
                        for _, row in model_df.iterrows()
                    }

                # Ensemble performance
                ensemble = self.calculate_ensemble_accuracy()
                summary['ensemble_performance'] = ensemble

                return summary

        except Exception as e:
            logger.error(f"❌ Failed to generate performance summary: {e}")
            return {'error': str(e)}


def create_accuracy_tracker(db_path: str = "data/ai_predictions.db") -> AIAccuracyTracker:
    """Create and return an AIAccuracyTracker instance."""
    return AIAccuracyTracker(db_path)
