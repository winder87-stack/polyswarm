"""
Model Calibration - Correct systematic biases in AI predictions

Tracks prediction accuracy vs outcomes and applies calibration adjustments
to improve probability estimates for prediction market trading.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

from loguru import logger
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.agents.swarm_agent import MODEL_WEIGHTS
except ImportError:
    # Fallback for testing
    MODEL_WEIGHTS = {
        "claude": 1.3, "gemini": 1.3, "gpt": 1.2,
        "deepseek": 1.0, "perplexity": 1.2
    }


@dataclass
class CalibrationBucket:
    """Data for one bucket in calibration curve."""
    predicted_range: Tuple[float, float]  # e.g., (0.60, 0.70)
    predictions_count: int
    actual_yes_rate: float  # What % actually resolved YES
    calibration_error: float  # predicted_midpoint - actual_rate
    brier_score: float = 0.0  # Brier score for this bucket


@dataclass
class ModelData:
    """Historical predictions and outcomes for a model."""
    predictions: List[float] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)  # 1 for YES, 0 for NO
    timestamps: List[datetime] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


class ModelCalibration:
    """
    Track and correct model calibration for improved prediction accuracy.

    Maintains database of predictions vs outcomes and provides calibration
    adjustments to correct systematic biases in model outputs.
    """

    def __init__(self, db_path: str = "data/calibration.db") -> None:
        """Initialize calibration tracker."""
        self.db_path = db_path
        self._init_database()
        self.models_data = {}  # Cache for faster access

        logger.info(f"📊 Model calibration initialized: {db_path}")

    def _init_database(self):
        """Create calibration database tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    predicted_prob REAL NOT NULL,
                    category TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT NULL,
                    actual_outcome INTEGER DEFAULT NULL,  -- 1 for YES, 0 for NO
                    UNIQUE(model, market_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def record_prediction(
        self,
        model: str,
        predicted_prob: float,
        market_id: str,
        category: str = "unknown"
    ):
        """
        Record a prediction for later calibration analysis.

        Args:
            model: Model name (e.g., "claude", "gpt")
            predicted_prob: Predicted probability (0-1)
            market_id: Unique market identifier
            category: Market category
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions
                    (model, market_id, predicted_prob, category)
                    VALUES (?, ?, ?, ?)
                """, (model, market_id, predicted_prob, category))
                conn.commit()

            logger.debug(f"📝 Recorded prediction: {model} {market_id} -> {predicted_prob:.1%}")

        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")

    def record_resolution(self, market_id: str, resolved_yes: bool):
        """
        Record the actual outcome for a market.

        Args:
            market_id: Market identifier
            resolved_yes: True if resolved YES, False if NO
        """
        try:
            actual_outcome = 1 if resolved_yes else 0

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE predictions
                    SET resolved = TRUE, actual_outcome = ?
                    WHERE market_id = ?
                """, (actual_outcome, market_id))

                updated = cursor.rowcount
                conn.commit()

            if updated > 0:
                logger.info(f"✅ Recorded resolution: {market_id} -> {'YES' if resolved_yes else 'NO'} ({updated} predictions)")
            else:
                logger.warning(f"No predictions found for resolved market: {market_id}")

        except Exception as e:
            logger.error(f"Failed to record resolution: {e}")

    def get_model_data(self, model: str, category: Optional[str] = None) -> Tuple[List[float], List[int]]:
        """
        Get historical predictions and outcomes for calibration.

        Returns:
            (predictions, actuals) - lists of equal length
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if category:
                    rows = conn.execute("""
                        SELECT predicted_prob, actual_outcome
                        FROM predictions
                        WHERE model = ? AND category = ? AND resolved = TRUE
                        ORDER BY timestamp
                    """, (model, category)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT predicted_prob, actual_outcome
                        FROM predictions
                        WHERE model = ? AND resolved = TRUE
                        ORDER BY timestamp
                    """, (model,)).fetchall()

            predictions = [row[0] for row in rows if row[1] is not None]
            actuals = [row[1] for row in rows if row[1] is not None]

            return predictions, actuals

        except Exception as e:
            logger.error(f"Failed to get model data: {e}")
            return [], []

    def calculate_calibration_curve(
        self,
        model: str,
        category: Optional[str] = None,
        buckets: int = 10
    ) -> List[CalibrationBucket]:
        """
        Calculate calibration curve showing predicted vs actual outcomes.

        Perfect calibration: when model says X%, outcomes are X% YES.

        Args:
            model: Model name
            category: Optional category filter
            buckets: Number of probability buckets

        Returns:
            List of CalibrationBucket objects
        """
        predictions, actuals = self.get_model_data(model, category)

        if len(predictions) < 10:
            logger.warning(f"Insufficient data for {model} calibration: {len(predictions)} predictions")
            return []

        try:
            # Create buckets
            pred_array = np.array(predictions)
            actual_array = np.array(actuals)

            # Create equal-sized buckets
            bucket_edges = np.linspace(0, 1, buckets + 1)

            calibration_buckets = []

            for i in range(buckets):
                bucket_min = bucket_edges[i]
                bucket_max = bucket_edges[i + 1]

                # Find predictions in this bucket
                mask = (pred_array >= bucket_min) & (pred_array < bucket_max)
                bucket_predictions = pred_array[mask]
                bucket_actuals = actual_array[mask]

                if len(bucket_predictions) > 0:
                    actual_yes_rate = np.mean(bucket_actuals)
                    predicted_midpoint = (bucket_min + bucket_max) / 2
                    calibration_error = predicted_midpoint - actual_yes_rate

                    # Calculate Brier score for this bucket
                    brier_score = np.mean((bucket_predictions - bucket_actuals) ** 2)

                    bucket = CalibrationBucket(
                        predicted_range=(bucket_min, bucket_max),
                        predictions_count=len(bucket_predictions),
                        actual_yes_rate=actual_yes_rate,
                        calibration_error=calibration_error,
                        brier_score=brier_score
                    )
                    calibration_buckets.append(bucket)

            return calibration_buckets

        except Exception as e:
            logger.error(f"Failed to calculate calibration curve: {e}")
            return []

    def get_calibration_adjustment(
        self,
        model: str,
        predicted_prob: float,
        category: Optional[str] = None
    ) -> float:
        """
        Get calibrated probability based on historical performance.

        Uses simple bucket-based adjustment as fallback for isotonic regression.

        Args:
            model: Model name
            predicted_prob: Raw predicted probability
            category: Optional category filter

        Returns:
            Calibrated probability (0-1)
        """
        buckets = self.calculate_calibration_curve(model, category, buckets=10)

        if not buckets:
            return predicted_prob  # No adjustment data

        # Find the bucket this prediction falls into
        for bucket in buckets:
            bucket_min, bucket_max = bucket.predicted_range
            if bucket_min <= predicted_prob < bucket_max:
                # Adjust based on calibration error
                adjusted = predicted_prob - bucket.calibration_error

                # Clamp to valid range
                adjusted = max(0.01, min(0.99, adjusted))

                logger.debug(f"🎯 Calibrated {model}: {predicted_prob:.1%} -> {adjusted:.1%} (error: {bucket.calibration_error:+.1%})")
                return adjusted

        # Fallback: return original if no bucket matches
        return predicted_prob

    def get_brier_score(self, model: str, category: Optional[str] = None) -> float:
        """
        Calculate Brier score for model.

        Brier = mean((predicted - actual)^2)
        Lower is better. 0 = perfect, 0.25 = random guessing
        """
        predictions, actuals = self.get_model_data(model, category)

        if len(predictions) < 2:
            return 0.25  # Default for insufficient data

        try:
            brier_score = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
            return float(brier_score)
        except Exception:
            return 0.25

    def rank_models_by_accuracy(self) -> pd.DataFrame:
        """
        Rank all models by various accuracy metrics.

        Returns DataFrame with model performance metrics.
        """
        models = ["claude", "gemini", "gpt", "deepseek", "perplexity"]
        results = []

        for model in models:
            predictions, actuals = self.get_model_data(model)

            if len(predictions) < 5:
                continue  # Skip models with insufficient data

            try:
                pred_array = np.array(predictions)
                actual_array = np.array(actuals)

                # Brier score
                brier_score = np.mean((pred_array - actual_array) ** 2)

                # Log loss (cross-entropy)
                epsilon = 1e-15
                pred_clipped = np.clip(pred_array, epsilon, 1 - epsilon)
                log_loss = -np.mean(
                    actual_array * np.log(pred_clipped) +
                    (1 - actual_array) * np.log(1 - pred_clipped)
                )

                # Mean absolute error
                mae = np.mean(np.abs(pred_array - actual_array))

                # Calibration error (mean calibration error across buckets)
                calibration_buckets = self.calculate_calibration_curve(model)
                avg_calibration_error = np.mean([
                    abs(bucket.calibration_error) for bucket in calibration_buckets
                ]) if calibration_buckets else 0

                results.append({
                    "model": model,
                    "predictions": len(predictions),
                    "brier_score": brier_score,
                    "log_loss": log_loss,
                    "mae": mae,
                    "calibration_error": avg_calibration_error,
                    "win_rate": np.mean(actual_array == (pred_array > 0.5))
                })

            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {model}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("brier_score")  # Lower Brier is better

        return df

    def generate_calibration_report(self) -> str:
        """
        Generate comprehensive markdown report on model calibration.
        """
        report_lines = [
            "# Model Calibration Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Rankings",
            ""
        ]

        rankings_df = self.rank_models_by_accuracy()
        if not rankings_df.empty:
            # Convert to markdown table
            table_lines = ["| Model | Predictions | Brier Score | Log Loss | MAE | Calibration Error | Win Rate |",
                          "|-------|-------------|-------------|----------|-----|------------------|----------|"]

            for _, row in rankings_df.iterrows():
                table_lines.append(
                    f"| {row['model']} | {int(row['predictions'])} | "
                    f"{row['brier_score']:.3f} | {row['log_loss']:.3f} | "
                    f"{row['mae']:.3f} | {row['calibration_error']:.3f} | "
                    f"{row['win_rate']:.1%} |"
                )

            report_lines.extend(table_lines)
        else:
            report_lines.append("No sufficient data for model rankings.")

        report_lines.extend([
            "",
            "## Calibration Analysis",
            "",
            "Brier Score: Lower is better (0 = perfect, 0.25 = random guessing)",
            "Calibration Error: Average absolute deviation from perfect calibration",
            "",
            "## Recommendations",
            ""
        ])

        if not rankings_df.empty:
            best_model = rankings_df.iloc[0]['model']
            worst_model = rankings_df.iloc[-1]['model']

            report_lines.extend([
                f"- **Best performing**: {best_model} (consider increasing weight)",
                f"- **Needs improvement**: {worst_model} (consider decreasing weight or recalibrating)",
                "",
                "### Weight Adjustment Suggestions",
                ""
            ])

            # Suggest weight adjustments based on performance
            current_weights = MODEL_WEIGHTS.copy()
            suggestions = []

            for _, row in rankings_df.iterrows():
                model = row['model']
                brier = row['brier_score']
                current_weight = current_weights.get(model, 1.0)

                # Simple adjustment: better models get slightly higher weight
                if brier < 0.2:  # Good performance
                    new_weight = current_weight * 1.1
                    suggestions.append(f"- {model}: {current_weight:.1f} → {new_weight:.1f} (good performance)")
                elif brier > 0.3:  # Poor performance
                    new_weight = current_weight * 0.9
                    suggestions.append(f"- {model}: {current_weight:.1f} → {new_weight:.1f} (poor performance)")

            if suggestions:
                report_lines.extend(suggestions)
            else:
                report_lines.append("- No significant weight adjustments recommended")
        else:
            report_lines.append("Insufficient data for recommendations.")

        report_lines.extend([
            "",
            "## Calibration Curves",
            "",
            "Run `python main.py calibration-report --plots` to see detailed calibration curves.",
            "",
            "---",
            "*Generated by Polymarket AI Trading Bot*"
        ])

        return "\n".join(report_lines)

    def get_stats_summary(self) -> Dict:
        """Get summary statistics for dashboard."""
        try:
            rankings = self.rank_models_by_accuracy()
            if rankings.empty:
                return {"total_predictions": 0, "models_tracked": 0}

            return {
                "total_predictions": int(rankings["predictions"].sum()),
                "models_tracked": len(rankings),
                "best_model": rankings.iloc[0]["model"],
                "best_brier": float(rankings.iloc[0]["brier_score"]),
                "avg_calibration_error": float(rankings["calibration_error"].mean())
            }
        except Exception:
            return {"total_predictions": 0, "models_tracked": 0}


class AutoCalibrator:
    """
    Automatically calibrate model outputs using isotonic regression.

    Builds calibration functions for each model based on historical
    predictions vs outcomes, then applies corrections to new predictions.
    """

    def __init__(self, calibration: ModelCalibration) -> None:
        """Initialize auto-calibrator."""
        self.calibration = calibration
        self.adjustment_functions = {}
        self._build_adjustment_functions()

        logger.info("🎯 Auto-calibrator initialized")

    def _build_adjustment_functions(self):
        """
        Build isotonic regression calibration functions for each model.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LinearRegression

            models = ["claude", "gemini", "gpt", "deepseek", "perplexity"]

            for model in models:
                predictions, actuals = self.calibration.get_model_data(model)

                if len(predictions) >= 50:  # Need sufficient data for reliable calibration
                    try:
                        # Use isotonic regression for monotonic calibration
                        ir = IsotonicRegression(y_min=0, y_max=1, increasing=True)
                        ir.fit(predictions, actuals)

                        self.adjustment_functions[model] = ir
                        logger.info(f"✅ Built calibration function for {model} ({len(predictions)} samples)")

                    except Exception as e:
                        logger.warning(f"Failed to build isotonic regression for {model}: {e}")

                        # Fallback to linear regression
                        try:
                            lr = LinearRegression()
                            X = np.array(predictions).reshape(-1, 1)
                            lr.fit(X, actuals)
                            self.adjustment_functions[model] = lr
                            logger.info(f"⚠️ Using linear regression fallback for {model}")
                        except Exception:
                            logger.warning(f"Failed to build any calibration for {model}")

                else:
                    logger.info(f"⏳ Insufficient data for {model} calibration: {len(predictions)} predictions")

        except ImportError:
            logger.warning("sklearn not available - calibration functions disabled")

    def calibrate_probability(
        self,
        model: str,
        raw_probability: float
    ) -> float:
        """
        Apply calibration adjustment to raw model output.

        Args:
            model: Model name
            raw_probability: Raw predicted probability (0-1)

        Returns:
            Calibrated probability (0-1)
        """
        if model not in self.adjustment_functions:
            return raw_probability

        try:
            calibrated = self.adjustment_functions[model].predict([raw_probability])[0]

            # Clamp to valid probability range
            calibrated = max(0.001, min(0.999, calibrated))

            logger.debug(f"🎯 {model} calibration: {raw_probability:.1%} → {calibrated:.1%}")
            return float(calibrated)

        except Exception as e:
            logger.warning(f"Calibration failed for {model}: {e}")
            return raw_probability

    def calibrate_ensemble(
        self,
        model_probabilities: Dict[str, float]
    ) -> float:
        """
        Calibrate each model's output then combine with weights.

        Better than calibrating after combining since calibration is nonlinear.

        Args:
            model_probabilities: Dict of model -> raw_probability

        Returns:
            Calibrated ensemble probability
        """
        calibrated_probs = {}
        total_weight = 0

        for model, raw_prob in model_probabilities.items():
            calibrated_prob = self.calibrate_probability(model, raw_prob)
            calibrated_probs[model] = calibrated_prob

            weight = MODEL_WEIGHTS.get(model, 1.0)
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Fallback

        # Weighted average of calibrated probabilities
        ensemble_prob = sum(
            prob * MODEL_WEIGHTS.get(model, 1.0)
            for model, prob in calibrated_probs.items()
        ) / total_weight

        ensemble_prob = max(0.001, min(0.999, ensemble_prob))

        logger.debug(f"🎯 Ensemble calibration: {list(calibrated_probs.values())} → {ensemble_prob:.1%}")
        return ensemble_prob

    def get_calibration_status(self) -> Dict[str, bool]:
        """Get which models have calibration functions available."""
        return {
            model: (model in self.adjustment_functions)
            for model in ["claude", "gemini", "gpt", "deepseek", "perplexity"]
        }


# Global instances
model_calibration = ModelCalibration()
auto_calibrator = AutoCalibrator(model_calibration)
