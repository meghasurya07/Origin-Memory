"""
Prediction Error Engine — Friston FEP (2005) / Google Titans (2024)

Implements surprise-gated memory encoding. Only information that violates
the brain's current predictions gets stored — routine, expected input is
discarded. This mirrors the NMDA receptor coincidence detection and the
Free Energy Principle.

Brain analogue: Hippocampal CA1 mismatch detection, dopaminergic VTA signaling
Neuroscience: Prediction error, precision weighting, BCM theory

Mathematical foundation (Doc 25 §2):
    Δθ ∝ π · ε · ∂x̂/∂θ
    where ε = x - x̂ (prediction error)
          π = 1/σ² (precision = inverse variance)
    
    High precision × high error = STRONG encoding (novel signal in reliable context)
    Low precision × any error = WEAK encoding (noisy environment)
"""

from __future__ import annotations

import math
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """Result of a prediction error computation."""
    raw_error: float              # ||x - x̂|| (raw distance)
    precision: float              # π = estimated reliability of the prediction
    weighted_error: float         # π * ε (precision-weighted prediction error)
    surprise_score: float         # Normalized surprise in [0, 1]
    should_encode: bool           # Whether this exceeds the encoding threshold
    encoding_strength: float      # How strongly to encode (0.0 - 1.0)
    prediction_source: str        # What model/heuristic generated the prediction


class PredictionEngineConfig(BaseModel):
    """Configuration for the Prediction Error Engine."""
    surprise_threshold: float = 0.3     # Minimum surprise to trigger encoding
    high_surprise_threshold: float = 0.7  # Threshold for "flashbulb" strong encoding
    precision_window: int = 50          # Number of recent inputs to estimate precision
    min_precision: float = 0.1          # Floor precision (prevent division by zero)
    max_precision: float = 10.0         # Ceiling precision (prevent extreme values)
    adaptation_rate: float = 0.05       # How fast the prediction model adapts
    use_content_prediction: bool = True # Use content-based prediction (vs only embedding)


class PredictionEngine:
    """
    Implements prediction-error-based memory gating.
    
    The engine maintains a running model of "what the agent expects to see."
    When new input arrives, it computes the prediction error — how much the
    input diverges from the expectation. Only sufficiently surprising inputs
    pass the encoding gate.
    
    This implements Pillar 1 of the Origin Brain thesis:
    "Prediction Error Encoding — Only store what is surprising."
    
    Usage:
        engine = PredictionEngine()
        
        # Check if new input is surprising enough to encode
        result = engine.evaluate("The sky is blue")          # Low surprise
        result = engine.evaluate("The sky turned green!")     # High surprise
        
        if result.should_encode:
            brain.encode(content, salience=result.encoding_strength)
    """

    def __init__(self, config: Optional[PredictionEngineConfig] = None):
        self.config = config or PredictionEngineConfig()
        
        # Running statistics for precision estimation
        self._recent_errors: deque[float] = deque(maxlen=self.config.precision_window)
        self._error_mean: float = 0.5
        self._error_variance: float = 0.25
        
        # Content prediction model: tracks frequent patterns
        self._content_patterns: Dict[str, int] = {}
        self._total_inputs: int = 0
        
        # Category-specific prediction models
        self._category_means: Dict[str, float] = {}
        self._category_counts: Dict[str, int] = {}
        
        # History for adaptation
        self._input_history: deque[str] = deque(maxlen=200)
        
        # Embedding-based prediction (optional external)
        self._embedding_fn: Optional[Callable[[str], List[float]]] = None
        self._recent_embeddings: deque[List[float]] = deque(maxlen=self.config.precision_window)
        self._predicted_embedding: Optional[List[float]] = None

        logger.info(f"PredictionEngine initialized (threshold={self.config.surprise_threshold})")

    def set_embedding_fn(self, fn: Callable[[str], List[float]]) -> None:
        """Set an external embedding function for semantic prediction."""
        self._embedding_fn = fn

    def _compute_content_surprise(self, content: str) -> float:
        """
        Compute content-based surprise using pattern frequency.
        
        Rare patterns = high surprise. Frequent patterns = low surprise.
        This mirrors the neural mechanism where repeated stimuli cause
        adaptation (reduced firing) while novel stimuli cause strong responses.
        """
        words = content.lower().split()
        if not words:
            return 0.5
        
        # Compute word-level surprisal: -log2(P(word))
        total_surprisal = 0.0
        for word in words:
            count = self._content_patterns.get(word, 0)
            if self._total_inputs > 0 and count > 0:
                prob = count / max(self._total_inputs, 1)
                # Clamp to avoid log(0)
                prob = max(prob, 1e-10)
                total_surprisal += -math.log2(prob)
            else:
                # Completely novel word
                total_surprisal += 10.0  # High surprisal for unknown

        # Average surprisal per word
        avg_surprisal = total_surprisal / len(words)
        
        # Normalize to [0, 1] using sigmoid-like scaling
        # 10 bits of surprisal ≈ 1.0, 0 bits ≈ 0.0
        normalized = 1.0 / (1.0 + math.exp(-0.5 * (avg_surprisal - 5.0)))
        
        return normalized

    def _compute_embedding_surprise(self, content: str) -> float:
        """
        Compute surprise based on embedding distance from predicted embedding.
        
        The predicted embedding is the running centroid of recent inputs.
        Large deviation = high surprise.
        """
        if self._embedding_fn is None:
            return 0.5  # Neutral if no embedding function

        try:
            embedding = self._embedding_fn(content)
        except Exception:
            return 0.5

        if self._predicted_embedding is None or not self._recent_embeddings:
            self._recent_embeddings.append(embedding)
            self._predicted_embedding = list(embedding)
            return 0.5  # First input, no reference

        # Compute cosine distance from predicted embedding
        dot = sum(a * b for a, b in zip(embedding, self._predicted_embedding))
        norm_a = math.sqrt(sum(a * a for a in embedding))
        norm_b = math.sqrt(sum(b * b for b in self._predicted_embedding))
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            cosine_sim = 0.0
        else:
            cosine_sim = dot / (norm_a * norm_b)
        
        # Distance = 1 - cosine similarity
        distance = 1.0 - max(-1.0, min(1.0, cosine_sim))
        
        # Update predicted embedding (exponential moving average)
        alpha = self.config.adaptation_rate
        self._predicted_embedding = [
            (1.0 - alpha) * pred + alpha * curr
            for pred, curr in zip(self._predicted_embedding, embedding)
        ]
        self._recent_embeddings.append(embedding)
        
        # Normalize distance to [0, 1]
        return min(1.0, distance * 2.0)

    def _estimate_precision(self) -> float:
        """
        Estimate the current precision (inverse variance of recent errors).
        
        High precision = the environment is stable and predictions are reliable.
        Low precision = the environment is noisy, predictions are unreliable.
        
        In a noisy environment, even large errors should be treated cautiously.
        In a stable environment, even moderate errors signal genuine novelty.
        """
        if len(self._recent_errors) < 3:
            return 1.0  # Default precision

        errors = list(self._recent_errors)
        n = len(errors)
        mean = sum(errors) / n
        variance = sum((e - mean) ** 2 for e in errors) / max(n - 1, 1)
        
        # Precision = 1/variance (clamped)
        if variance < 1e-10:
            precision = self.config.max_precision
        else:
            precision = 1.0 / variance
        
        # Clamp
        precision = max(self.config.min_precision, min(self.config.max_precision, precision))
        
        self._error_mean = mean
        self._error_variance = variance
        
        return precision

    def _update_content_model(self, content: str) -> None:
        """Update the content prediction model with new input."""
        words = content.lower().split()
        for word in words:
            self._content_patterns[word] = self._content_patterns.get(word, 0) + 1
        self._total_inputs += 1
        self._input_history.append(content)

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> PredictionResult:
        """
        Evaluate whether new input should be encoded based on prediction error.
        
        This is the core gate: it computes how surprising the input is
        relative to the agent's current predictive model, then decides
        whether the surprise is sufficient to warrant memory encoding.
        
        Args:
            content: The text content to evaluate
            context: Optional context information
            category: Optional category for category-specific prediction
            
        Returns:
            PredictionResult with surprise score and encoding decision
        """
        # Step 1: Compute raw prediction error from multiple signals
        content_surprise = self._compute_content_surprise(content)
        embedding_surprise = self._compute_embedding_surprise(content)
        
        # Combine signals (weighted average)
        if self._embedding_fn is not None:
            raw_error = 0.4 * content_surprise + 0.6 * embedding_surprise
            source = "content+embedding"
        else:
            raw_error = content_surprise
            source = "content_only"

        # Step 2: Category-specific adjustment
        if category and category in self._category_means:
            # How much does this deviate from the category's typical error?
            cat_mean = self._category_means[category]
            category_deviation = abs(raw_error - cat_mean)
            raw_error = 0.7 * raw_error + 0.3 * category_deviation

        # Step 3: Estimate precision (reliability of predictions)
        precision = self._estimate_precision()
        
        # Step 4: Compute precision-weighted prediction error
        weighted_error = precision * raw_error
        
        # Step 5: Normalize to surprise score [0, 1]
        # Using sigmoid: maps (-∞, +∞) → (0, 1)
        surprise_score = 1.0 / (1.0 + math.exp(-3.0 * (weighted_error - 0.5)))
        
        # Step 6: Encoding decision
        should_encode = surprise_score >= self.config.surprise_threshold
        
        # Step 7: Encoding strength (how strongly to encode)
        if surprise_score >= self.config.high_surprise_threshold:
            encoding_strength = 1.0  # Flashbulb encoding
        elif should_encode:
            # Linear interpolation between threshold and high threshold
            range_size = self.config.high_surprise_threshold - self.config.surprise_threshold
            if range_size > 0:
                encoding_strength = 0.3 + 0.7 * (
                    (surprise_score - self.config.surprise_threshold) / range_size
                )
            else:
                encoding_strength = 0.5
        else:
            encoding_strength = 0.0
        
        # Step 8: Update internal models
        self._recent_errors.append(raw_error)
        self._update_content_model(content)
        
        if category:
            old_count = self._category_counts.get(category, 0)
            old_mean = self._category_means.get(category, 0.5)
            new_count = old_count + 1
            new_mean = (old_mean * old_count + raw_error) / new_count
            self._category_means[category] = new_mean
            self._category_counts[category] = new_count

        result = PredictionResult(
            raw_error=raw_error,
            precision=precision,
            weighted_error=weighted_error,
            surprise_score=surprise_score,
            should_encode=should_encode,
            encoding_strength=encoding_strength,
            prediction_source=source
        )

        logger.debug(
            f"Prediction evaluation: surprise={surprise_score:.3f}, "
            f"encode={should_encode}, strength={encoding_strength:.3f}"
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_inputs": self._total_inputs,
            "vocabulary_size": len(self._content_patterns),
            "recent_error_mean": self._error_mean,
            "recent_error_variance": self._error_variance,
            "current_precision": self._estimate_precision(),
            "category_count": len(self._category_means),
            "has_embedding_fn": self._embedding_fn is not None,
        }
