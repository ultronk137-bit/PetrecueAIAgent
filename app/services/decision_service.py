"""
Decision service for determining rescue priority.
Encapsulates all business logic for priority scoring.
"""
from app.models.response_models import AnimalAnalysis
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionService:
    """
    Service responsible for determining rescue priority score.
    Priority ranges from 1 (low) to 5 (critical emergency).
    """

    # Severity weights
    _SEVERITY_SCORES: dict[str, int] = {
        "critical": 5,
        "high": 4,
        "moderate": 3,
        "low": 2,
        "none": 1,
    }

    # Injury count impact thresholds
    _INJURY_THRESHOLDS = [
        (3, 1),   # 3+ injuries → +1 to priority
        (1, 0),   # 1-2 injuries → no change
    ]

    def determine_priority(self, analysis: AnimalAnalysis) -> int:
        """
        Determine rescue priority based on injury analysis.

        Scoring factors:
          1. Severity level (primary factor)
          2. Number of identified injuries
          3. Confidence level (low confidence reduces priority)

        Args:
            analysis: Completed AnimalAnalysis

        Returns:
            Priority score between 1 and 5
        """
        # Base score from severity
        base_score = self._SEVERITY_SCORES.get(analysis.severity.lower(), 3)

        # Injury count modifier
        injury_modifier = 0
        injury_count = len(analysis.injuries)
        for threshold, modifier in self._INJURY_THRESHOLDS:
            if injury_count >= threshold:
                injury_modifier = modifier
                break

        # Confidence modifier
        confidence_modifier = self._confidence_modifier(analysis.confidence)

        # Compute final score clamped to 1–5
        raw_score = base_score + injury_modifier + confidence_modifier
        final_score = max(1, min(5, raw_score))

        logger.info(
            "Priority determined",
            extra={
                "severity": analysis.severity,
                "injury_count": injury_count,
                "confidence": analysis.confidence,
                "base_score": base_score,
                "injury_modifier": injury_modifier,
                "confidence_modifier": confidence_modifier,
                "final_priority": final_score,
            }
        )

        return final_score

    def _confidence_modifier(self, confidence: float) -> int:
        """
        Return score modifier based on confidence.

        - High confidence (≥0.8): no adjustment
        - Medium confidence (0.5–0.79): no adjustment
        - Low confidence (<0.5): reduce by 1 (uncertain analysis should not over-escalate)
        """
        if confidence < 0.5:
            return -1
        return 0

    def get_priority_label(self, priority: int) -> str:
        """
        Get a human-readable label for a priority score.

        Args:
            priority: Integer priority score (1-5)

        Returns:
            Priority label string
        """
        labels = {
            5: "CRITICAL — Immediate emergency response required",
            4: "HIGH — Urgent care needed within 1-2 hours",
            3: "MODERATE — Care needed within a few hours",
            2: "LOW — Monitor and seek care when possible",
            1: "MINIMAL — No immediate intervention required",
        }
        return labels.get(priority, "UNKNOWN")
