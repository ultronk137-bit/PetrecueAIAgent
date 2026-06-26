"""
AI provider router — primary/fallback orchestration.
Tries OpenAI first; on any failure automatically falls back to Gemini.
RescueAgent imports only this class and is completely unaware of which
provider is actually serving the request.
"""
from typing import Any

from app.models.response_models import AnimalAnalysis, RescuePlan
from app.services.ai_service import AIService
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService
from app.utils.exceptions import AIException, GeminiException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AIProvider(AIService):
    """
    Composite AI service that implements the AIService interface.

    Execution strategy:
        1. Call OpenAI (primary)
        2. If OpenAI raises any exception → call Gemini (fallback)
        3. If Gemini also fails → re-raise the Gemini exception
    """

    def __init__(self) -> None:
        self._primary = OpenAIService()
        self._fallback = GeminiService()

    async def analyze_animal_image(
        self,
        image_data: bytes,
        report_id: str,
    ) -> AnimalAnalysis:
        """
        Analyze animal image. Tries OpenAI first, falls back to Gemini.
        """
        try:
            logger.info(
                "AIProvider: attempting analysis with OpenAI (primary)",
                extra={"report_id": report_id}
            )
            result = await self._primary.analyze_animal_image(image_data, report_id)
            logger.info(
                "AIProvider: OpenAI analysis succeeded",
                extra={"report_id": report_id}
            )
            return result

        except Exception as primary_error:
            logger.warning(
                "AIProvider: OpenAI failed — falling back to Gemini",
                extra={"report_id": report_id, "openai_error": str(primary_error)}
            )

        # Fallback — let Gemini exceptions propagate naturally
        result = await self._fallback.analyze_animal_image(image_data, report_id)
        logger.info(
            "AIProvider: Gemini fallback analysis succeeded",
            extra={"report_id": report_id}
        )
        return result

    async def generate_rescue_plan(
        self,
        species: str,
        injuries: list[str],
        severity: str,
        first_aid: list[str],
        vet_distance: float,
        rescuer_distance: float,
        report_id: str,
    ) -> RescuePlan:
        """
        Generate rescue plan. Tries OpenAI first, falls back to Gemini.
        """
        try:
            logger.info(
                "AIProvider: attempting rescue plan with OpenAI (primary)",
                extra={"report_id": report_id}
            )
            result = await self._primary.generate_rescue_plan(
                species, injuries, severity, first_aid,
                vet_distance, rescuer_distance, report_id
            )
            logger.info(
                "AIProvider: OpenAI rescue plan succeeded",
                extra={"report_id": report_id}
            )
            return result

        except Exception as primary_error:
            logger.warning(
                "AIProvider: OpenAI failed — falling back to Gemini for rescue plan",
                extra={"report_id": report_id, "openai_error": str(primary_error)}
            )

        result = await self._fallback.generate_rescue_plan(
            species, injuries, severity, first_aid,
            vet_distance, rescuer_distance, report_id
        )
        logger.info(
            "AIProvider: Gemini fallback rescue plan succeeded",
            extra={"report_id": report_id}
        )
        return result

    async def reason_next_action(
        self,
        goal: str,
        completed_steps: list[str],
        available_tools: list[dict[str, str]],
        report_id: str,
    ) -> dict[str, Any]:
        """
        Orchestrate reasoning: try OpenAI first, fall back to Gemini.
        """
        try:
            logger.info(
                "AIProvider: attempting reasoning with OpenAI (primary)",
                extra={"report_id": report_id}
            )
            result = await self._primary.reason_next_action(
                goal, completed_steps, available_tools, report_id
            )
            logger.info(
                "AIProvider: OpenAI reasoning succeeded",
                extra={"report_id": report_id}
            )
            return result

        except Exception as primary_error:
            logger.warning(
                "AIProvider: OpenAI reasoning failed — falling back to Gemini",
                extra={"report_id": report_id, "openai_error": str(primary_error)}
            )

        result = await self._fallback.reason_next_action(
            goal, completed_steps, available_tools, report_id
        )
        logger.info(
            "AIProvider: Gemini fallback reasoning succeeded",
            extra={"report_id": report_id}
        )
        return result
