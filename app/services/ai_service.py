"""
Abstract AI service interface.
RescueAgent depends only on this contract — provider details are invisible to it.
"""
from abc import ABC, abstractmethod
from typing import Any

from app.models.response_models import AnimalAnalysis, RescuePlan


class AIService(ABC):
    """
    Abstract base class for AI analysis providers.
    Implement this to add any new provider (OpenAI, Gemini, Claude, etc.)
    without touching the agent or any other layer.
    """

    @abstractmethod
    async def analyze_animal_image(
        self,
        image_data: bytes,
        report_id: str,
    ) -> AnimalAnalysis:
        """
        Analyze an image of an injured animal.

        Args:
            image_data: Raw image bytes
            report_id: Correlation ID for logging

        Returns:
            Structured AnimalAnalysis
        """

    @abstractmethod
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
        Generate a rescue plan based on analysis results.

        Args:
            species: Identified species
            injuries: List of injuries
            severity: Severity level
            first_aid: First aid instructions already given
            vet_distance: Distance to nearest vet in km
            rescuer_distance: Distance to nearest rescuer in km
            report_id: Correlation ID for logging

        Returns:
            Structured RescuePlan
        """

    @abstractmethod
    async def reason_next_action(
        self,
        goal: str,
        completed_steps: list[str],
        available_tools: list[dict[str, str]],
        report_id: str,
    ) -> dict[str, Any]:
        """
        Reason about the next action to perform using the agent loop.
        Returns a dict containing 'thought' and a list of 'tools' to call.
        """
