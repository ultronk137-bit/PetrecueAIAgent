"""
Gemini 2.0 Flash service — fallback AI provider.
Used automatically when OpenAI fails or is unavailable.
"""
import asyncio
import time
from typing import Any

import google.generativeai as genai
from PIL import Image
from io import BytesIO

from app.config import get_settings
from app.models.response_models import AnimalAnalysis, RescuePlan
from app.prompts.ai_prompts import (
    ANIMAL_ANALYSIS_SYSTEM_PROMPT,
    ANIMAL_ANALYSIS_USER_PROMPT,
    RESCUE_PLAN_SYSTEM_PROMPT,
    RESCUE_PLAN_USER_PROMPT,
)
from app.services.ai_service import AIService
from app.utils.exceptions import GeminiException
from app.utils.helpers import extract_json_from_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

AIProviderException = GeminiException


class GeminiService(AIService):
    """
    AI service backed by Google Gemini 2.0 Flash.
    Acts as the fallback when OpenAI raises an exception.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = self._initialize_client()
        logger.info(
            "GeminiService initialized",
            extra={"model": self._settings.gemini_model}
        )

    def _initialize_client(self) -> genai.GenerativeModel:
        genai.configure(api_key=self._settings.gemini_api_key)
        return genai.GenerativeModel(
            model_name=self._settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=10,
                max_output_tokens=1024,
            ),
        )

    async def analyze_animal_image(
        self,
        image_data: bytes,
        report_id: str,
    ) -> AnimalAnalysis:
        """
        Analyze an animal image using Gemini Vision.

        Raises:
            AIProviderException: After exhausting retries
        """
        logger.info(
            "Gemini: starting image analysis (fallback)",
            extra={"report_id": report_id}
        )

        combined_prompt = f"{ANIMAL_ANALYSIS_SYSTEM_PROMPT}\n\n{ANIMAL_ANALYSIS_USER_PROMPT}"
        pil_image = Image.open(BytesIO(image_data))
        last_exc: Exception | None = None

        for attempt in range(1, self._settings.ai_max_retries + 1):
            try:
                start = time.time()
                response = await self._model.generate_content_async([combined_prompt, pil_image])
                elapsed = round(time.time() - start, 2)
                raw = response.text

                logger.info(
                    "Gemini: response received",
                    extra={
                        "report_id": report_id,
                        "attempt": attempt,
                        "elapsed_seconds": elapsed,
                    }
                )

                parsed = extract_json_from_text(raw)
                if parsed is None:
                    raise AIProviderException(
                        "Gemini returned non-JSON response",
                        details={"raw": raw[:500], "attempt": attempt}
                    )

                analysis = self._build_analysis(parsed)

                logger.info(
                    "Gemini: analysis complete",
                    extra={
                        "report_id": report_id,
                        "species": analysis.species,
                        "severity": analysis.severity,
                    }
                )

                return analysis

            except AIProviderException as e:
                last_exc = e
                logger.warning(
                    "Gemini: JSON parse failed, retrying",
                    extra={"report_id": report_id, "attempt": attempt}
                )
                if attempt < self._settings.ai_max_retries:
                    await asyncio.sleep(1)

            except Exception as e:
                last_exc = e
                logger.error(
                    "Gemini: API error",
                    extra={"report_id": report_id, "attempt": attempt, "error": str(e)}
                )
                if attempt < self._settings.ai_max_retries:
                    await asyncio.sleep(2)

        raise AIProviderException(
            "Gemini image analysis failed after all retries",
            details={"last_error": str(last_exc)}
        )

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
        Generate a rescue plan using Gemini.

        Raises:
            AIProviderException: After exhausting retries
        """
        logger.info(
            "Gemini: generating rescue plan (fallback)",
            extra={"report_id": report_id}
        )

        user_prompt = RESCUE_PLAN_USER_PROMPT.format(
            species=species,
            injuries=", ".join(injuries),
            severity=severity,
            first_aid=", ".join(first_aid),
            vet_distance=round(vet_distance, 1),
            rescuer_distance=round(rescuer_distance, 1),
        )
        combined_prompt = f"{RESCUE_PLAN_SYSTEM_PROMPT}\n\n{user_prompt}"
        last_exc: Exception | None = None

        for attempt in range(1, self._settings.ai_max_retries + 1):
            try:
                response = await self._model.generate_content_async(combined_prompt)
                raw = response.text
                parsed = extract_json_from_text(raw)

                if parsed is None:
                    raise AIProviderException(
                        "Gemini returned non-JSON rescue plan",
                        details={"raw": raw[:500]}
                    )

                plan = RescuePlan(
                    immediate_actions=parsed.get("immediate_actions", []),
                    transport_instructions=parsed.get(
                        "transport_instructions",
                        "Transport carefully to the nearest veterinary facility."
                    ),
                    what_to_bring=parsed.get("what_to_bring", []),
                    precautions=parsed.get("precautions", []),
                    estimated_time=parsed.get("estimated_time", "30-60 minutes"),
                )

                logger.info("Gemini: rescue plan generated", extra={"report_id": report_id})
                return plan

            except AIProviderException as e:
                last_exc = e
                if attempt < self._settings.ai_max_retries:
                    await asyncio.sleep(1)

            except Exception as e:
                last_exc = e
                if attempt < self._settings.ai_max_retries:
                    await asyncio.sleep(2)

        raise AIProviderException(
            "Gemini rescue plan generation failed after all retries",
            details={"last_error": str(last_exc)}
        )

    def _build_analysis(self, parsed: dict[str, Any]) -> AnimalAnalysis:
        valid_severities = {"critical", "high", "moderate", "low", "none"}
        severity = parsed.get("severity", "moderate").lower()
        if severity not in valid_severities:
            severity = "moderate"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        injuries = parsed.get("injuries", [])
        if isinstance(injuries, str):
            injuries = [injuries]

        first_aid = parsed.get("first_aid", [])
        if isinstance(first_aid, str):
            first_aid = [first_aid]

        return AnimalAnalysis(
            species=parsed.get("species", "Unknown animal"),
            injuries=injuries,
            severity=severity,
            confidence=confidence,
            first_aid=first_aid,
            additional_notes=parsed.get("additional_notes"),
        )

    async def reason_next_action(
        self,
        goal: str,
        completed_steps: list[str],
        available_tools: list[dict[str, str]],
        report_id: str,
    ) -> dict[str, Any]:
        """
        Reason about the next action using Gemini.
        """
        logger.info(
            "Gemini: reasoning next action",
            extra={"report_id": report_id, "completed_steps": completed_steps}
        )

        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in available_tools
        )
        
        system_prompt = (
            "You are the reasoning core of the PetRescue AI Agent.\n"
            "Your goal is to process the user's request and coordinate the execution of tools to generate a complete rescue report.\n\n"
            f"Goal: {goal}\n\n"
            "Available Tools:\n"
            f"{tools_desc}\n\n"
            "Currently completed steps:\n"
            f"{', '.join(completed_steps) if completed_steps else 'None'}\n\n"
            "Your job is to:\n"
            "1. Reason about the current progress.\n"
            "2. Decide which tool(s) should be executed next. You can choose to run multiple tools in parallel if they do not depend on each other.\n"
            "   - 'upload_and_analyze_image' and 'find_nearby_facilities' can run in parallel first.\n"
            "   - 'determine_priority' and 'generate_rescue_plan' depend on image analysis and facilities being complete.\n"
            "   - 'save_report' depends on all prior steps being complete.\n"
            "   - If all steps are complete, select ['finish'].\n\n"
            "3. Respond in strict JSON format:\n"
            "{\n"
            '  "thought": "Your reasoning text...",\n'
            '  "tools": ["tool_name_1", "tool_name_2"]\n'
            "}\n"
            "Do not return any other text, only valid JSON."
        )

        last_exc: Exception | None = None
        for attempt in range(1, self._settings.ai_max_retries + 1):
            try:
                response = await self._model.generate_content_async(
                    system_prompt + "\n\nDetermine the next action."
                )
                raw = response.text
                parsed = extract_json_from_text(raw)
                if parsed is None or "tools" not in parsed:
                    raise GeminiException("Gemini returned invalid JSON for reasoning")
                
                logger.info(
                    "Gemini: reasoning complete",
                    extra={"report_id": report_id, "thought": parsed.get("thought"), "tools": parsed.get("tools")}
                )
                return parsed

            except Exception as e:
                last_exc = e
                if attempt < self._settings.ai_max_retries:
                    await asyncio.sleep(1)

        raise GeminiException(
            "Gemini reasoning failed after all retries",
            details={"last_error": str(last_exc)}
        )
