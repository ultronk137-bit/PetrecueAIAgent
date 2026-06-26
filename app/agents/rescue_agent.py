"""
RescueAgent — the core autonomous orchestrator.
Coordinates all services to produce a complete rescue report.
All business logic lives here; the API layer only validates and delegates.
"""
import asyncio
import time

from fastapi import UploadFile

from app.config import get_settings
from app.models.request_models import RescueRequest
from app.models.response_models import RescueReport
from app.services.ai_service import AIService
from app.services.decision_service import DecisionService
from app.services.firestore_service import FirestoreService
from app.services.location_service import LocationService
from app.services.storage_service import StorageService
from app.utils.exceptions import (
    AgentExecutionException,
    ImageUploadException,
    ValidationException,
)
from app.utils.helpers import (
    generate_report_id,
    get_current_timestamp,
    validate_image_format,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RescueAgent:
    """
    Autonomous agent that orchestrates the full rescue pipeline.

    Execution order:
        1.  Validate image
        2.  Upload image to GCS
        3.  Analyze image (OpenAI → Gemini fallback, via AIProvider)
        4.  Determine rescue priority
        5.  Locate nearest vet     ┐ parallel
        6.  Locate nearest rescuer ┘
        7.  Generate rescue plan (OpenAI → Gemini fallback)
        8.  Persist report to Firestore
        9.  Return structured RescueReport
    """

    def __init__(
        self,
        ai_service: AIService,
        storage_service: StorageService,
        firestore_service: FirestoreService,
        location_service: LocationService,
        decision_service: DecisionService,
    ) -> None:
        self._ai = ai_service
        self._storage = storage_service
        self._firestore = firestore_service
        self._location = location_service
        self._decision = decision_service
        self._settings = get_settings()

    async def execute(
        self,
        image: UploadFile,
        request: RescueRequest,
    ) -> RescueReport:
        """
        Execute the full rescue agent pipeline with an autonomous reasoning loop.

        Args:
            image:   Uploaded image file
            request: Validated rescue request with location and description

        Returns:
            Complete RescueReport

        Raises:
            ValidationException:     On invalid image input
            AgentExecutionException: On pipeline failure
        """
        report_id = generate_report_id()
        start_time = time.time()

        logger.info(
            "RescueAgent execution started (Reasoning Agent)",
            extra={
                "report_id": report_id,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "has_description": bool(request.description),
            }
        )

        available_tools = [
            {
                "name": "upload_and_analyze_image",
                "description": "Upload the animal photo and perform AI vision analysis. Returns animal species, injuries, and severity."
            },
            {
                "name": "find_nearby_facilities",
                "description": "Locate the nearest veterinary hospital and rescue organization based on coordinate location."
            },
            {
                "name": "determine_priority",
                "description": "Calculate the rescue priority level (1-5) based on the image analysis. Requires upload_and_analyze_image first."
            },
            {
                "name": "generate_rescue_plan",
                "description": "Formulate detailed first-aid and transport instruction plan. Requires upload_and_analyze_image and find_nearby_facilities."
            },
            {
                "name": "save_report",
                "description": "Persist the completed report to the Firestore database. Requires all other tools to be completed first."
            },
        ]

        completed_steps = []
        state = {
            "image_url": None,
            "analysis": None,
            "priority": None,
            "nearest_vet": None,
            "nearest_rescuer": None,
            "rescue_plan": None,
            "report_saved": False,
            "report": None,
        }

        goal = (
            f"Process a rescue request for an animal at location ({request.latitude}, {request.longitude}) "
            f"with description: {request.description or 'None'}. "
            "Complete all steps: upload and analyze image, find nearby facilities, determine priority, "
            "generate the rescue plan, and save the report to the database."
        )

        max_turns = 8
        for turn in range(1, max_turns + 1):
            # 1. Reason: Call LLM to decide the next tool(s)
            try:
                decision = await self._ai.reason_next_action(
                    goal=goal,
                    completed_steps=completed_steps,
                    available_tools=available_tools,
                    report_id=report_id,
                )
                thought = decision.get("thought", "")
                selected_tools = decision.get("tools", [])
            except Exception as e:
                logger.warning(
                    "Reasoning failed, switching to hardcoded fallback planning",
                    extra={"report_id": report_id, "error": str(e)}
                )
                # Fail-safe local planner if AI reasoning fails
                thought = "Local planner orchestrating next logical steps."
                if "upload_and_analyze_image" not in completed_steps or "find_nearby_facilities" not in completed_steps:
                    selected_tools = [t for t in ["upload_and_analyze_image", "find_nearby_facilities"] if t not in completed_steps]
                elif "determine_priority" not in completed_steps or "generate_rescue_plan" not in completed_steps:
                    selected_tools = [t for t in ["determine_priority", "generate_rescue_plan"] if t not in completed_steps]
                elif "save_report" not in completed_steps:
                    selected_tools = ["save_report"]
                else:
                    selected_tools = ["finish"]

            logger.info(
                f"Agent reasoning turn {turn}",
                extra={
                    "report_id": report_id,
                    "thought": thought,
                    "selected_tools": selected_tools,
                    "completed_steps": completed_steps,
                }
            )

            if "finish" in selected_tools or not selected_tools:
                break

            # 2. Execute selected tools (in parallel)
            tasks = []
            tools_to_run = []
            for tool in selected_tools:
                if tool == "upload_and_analyze_image" and "upload_and_analyze_image" not in completed_steps:
                    tasks.append(self._run_upload_and_analyze_image(image, report_id, state))
                    tools_to_run.append(tool)
                elif tool == "find_nearby_facilities" and "find_nearby_facilities" not in completed_steps:
                    tasks.append(self._run_find_nearby_facilities(request.latitude, request.longitude, state))
                    tools_to_run.append(tool)
                elif tool == "determine_priority" and "determine_priority" not in completed_steps:
                    tasks.append(self._run_determine_priority(state))
                    tools_to_run.append(tool)
                elif tool == "generate_rescue_plan" and "generate_rescue_plan" not in completed_steps:
                    tasks.append(self._run_generate_rescue_plan(report_id, state))
                    tools_to_run.append(tool)
                elif tool == "save_report" and "save_report" not in completed_steps:
                    tasks.append(self._run_save_report(report_id, state))
                    tools_to_run.append(tool)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for run_tool, res in zip(tools_to_run, results):
                    if isinstance(res, Exception):
                        logger.error(
                            f"Tool execution failed: {run_tool}",
                            extra={"report_id": report_id, "error": str(res)},
                            exc_info=True
                        )
                        raise res
                    completed_steps.append(run_tool)

        # 3. Post-loop fallback verification: Ensure all steps completed successfully
        if not state["report_saved"] or not state["report"]:
            logger.warning(
                "Orchestration loop ended without saving report. Running fallback completion.",
                extra={"report_id": report_id}
            )
            # Ensure prerequisites
            if not state["image_url"] or not state["analysis"]:
                await self._run_upload_and_analyze_image(image, report_id, state)
            if not state["nearest_vet"] or not state["nearest_rescuer"]:
                await self._run_find_nearby_facilities(request.latitude, request.longitude, state)
            if state["priority"] is None:
                await self._run_determine_priority(state)
            if not state["rescue_plan"]:
                await self._run_generate_rescue_plan(report_id, state)
            if not state["report_saved"]:
                await self._run_save_report(report_id, state)

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            "RescueAgent execution complete",
            extra={
                "report_id": report_id,
                "elapsed_seconds": elapsed,
                "priority": state["priority"],
                "species": state["analysis"].species if state["analysis"] else "Unknown",
            }
        )

        return state["report"]

    async def _run_upload_and_analyze_image(self, image: UploadFile, report_id: str, state: dict) -> None:
        image_data = await self._read_and_validate_image(image, report_id)
        image_url = await self._upload_image(image_data, image.filename or "upload.jpg", report_id)
        analysis = await self._ai.analyze_animal_image(image_data, report_id)
        state["image_url"] = image_url
        state["analysis"] = analysis

    async def _run_find_nearby_facilities(self, latitude: float, longitude: float, state: dict) -> None:
        nearest_vet, nearest_rescuer = await asyncio.gather(
            self._location.find_nearest_vet(latitude, longitude),
            self._location.find_nearest_rescuer(latitude, longitude),
        )
        state["nearest_vet"] = nearest_vet
        state["nearest_rescuer"] = nearest_rescuer

    async def _run_determine_priority(self, state: dict) -> None:
        if not state["analysis"]:
            raise AgentExecutionException("Cannot determine priority: animal analysis is missing.")
        state["priority"] = self._decision.determine_priority(state["analysis"])

    async def _run_generate_rescue_plan(self, report_id: str, state: dict) -> None:
        if not state["analysis"] or not state["nearest_vet"] or not state["nearest_rescuer"]:
            raise AgentExecutionException("Cannot generate rescue plan: required inputs are missing.")
        state["rescue_plan"] = await self._ai.generate_rescue_plan(
            species=state["analysis"].species,
            injuries=state["analysis"].injuries,
            severity=state["analysis"].severity,
            first_aid=state["analysis"].first_aid,
            vet_distance=state["nearest_vet"].distance_km,
            rescuer_distance=state["nearest_rescuer"].distance_km,
            report_id=report_id,
        )

    async def _run_save_report(self, report_id: str, state: dict) -> None:
        required = ["image_url", "analysis", "priority", "nearest_vet", "nearest_rescuer", "rescue_plan"]
        for field in required:
            if state[field] is None:
                raise AgentExecutionException(f"Cannot save report: {field} is missing.")

        report = RescueReport(
            status="success",
            report_id=report_id,
            timestamp=get_current_timestamp(),
            image_url=state["image_url"],
            analysis=state["analysis"],
            priority=state["priority"],
            nearest_vet=state["nearest_vet"],
            nearest_rescuer=state["nearest_rescuer"],
            rescue_plan=state["rescue_plan"],
        )
        await self._firestore.save_report(report)
        state["report"] = report
        state["report_saved"] = True

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _read_and_validate_image(
        self,
        image: UploadFile,
        report_id: str,
    ) -> bytes:
        """Read and validate uploaded image bytes."""
        content_type = image.content_type or ""
        if not content_type.startswith("image/"):
            raise ValidationException(
                f"Invalid content type: {content_type}. Expected an image file.",
                details={"content_type": content_type},
            )

        try:
            image_data = await image.read()
        except Exception as e:
            raise ValidationException(
                "Failed to read uploaded image",
                details={"error": str(e)},
            )

        max_bytes = self._settings.max_image_size_bytes
        if len(image_data) > max_bytes:
            raise ValidationException(
                f"Image exceeds the maximum allowed size of {self._settings.max_image_size_mb} MB",
                details={
                    "file_size_mb": round(len(image_data) / 1_048_576, 2),
                    "max_size_mb": self._settings.max_image_size_mb,
                },
            )

        is_valid, error_message = validate_image_format(image_data)
        if not is_valid:
            raise ValidationException(
                f"Invalid image file: {error_message}",
                details={"validation_error": error_message},
            )

        logger.info(
            "Image validated",
            extra={
                "report_id": report_id,
                "size_bytes": len(image_data),
                "content_type": content_type,
            }
        )

        return image_data

    async def _upload_image(
        self,
        image_data: bytes,
        filename: str,
        report_id: str,
    ) -> str:
        """Upload image to GCS and return public URL."""
        try:
            return await self._storage.upload_image(image_data, filename, report_id)
        except ImageUploadException:
            raise
        except Exception as e:
            raise AgentExecutionException(
                "Failed to upload image during agent execution",
                details={"error": str(e), "report_id": report_id},
            )
