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
        Execute the full rescue agent pipeline.

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
            "RescueAgent execution started",
            extra={
                "report_id": report_id,
                "latitude": request.latitude,
                "longitude": request.longitude,
                "has_description": bool(request.description),
            }
        )

        # ── Step 1: Read and validate image ──────────────────────────────────
        image_data = await self._read_and_validate_image(image, report_id)

        # ── Step 2: Upload to GCS ─────────────────────────────────────────────
        image_url = await self._upload_image(
            image_data, image.filename or "upload.jpg", report_id
        )

        # ── Step 3: Analyze with AI (OpenAI primary → Gemini fallback) ───────
        analysis = await self._ai.analyze_animal_image(image_data, report_id)

        # ── Step 4: Determine priority ────────────────────────────────────────
        priority = self._decision.determine_priority(analysis)
        logger.info(
            "Priority determined",
            extra={
                "report_id": report_id,
                "priority": priority,
                "label": self._decision.get_priority_label(priority),
            }
        )

        # ── Steps 5 & 6: Nearest vet + rescuer (parallel) ─────────────────────
        nearest_vet, nearest_rescuer = await asyncio.gather(
            self._location.find_nearest_vet(request.latitude, request.longitude),
            self._location.find_nearest_rescuer(request.latitude, request.longitude),
        )

        logger.info(
            "Locations resolved",
            extra={
                "report_id": report_id,
                "vet": nearest_vet.name,
                "vet_km": nearest_vet.distance_km,
                "rescuer": nearest_rescuer.name,
                "rescuer_km": nearest_rescuer.distance_km,
            }
        )

        # ── Step 7: Generate rescue plan (OpenAI primary → Gemini fallback) ──
        rescue_plan = await self._ai.generate_rescue_plan(
            species=analysis.species,
            injuries=analysis.injuries,
            severity=analysis.severity,
            first_aid=analysis.first_aid,
            vet_distance=nearest_vet.distance_km,
            rescuer_distance=nearest_rescuer.distance_km,
            report_id=report_id,
        )

        # ── Step 8: Assemble report ───────────────────────────────────────────
        report = RescueReport(
            status="success",
            report_id=report_id,
            timestamp=get_current_timestamp(),
            image_url=image_url,
            analysis=analysis,
            priority=priority,
            nearest_vet=nearest_vet,
            nearest_rescuer=nearest_rescuer,
            rescue_plan=rescue_plan,
        )

        # ── Step 9: Persist to Firestore ──────────────────────────────────────
        await self._firestore.save_report(report)

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            "RescueAgent execution complete",
            extra={
                "report_id": report_id,
                "elapsed_seconds": elapsed,
                "priority": priority,
                "species": analysis.species,
            }
        )

        return report

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
        image_name: str,
        report_id: str,
    ) -> str:
        """Upload image to GCS and return public URL."""
        try:
            return await self._storage.upload_image(image_data, image_name, report_id)
        except ImageUploadException:
            raise
        except Exception as e:
            raise AgentExecutionException(
                "Failed to upload image during agent execution",
                details={"error": str(e), "report_id": report_id},
            )
