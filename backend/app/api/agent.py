"""
Agent API endpoint.
The only responsibility here is request parsing and delegating to RescueAgent.
Zero business logic lives in this layer.
"""
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.agents.rescue_agent import RescueAgent
from app.models.request_models import RescueRequest
from app.models.response_models import RescueReport
from app.services.ai_provider import AIProvider
from app.services.decision_service import DecisionService
from app.services.firestore_service import FirestoreService
from app.services.location_service import LocationService
from app.services.storage_service import StorageService
from app.utils.exceptions import (
    AgentExecutionException,
    GeminiException,
    ImageUploadException,
    ValidationException,
)
from app.utils.helpers import get_current_timestamp
from app.utils.logger import get_logger

router = APIRouter(prefix="/agent", tags=["Agent"])
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dependency injection factory
# ---------------------------------------------------------------------------

def get_rescue_agent() -> RescueAgent:
    """
    Dependency that constructs and wires the RescueAgent with all services.
    AIProvider wires OpenAI (primary) → Gemini (fallback) transparently.
    """
    return RescueAgent(
        ai_service=AIProvider(),
        storage_service=StorageService(),
        firestore_service=FirestoreService(),
        location_service=LocationService(),
        decision_service=DecisionService(),
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/report",
    response_model=RescueReport,
    status_code=200,
    summary="Generate Rescue Report",
    description=(
        "Upload an image of an injured animal along with GPS coordinates. "
        "The agent will analyze the image, locate the nearest vet and rescue organization, "
        "generate a rescue plan, and return a complete report."
    ),
)
async def generate_rescue_report(
    image: UploadFile = File(
        ...,
        description="Image of the injured animal (JPEG, PNG, WEBP — max 10 MB)"
    ),
    latitude: float = Form(
        ...,
        ge=-90,
        le=90,
        description="Latitude of the rescue location"
    ),
    longitude: float = Form(
        ...,
        ge=-180,
        le=180,
        description="Longitude of the rescue location"
    ),
    description: Optional[str] = Form(
        None,
        max_length=1000,
        description="Optional description of the situation"
    ),
    agent: RescueAgent = Depends(get_rescue_agent),
) -> RescueReport:
    """
    Single endpoint for the entire rescue pipeline.
    Accepts multipart/form-data and returns a structured RescueReport.
    """
    logger.info(
        "Incoming rescue report request",
        extra={
            "latitude": latitude,
            "longitude": longitude,
            "image_filename": image.filename,
            "content_type": image.content_type,
            "has_description": bool(description),
        }
    )

    # Build structured request model
    request = RescueRequest(
        latitude=latitude,
        longitude=longitude,
        description=description,
    )

    try:
        report = await agent.execute(image=image, request=request)
        return report

    except ValidationException as e:
        logger.warning(
            "Validation error",
            extra={"error": e.message, "details": e.details}
        )
        raise HTTPException(
            status_code=400,
            detail={"message": e.message, "details": e.details}
        )

    except ImageUploadException as e:
        logger.error(
            "Image upload failed",
            extra={"error": e.message, "details": e.details}
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to upload image. Please try again.", "details": e.details}
        )

    except GeminiException as e:
        logger.error(
            "Gemini analysis failed",
            extra={"error": e.message, "details": e.details}
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "AI analysis failed. Please try again.", "details": e.details}
        )

    except AgentExecutionException as e:
        logger.error(
            "Agent execution failed",
            extra={"error": e.message, "details": e.details}
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "Rescue agent failed. Please try again.", "details": e.details}
        )

    except Exception as e:
        logger.error(
            "Unexpected error during rescue report generation",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={"message": "An unexpected error occurred. Please try again."}
        )
