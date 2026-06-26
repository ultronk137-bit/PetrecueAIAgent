"""
Firestore service for storing rescue reports.
Handles document creation and retrieval.
"""
from typing import Any

from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError

from app.config import get_settings
from app.models.response_models import RescueReport
from app.utils.exceptions import FirestoreException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FirestoreService:
    """
    Service for persisting rescue reports to Firestore.
    """

    _client: firestore.AsyncClient | None = None

    def __init__(self) -> None:
        self._settings = get_settings()
        if FirestoreService._client is None:
            FirestoreService._client = firestore.AsyncClient(project=self._settings.gcp_project_id)
        self._client = FirestoreService._client
        self._collection = self._client.collection(self._settings.firestore_collection)

    async def save_report(self, report: RescueReport) -> str:
        """
        Save a complete rescue report to Firestore.

        Args:
            report: Fully constructed RescueReport

        Returns:
            Document ID of the saved report

        Raises:
            FirestoreException: If save operation fails
        """
        logger.info(
            "Saving rescue report to Firestore",
            extra={
                "report_id": report.report_id,
                "collection": self._settings.firestore_collection
            }
        )

        try:
            # Serialize to dict for Firestore
            report_data = self._serialize_report(report)

            # Use report_id as document ID for easy retrieval
            doc_ref = self._collection.document(report.report_id)
            await doc_ref.set(report_data)

            logger.info(
                "Rescue report saved to Firestore",
                extra={
                    "report_id": report.report_id,
                    "document_path": doc_ref.path
                }
            )

            return report.report_id

        except GoogleCloudError as e:
            logger.error(
                "Firestore save failed",
                extra={
                    "report_id": report.report_id,
                    "error": str(e)
                }
            )
            raise FirestoreException(
                "Failed to save rescue report to Firestore",
                details={"firestore_error": str(e), "report_id": report.report_id}
            )
        except Exception as e:
            logger.error(
                "Unexpected Firestore error",
                extra={"report_id": report.report_id, "error": str(e)}
            )
            raise FirestoreException(
                "Unexpected error saving to Firestore",
                details={"error": str(e), "report_id": report.report_id}
            )

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """
        Retrieve a rescue report from Firestore by ID.

        Args:
            report_id: Report ID to retrieve

        Returns:
            Report data dict or None if not found
        """
        try:
            doc_ref = self._collection.document(report_id)
            doc = await doc_ref.get()

            if doc.exists:
                return doc.to_dict()

            return None

        except Exception as e:
            logger.error(
                "Failed to retrieve report from Firestore",
                extra={"report_id": report_id, "error": str(e)}
            )
            raise FirestoreException(
                "Failed to retrieve report",
                details={"error": str(e), "report_id": report_id}
            )

    def _serialize_report(self, report: RescueReport) -> dict[str, Any]:
        """
        Serialize RescueReport to a Firestore-compatible dict.
        Firestore does not support nested Pydantic models directly.

        Args:
            report: RescueReport to serialize

        Returns:
            Dict representation suitable for Firestore
        """
        return {
            "status": report.status,
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "image_url": report.image_url,
            "analysis": {
                "species": report.analysis.species,
                "injuries": report.analysis.injuries,
                "severity": report.analysis.severity,
                "confidence": report.analysis.confidence,
                "first_aid": report.analysis.first_aid,
                "additional_notes": report.analysis.additional_notes,
            },
            "priority": report.priority,
            "nearest_vet": {
                "name": report.nearest_vet.name,
                "address": report.nearest_vet.address,
                "distance_km": report.nearest_vet.distance_km,
                "phone": report.nearest_vet.phone,
                "latitude": report.nearest_vet.latitude,
                "longitude": report.nearest_vet.longitude,
            },
            "nearest_rescuer": {
                "name": report.nearest_rescuer.name,
                "address": report.nearest_rescuer.address,
                "distance_km": report.nearest_rescuer.distance_km,
                "phone": report.nearest_rescuer.phone,
                "latitude": report.nearest_rescuer.latitude,
                "longitude": report.nearest_rescuer.longitude,
            },
            "rescue_plan": {
                "immediate_actions": report.rescue_plan.immediate_actions,
                "transport_instructions": report.rescue_plan.transport_instructions,
                "what_to_bring": report.rescue_plan.what_to_bring,
                "precautions": report.rescue_plan.precautions,
                "estimated_time": report.rescue_plan.estimated_time,
            },
        }
