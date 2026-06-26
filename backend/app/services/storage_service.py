"""
Google Cloud Storage service for image uploads.
Handles upload, public URL generation, and cleanup.
"""
import asyncio
import mimetypes
from datetime import datetime, timezone

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from app.config import get_settings
from app.utils.exceptions import ImageUploadException
from app.utils.helpers import sanitize_filename
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StorageService:
    """
    Service for uploading images to Google Cloud Storage.
    Returns public URLs for uploaded files.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = storage.Client(project=self._settings.gcp_project_id)
        self._bucket = self._client.bucket(self._settings.gcs_bucket_name)

    async def upload_image(
        self,
        image_data: bytes,
        original_filename: str,
        report_id: str
    ) -> str:
        """
        Upload image to GCS and return the public URL.

        Args:
            image_data: Raw image bytes
            original_filename: Original filename from the upload
            report_id: Report ID for unique path generation

        Returns:
            Public URL of the uploaded image

        Raises:
            ImageUploadException: If upload fails
        """
        logger.info(
            "Uploading image to GCS",
            extra={
                "report_id": report_id,
                "bucket": self._settings.gcs_bucket_name,
                "file_size_bytes": len(image_data)
            }
        )

        try:
            # Build a safe, unique blob name
            safe_filename = sanitize_filename(original_filename)
            date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            blob_name = f"rescue-images/{date_prefix}/{report_id}/{safe_filename}"

            # Determine content type
            content_type, _ = mimetypes.guess_type(original_filename)
            if content_type is None or not content_type.startswith("image/"):
                content_type = "image/jpeg"

            # Upload
            # Upload in thread pool
            blob = self._bucket.blob(blob_name)
            await asyncio.to_thread(
                blob.upload_from_string,
                image_data,
                content_type=content_type,
                timeout=60
            )

            # Build public URL directly — relies on bucket-level allUsers:objectViewer
            # IAM policy set during setup. Avoids make_public() which uses legacy ACLs
            # and fails on buckets with uniform access or publicAccessPrevention.
            public_url = f"https://storage.googleapis.com/{self._settings.gcs_bucket_name}/{blob_name}"

            logger.info(
                "Image uploaded successfully",
                extra={
                    "report_id": report_id,
                    "blob_name": blob_name,
                    "public_url": public_url
                }
            )

            return public_url

        except GoogleCloudError as e:
            logger.error(
                "GCS upload failed",
                extra={
                    "report_id": report_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "bucket": self._settings.gcs_bucket_name,
                    "project": self._settings.gcp_project_id,
                }
            )
            raise ImageUploadException(
                f"GCS upload failed: {str(e)}",
                details={"gcs_error": str(e), "report_id": report_id}
            )
        except Exception as e:
            logger.error(
                "Unexpected error during image upload",
                extra={
                    "report_id": report_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "bucket": self._settings.gcs_bucket_name,
                    "project": self._settings.gcp_project_id,
                },
                exc_info=True
            )
            raise ImageUploadException(
                f"Unexpected GCS error: {str(e)}",
                details={"error": str(e), "report_id": report_id}
            )

    async def delete_image(self, blob_name: str, report_id: str) -> None:
        """
        Delete an image from GCS (used for cleanup on failures).

        Args:
            blob_name: GCS blob path to delete
            report_id: Report ID for logging
        """
        try:
            blob = self._bucket.blob(blob_name)
            await asyncio.to_thread(blob.delete)
            logger.info(
                "Image deleted from GCS",
                extra={"report_id": report_id, "blob_name": blob_name}
            )
        except Exception as e:
            logger.warning(
                "Failed to delete image from GCS",
                extra={"report_id": report_id, "blob_name": blob_name, "error": str(e)}
            )
