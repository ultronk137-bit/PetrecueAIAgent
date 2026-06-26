"""
Request models using Pydantic v2 for validation.
"""
from typing import Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field, field_validator


class RescueRequest(BaseModel):
    """Request model for rescue report generation."""
    
    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude coordinate of rescue location"
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude coordinate of rescue location"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional description of the situation"
    )
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize description."""
        if v:
            v = v.strip()
            if not v:
                return None
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    "description": "Found injured dog near Golden Gate Park"
                }
            ]
        }
    }


async def parse_rescue_request(
    image: UploadFile = File(..., description="Image of the injured animal"),
    latitude: float = Form(..., ge=-90, le=90, description="Latitude coordinate"),
    longitude: float = Form(..., ge=-180, le=180, description="Longitude coordinate"),
    description: Optional[str] = Form(None, max_length=1000, description="Optional description")
) -> tuple[UploadFile, RescueRequest]:
    """
    Parse multipart form data into structured request.
    
    Args:
        image: Uploaded image file
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        description: Optional description
        
    Returns:
        Tuple of (image, request_data)
    """
    request_data = RescueRequest(
        latitude=latitude,
        longitude=longitude,
        description=description
    )
    
    return image, request_data
