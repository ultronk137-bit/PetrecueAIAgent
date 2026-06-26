"""
Response models using Pydantic v2 for structured outputs.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnimalAnalysis(BaseModel):
    """Analysis results from Gemini Vision."""
    
    species: str = Field(..., description="Identified animal species")
    injuries: list[str] = Field(default_factory=list, description="List of visible injuries")
    severity: str = Field(..., description="Injury severity level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    first_aid: list[str] = Field(default_factory=list, description="First aid instructions")
    additional_notes: Optional[str] = Field(None, description="Additional observations")


class Location(BaseModel):
    """Location information for vet or rescue organization."""
    
    name: str = Field(..., description="Name of facility")
    address: str = Field(..., description="Full address")
    distance_km: float = Field(..., description="Distance in kilometers")
    phone: Optional[str] = Field(None, description="Contact phone number")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")


class RescuePlan(BaseModel):
    """Generated rescue plan with recommendations."""
    
    immediate_actions: list[str] = Field(
        default_factory=list,
        description="Immediate actions to take"
    )
    transport_instructions: str = Field(..., description="How to safely transport the animal")
    what_to_bring: list[str] = Field(
        default_factory=list,
        description="Items to bring for rescue"
    )
    precautions: list[str] = Field(
        default_factory=list,
        description="Safety precautions"
    )
    estimated_time: str = Field(..., description="Estimated time for rescue operation")


class RescueReport(BaseModel):
    """Complete rescue report returned to client."""
    
    status: str = Field(default="success", description="Request status")
    report_id: str = Field(..., description="Unique report identifier")
    timestamp: str = Field(..., description="Report generation timestamp")
    image_url: str = Field(..., description="Public URL of uploaded image")
    
    analysis: AnimalAnalysis = Field(..., description="Animal analysis results")
    priority: int = Field(..., ge=1, le=5, description="Rescue priority (1-5)")
    
    nearest_vet: Location = Field(..., description="Nearest veterinary hospital")
    nearest_rescuer: Location = Field(..., description="Nearest rescue organization")
    
    rescue_plan: RescuePlan = Field(..., description="Generated rescue plan")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "report_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2026-06-26T13:45:00Z",
                    "image_url": "https://storage.googleapis.com/bucket/image.jpg",
                    "analysis": {
                        "species": "Domestic Dog (Labrador Retriever)",
                        "injuries": ["Laceration on right front leg", "Limping"],
                        "severity": "moderate",
                        "confidence": 0.87,
                        "first_aid": [
                            "Clean wound with saline solution",
                            "Apply gentle pressure to stop bleeding"
                        ],
                        "additional_notes": "Animal appears alert and responsive"
                    },
                    "priority": 3,
                    "nearest_vet": {
                        "name": "City Veterinary Hospital",
                        "address": "123 Main St, San Francisco, CA 94102",
                        "distance_km": 2.3,
                        "phone": "+1-415-555-0100",
                        "latitude": 37.7749,
                        "longitude": -122.4194
                    },
                    "nearest_rescuer": {
                        "name": "SF Animal Rescue",
                        "address": "456 Oak St, San Francisco, CA 94110",
                        "distance_km": 1.8,
                        "phone": "+1-415-555-0200",
                        "latitude": 37.7599,
                        "longitude": -122.4148
                    },
                    "rescue_plan": {
                        "immediate_actions": [
                            "Approach calmly and speak softly",
                            "Check for identification tags"
                        ],
                        "transport_instructions": "Use a blanket as stretcher, support injured leg",
                        "what_to_bring": ["Blanket", "Water", "Muzzle (if needed)"],
                        "precautions": ["Watch for signs of aggression", "Avoid touching injured area"],
                        "estimated_time": "30-45 minutes"
                    }
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy", description="Service health status")
    timestamp: Optional[str] = Field(None, description="Health check timestamp")
    version: Optional[str] = Field(None, description="API version")
