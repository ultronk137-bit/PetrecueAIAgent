"""
Helper utilities for common operations.
"""
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from io import BytesIO


def generate_report_id() -> str:
    """
    Generate a unique report ID.
    
    Returns:
        UUID string
    """
    return str(uuid4())


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO 8601 format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """
    Extract JSON from text that may contain markdown code blocks or other content.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object in text
    json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_object_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None


def validate_image_format(image_data: bytes) -> tuple[bool, Optional[str]]:
    """
    Validate image format and integrity.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # First pass: check format (verify() closes the fp, so we open twice)
        image = Image.open(BytesIO(image_data))
        fmt = image.format

        if fmt not in ('JPEG', 'PNG', 'WEBP'):
            return False, f"Unsupported image format: {fmt}. Supported: JPEG, PNG, WEBP"

        # Second pass: verify integrity (verify() can only be called once on a fresh open)
        Image.open(BytesIO(image_data)).verify()

        return True, None

    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove or replace special characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Limit length
    name_parts = filename.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        name = name[:200]  # Limit to 200 chars
        filename = f"{name}.{ext}"
    else:
        filename = filename[:200]
    
    return filename
