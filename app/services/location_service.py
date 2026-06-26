"""
Location service abstraction for finding nearby vets and rescue organizations.
Supports Google Maps API with seamless fallback to mock data.
The RescueAgent depends only on the abstract interface — swapping
providers requires no changes outside this file.
"""
import math
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.config import get_settings
from app.models.response_models import Location
from app.utils.exceptions import LocationServiceException
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------

class LocationProvider(ABC):
    """Abstract location provider interface."""

    @abstractmethod
    async def find_nearest_vet(
        self,
        latitude: float,
        longitude: float
    ) -> Location:
        """Find the nearest veterinary hospital."""

    @abstractmethod
    async def find_nearest_rescuer(
        self,
        latitude: float,
        longitude: float
    ) -> Location:
        """Find the nearest animal rescue organization."""


# ---------------------------------------------------------------------------
# Haversine distance helper
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two coordinates.

    Returns:
        Distance in kilometres
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Mock Provider (always available, no API key required)
# ---------------------------------------------------------------------------

class MockLocationProvider(LocationProvider):
    """
    Mock location provider using realistic generated data.
    Computes distance from user coordinates to simulated facilities
    placed nearby so distances are geographically plausible.
    """

    _VETS = [
        {
            "name": "City Emergency Veterinary Hospital",
            "address": "450 Sutter St, San Francisco, CA 94108",
            "phone": "+1-415-555-0101",
            "lat_offset": 0.012,
            "lon_offset": -0.008,
        },
        {
            "name": "Animal Care & Emergency Clinic",
            "address": "1312 Mission St, San Francisco, CA 94103",
            "phone": "+1-415-555-0102",
            "lat_offset": -0.018,
            "lon_offset": 0.014,
        },
        {
            "name": "Bay Area Animal Hospital",
            "address": "3004 16th St, San Francisco, CA 94103",
            "phone": "+1-415-555-0103",
            "lat_offset": 0.024,
            "lon_offset": -0.021,
        },
    ]

    _RESCUERS = [
        {
            "name": "SF Animal Rescue Organization",
            "address": "1200 Harrison St, San Francisco, CA 94103",
            "phone": "+1-415-555-0201",
            "lat_offset": -0.009,
            "lon_offset": 0.011,
        },
        {
            "name": "Bay Animal Rescue & Shelter",
            "address": "2500 16th St, San Francisco, CA 94103",
            "phone": "+1-415-555-0202",
            "lat_offset": 0.015,
            "lon_offset": -0.019,
        },
        {
            "name": "Humane Society Bay Area",
            "address": "600 5th St, San Francisco, CA 94107",
            "phone": "+1-415-555-0203",
            "lat_offset": -0.022,
            "lon_offset": 0.017,
        },
    ]

    async def find_nearest_vet(self, latitude: float, longitude: float) -> Location:
        """Return the mock vet closest to the provided coordinates."""
        return self._find_nearest(latitude, longitude, self._VETS)

    async def find_nearest_rescuer(self, latitude: float, longitude: float) -> Location:
        """Return the mock rescuer closest to the provided coordinates."""
        return self._find_nearest(latitude, longitude, self._RESCUERS)

    def _find_nearest(
        self,
        latitude: float,
        longitude: float,
        candidates: list[dict]
    ) -> Location:
        best = None
        best_dist = float("inf")

        for candidate in candidates:
            fac_lat = latitude + candidate["lat_offset"]
            fac_lon = longitude + candidate["lon_offset"]
            dist = _haversine_km(latitude, longitude, fac_lat, fac_lon)

            if dist < best_dist:
                best_dist = dist
                best = {**candidate, "lat": fac_lat, "lon": fac_lon, "dist": dist}

        if best is None:
            raise LocationServiceException("No mock locations available")

        logger.debug(
            "Mock location selected",
            extra={"name": best["name"], "distance_km": round(best["dist"], 2)}
        )

        return Location(
            name=best["name"],
            address=best["address"],
            distance_km=round(best["dist"], 2),
            phone=best["phone"],
            latitude=round(best["lat"], 6),
            longitude=round(best["lon"], 6),
        )


# ---------------------------------------------------------------------------
# Google Maps Provider
# ---------------------------------------------------------------------------

class GoogleMapsLocationProvider(LocationProvider):
    """
    Location provider backed by Google Maps Places API (Nearby Search).
    Falls back to MockLocationProvider on API errors.
    """

    _client: httpx.AsyncClient | None = None
    _PLACES_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._fallback = MockLocationProvider()

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None or cls._client.is_closed:
            cls._client = httpx.AsyncClient(timeout=10.0)
        return cls._client

    @classmethod
    async def close_client(cls) -> None:
        if cls._client is not None and not cls._client.is_closed:
            await cls._client.aclose()

    async def find_nearest_vet(self, latitude: float, longitude: float) -> Location:
        """Find nearest veterinary care using Google Maps."""
        return await self._search(
            latitude,
            longitude,
            keyword="veterinary hospital emergency animal",
            fallback_fn=self._fallback.find_nearest_vet,
        )

    async def find_nearest_rescuer(self, latitude: float, longitude: float) -> Location:
        """Find nearest animal rescue organization using Google Maps."""
        return await self._search(
            latitude,
            longitude,
            keyword="animal rescue organization shelter",
            fallback_fn=self._fallback.find_nearest_rescuer,
        )

    async def _search(
        self,
        latitude: float,
        longitude: float,
        keyword: str,
        fallback_fn,
    ) -> Location:
        """
        Call Google Maps Places API and parse the first result.
        Falls back to mock data on any error.
        """
        try:
            client = self.get_client()
            params = {
                "location": f"{latitude},{longitude}",
                "rankby": "distance",
                "keyword": keyword,
                "key": self._settings.google_maps_api_key,
            }
            response = await client.get(self._PLACES_URL, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK" or not data.get("results"):
                logger.warning(
                    "Google Maps returned no results, using mock fallback",
                    extra={"status": data.get("status"), "keyword": keyword}
                )
                return await fallback_fn(latitude, longitude)

            result = data["results"][0]
            place_lat = result["geometry"]["location"]["lat"]
            place_lon = result["geometry"]["location"]["lng"]
            distance = _haversine_km(latitude, longitude, place_lat, place_lon)

            return Location(
                name=result.get("name", "Unknown Facility"),
                address=result.get("vicinity", "Address not available"),
                distance_km=round(distance, 2),
                phone=None,  # Requires a separate Place Details call
                latitude=round(place_lat, 6),
                longitude=round(place_lon, 6),
            )

        except Exception as e:
            logger.warning(
                "Google Maps API failed, using mock fallback",
                extra={"error": str(e), "keyword": keyword}
            )
            return await fallback_fn(latitude, longitude)


# ---------------------------------------------------------------------------
# Location Service (facade that picks the right provider)
# ---------------------------------------------------------------------------

class LocationService:
    """
    Facade over LocationProvider implementations.
    Automatically selects Google Maps if an API key is configured,
    otherwise uses the mock provider.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if settings.use_google_maps:
            self._provider: LocationProvider = GoogleMapsLocationProvider()
            logger.info("LocationService using Google Maps provider")
        else:
            self._provider = MockLocationProvider()
            logger.info("LocationService using mock provider (no Maps API key)")

    async def find_nearest_vet(
        self,
        latitude: float,
        longitude: float
    ) -> Location:
        """
        Find the nearest veterinary hospital.

        Args:
            latitude: User's latitude
            longitude: User's longitude

        Returns:
            Location of nearest vet
        """
        logger.info(
            "Finding nearest vet",
            extra={"latitude": latitude, "longitude": longitude}
        )
        return await self._provider.find_nearest_vet(latitude, longitude)

    async def find_nearest_rescuer(
        self,
        latitude: float,
        longitude: float
    ) -> Location:
        """
        Find the nearest animal rescue organization.

        Args:
            latitude: User's latitude
            longitude: User's longitude

        Returns:
            Location of nearest rescuer
        """
        logger.info(
            "Finding nearest rescuer",
            extra={"latitude": latitude, "longitude": longitude}
        )
        return await self._provider.find_nearest_rescuer(latitude, longitude)
