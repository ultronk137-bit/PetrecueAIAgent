# PetRescue AI Agent — Backend

An autonomous AI agent that receives a photo of an injured animal and a GPS location, then coordinates multiple services to return a complete rescue report in a single API call.

---

## Architecture

```
POST /agent/report
        │
        ▼
   FastAPI (api/agent.py)          ← validates input only
        │
        ▼
   RescueAgent.execute()           ← orchestrates everything
        │
        ├─ StorageService          → uploads image to GCS
        │
        ├─ AIProvider              → primary/fallback router
        │    ├─ OpenAIService      → GPT-4o vision (primary)
        │    └─ GeminiService      → Gemini 2.0 Flash (fallback)
        │
        ├─ DecisionService         → calculates rescue priority (1–5)
        │
        ├─ LocationService         → finds nearest vet & rescuer
        │    ├─ GoogleMapsProvider → real Places API (if key set)
        │    └─ MockProvider       → realistic mock (default)
        │
        ├─ AIProvider              → generates rescue plan
        │
        └─ FirestoreService        → persists complete report
```

**AI Fallback strategy**: Every request tries OpenAI GPT-4o first. If OpenAI raises any exception (rate limit, timeout, API error), the same request is automatically retried with Gemini 2.0 Flash — transparently, with no changes to the agent.

---

## Folder Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app factory, middleware, handlers
│   ├── config.py                # Pydantic Settings (env vars)
│   ├── api/
│   │   ├── agent.py             # POST /agent/report
│   │   └── health.py            # GET /health
│   ├── agents/
│   │   └── rescue_agent.py      # Core orchestrator
│   ├── services/
│   │   ├── ai_service.py        # Abstract AI interface
│   │   ├── ai_provider.py       # OpenAI → Gemini fallback router
│   │   ├── openai_service.py    # GPT-4o implementation
│   │   ├── gemini_service.py    # Gemini 2.0 Flash implementation
│   │   ├── storage_service.py   # Google Cloud Storage
│   │   ├── firestore_service.py # Firestore persistence
│   │   ├── location_service.py  # Vet/rescuer finder (Maps + mock)
│   │   └── decision_service.py  # Priority scoring
│   ├── models/
│   │   ├── request_models.py    # Pydantic request validation
│   │   └── response_models.py   # Pydantic response schemas
│   ├── prompts/
│   │   └── ai_prompts.py        # Shared prompts for all AI providers
│   └── utils/
│       ├── logger.py            # Structured JSON logging
│       ├── exceptions.py        # Custom exception hierarchy
│       └── helpers.py           # UUID, timestamps, JSON parsing, validation
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GCP_PROJECT_ID` | ✅ | — | Google Cloud project ID |
| `GCP_REGION` | | `us-central1` | GCP region |
| `GCS_BUCKET_NAME` | ✅ | — | Cloud Storage bucket for images |
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key (primary AI) |
| `OPENAI_MODEL` | | `gpt-4o` | OpenAI model to use |
| `GEMINI_API_KEY` | ✅ | — | Gemini API key (fallback AI) |
| `GEMINI_MODEL` | | `gemini-2.0-flash-exp` | Gemini model to use |
| `AI_TIMEOUT_SECONDS` | | `30` | Timeout per AI call |
| `AI_MAX_RETRIES` | | `2` | Retries per provider before fallback |
| `FIRESTORE_COLLECTION` | | `rescue_reports` | Firestore collection name |
| `GOOGLE_MAPS_API_KEY` | | — | Maps API key (mock used if empty) |
| `MAX_IMAGE_SIZE_MB` | | `10` | Maximum upload size |
| `LOG_LEVEL` | | `INFO` | Logging verbosity |
| `ENVIRONMENT` | | `development` | `development` or `production` |

---

## Running Locally

### Prerequisites

- Python 3.12
- A Google Cloud project with Firestore and Cloud Storage enabled
- Google Application Default Credentials configured (`gcloud auth application-default login`)

### Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your actual values
```

### Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

---

## API Documentation

### GET /health

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-06-26T13:00:00Z",
  "version": "1.0.0"
}
```

---

### POST /agent/report

Accepts a multipart/form-data request and returns a complete rescue report.

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | Image of the injured animal (JPEG, PNG, WEBP, max 10 MB) |
| `latitude` | float | ✅ | Latitude of the rescue location (-90 to 90) |
| `longitude` | float | ✅ | Longitude of the rescue location (-180 to 180) |
| `description` | string | | Optional situation description (max 1000 chars) |

**Example curl:**
```bash
curl -X POST http://localhost:8000/agent/report \
  -F "image=@/path/to/animal.jpg" \
  -F "latitude=37.7749" \
  -F "longitude=-122.4194" \
  -F "description=Injured dog near the park"
```

**Response (200):**
```json
{
  "status": "success",
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-06-26T13:45:00Z",
  "image_url": "https://storage.googleapis.com/petrescue-images/rescue-images/2026/06/26/...",
  "analysis": {
    "species": "Domestic Dog - Labrador Retriever",
    "injuries": ["Laceration on right front leg", "Limping gait"],
    "severity": "moderate",
    "confidence": 0.87,
    "first_aid": [
      "Step 1: Approach calmly to avoid startling the animal",
      "Step 2: Apply clean cloth with gentle pressure to the wound"
    ],
    "additional_notes": "Animal is alert and responsive"
  },
  "priority": 3,
  "nearest_vet": {
    "name": "City Emergency Veterinary Hospital",
    "address": "450 Sutter St, San Francisco, CA 94108",
    "distance_km": 1.43,
    "phone": "+1-415-555-0101",
    "latitude": 37.787,
    "longitude": -122.427
  },
  "nearest_rescuer": {
    "name": "SF Animal Rescue Organization",
    "address": "1200 Harrison St, San Francisco, CA 94103",
    "distance_km": 1.08,
    "phone": "+1-415-555-0201",
    "latitude": 37.766,
    "longitude": -122.408
  },
  "rescue_plan": {
    "immediate_actions": [
      "Approach slowly from the front so the dog can see you",
      "Muzzle gently if the dog shows signs of pain-induced aggression",
      "Apply pressure bandage to the leg laceration"
    ],
    "transport_instructions": "Slide a blanket under the dog and use it as a stretcher. Keep the injured leg immobilised during transport.",
    "what_to_bring": ["Clean cloth or bandage", "Blanket or towel", "Muzzle", "Water bottle"],
    "precautions": [
      "Do not let the dog bear weight on the injured leg",
      "Keep the animal warm to prevent shock"
    ],
    "estimated_time": "25-35 minutes"
  }
}
```

**Error responses:**

| Status | Cause |
|---|---|
| `400` | Invalid image format, oversized file, or invalid coordinates |
| `500` | AI analysis failed (both OpenAI and Gemini exhausted), storage error, or Firestore error |

---

## Deploying to Cloud Run

### 1. Build and push the Docker image

```bash
# Set your project and region
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export IMAGE=gcr.io/$PROJECT_ID/petrescue-backend:latest

# Configure Docker for GCR
gcloud auth configure-docker

# Build
docker build -t $IMAGE ./backend

# Push
docker push $IMAGE
```

### 2. Create the GCS bucket

```bash
gsutil mb -p $PROJECT_ID -l $REGION gs://petrescue-images
gsutil iam ch allUsers:objectViewer gs://petrescue-images
```

### 3. Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com \
  aiplatform.googleapis.com \
  --project $PROJECT_ID
```

### 4. Create Firestore database (if not exists)

```bash
gcloud firestore databases create --region=$REGION --project $PROJECT_ID
```

### 5. Deploy to Cloud Run

```bash
gcloud run deploy petrescue-backend \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --concurrency 80 \
  --set-env-vars "ENVIRONMENT=production" \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
  --set-env-vars "GCP_REGION=$REGION" \
  --set-env-vars "GCS_BUCKET_NAME=petrescue-images" \
  --set-env-vars "FIRESTORE_COLLECTION=rescue_reports" \
  --set-env-vars "OPENAI_MODEL=gpt-4o" \
  --set-env-vars "GEMINI_MODEL=gemini-2.0-flash-exp" \
  --set-env-vars "AI_TIMEOUT_SECONDS=30" \
  --set-env-vars "AI_MAX_RETRIES=2" \
  --set-env-vars "MAX_IMAGE_SIZE_MB=10" \
  --set-env-vars "LOG_LEVEL=INFO" \
  --set-secrets "OPENAI_API_KEY=openai-api-key:latest" \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --project $PROJECT_ID
```

> API keys are stored as Secret Manager secrets for security. Create them first:
> ```bash
> echo -n "sk-..." | gcloud secrets create openai-api-key --data-file=- --project $PROJECT_ID
> echo -n "AI..." | gcloud secrets create gemini-api-key --data-file=- --project $PROJECT_ID
> ```

### 6. Grant Cloud Run service account permissions

```bash
# Get the service account
SA=$(gcloud run services describe petrescue-backend \
  --region $REGION \
  --format "value(spec.template.spec.serviceAccountName)" \
  --project $PROJECT_ID)

# Grant Storage access
gsutil iam ch serviceAccount:$SA:objectAdmin gs://petrescue-images

# Grant Firestore access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/datastore.user"

# Grant Secret Manager access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/secretmanager.secretAccessor"
```

### 7. Verify deployment

```bash
SERVICE_URL=$(gcloud run services describe petrescue-backend \
  --region $REGION \
  --format "value(status.url)" \
  --project $PROJECT_ID)

curl $SERVICE_URL/health
```
