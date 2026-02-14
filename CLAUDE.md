# Admin_Bud-E Project Guide

## Architecture
- **Type:** Middleware Proxy & Admin Dashboard
- **Stack:** Python 3.11+, FastAPI, SQLAlchemy (Async), SQLite/Postgres.
- **Frontend:** Static HTML/JS in `static/` folder (Admin UI).
- **Billing:** Credit-based system using `CreditLedger` and `UsageLog`.

## Key Files
- `serve.py`: Main entry point. Configures env and starts uvicorn.
- `main.py`: FastAPI app definition, CORS, and Routes (including IMAGE and MUSIC).
- `vertex_openai_proxy.py`: Handles direct communication with Google Vertex AI (chat, TTS, images, music).
- `providers.py`: Logic for standard OpenAI-compatible requests (HTTPX calls) + image/music forwarding.
- `models.py`: Database schema (User, Project, RoutePref, UsageLog, ModelPricing).
- `billing.py`: Credit charging functions (charge_llm, charge_tts, charge_asr, charge_image, charge_music).
- `admin.py`: Admin API endpoints (CRUD for users, routes, pricing).
- `instructions.txt`: Comprehensive API documentation for image and music generation.

## Implemented Features

### Image Generation (`/v1/images/generations`)
- **Vertex AI Models:**
  - `gemini-3-pro-image-preview` (multimodal: text+images -> text+images)
  - `gemini-2.5-flash-image` (multimodal)
  - `imagen-4.0-generate-001` (text-to-image)
  - `imagen-4.0-fast-generate-001` (text-to-image, faster)
  - `imagen-4.0-ultra-generate-001` (text-to-image, highest quality)
- **Black Forest Labs FLUX.2 Models:**
  - `flux-2-klein-4b` (fastest, sub-second, ~$0.014/image)
  - `flux-2-klein-9b` (better prompts, ~$0.014/image)
  - `flux-2-pro` (production-grade, ~$0.03-0.05/image)
  - `flux-2-max` (maximum quality, ~$0.07/image)
- **OpenAI-Compatible Providers (HyperLab, etc.):**
  - `nano-banana`, `nano-banana-pro`
  - `flux-2-dev`, `seedream-4.5`
- **Features:**
  - Image editing with reference images (Gemini, FLUX.2)
  - FLUX.2: Up to 4 refs (klein) or 8 refs (pro/max)
  - Negative prompts
  - Seed for reproducibility
  - Provider failover

### Music Generation (`/v1/audio/generations`)
- **Vertex AI Lyria:**
  - `lyria-002` (instrumental music, 48kHz WAV, up to 32.8s)
- **Features:**
  - Negative prompts
  - Multiple clips (n=1-4)
  - Seed for reproducibility

### Billing
- `charge_image()`: Bills per generated image using `price_per_image`
- `charge_music()`: Bills per second of audio using `price_per_audio_second`

### Database
- `RouteKind`: LLM, VLM, TTS, ASR, IMAGE, MUSIC, OTHER
- `ModelType`: LLM, VLM, TTS, ASR, EMB, IMAGE, MUSIC
- `ModelPricing`: Added `price_per_image`, `price_per_audio_second` columns

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat/VLM with provider failover |
| `/v1/audio/speech` | POST | Text-to-Speech |
| `/v1/audio/transcriptions` | POST | Speech-to-Text |
| `/v1/images/generations` | POST | **Image generation** |
| `/v1/audio/generations` | POST | **Music generation** |

## Quick Setup

### 1. Vertex AI Configuration
```bash
VERTEX_REGION=europe-west4
VERTEX_PROJECT_ID=your-project-id
VERTEX_SA_JSON=/path/to/service-account.json
GEMINI_API_KEY=your-api-key  # Optional, for AI Studio
```

### 2. Add Providers (Admin UI)
- **vertex**: `http://127.0.0.1:8001` (local proxy)
- **bfl**: `https://api.bfl.ai` + BFL API key (from https://api.bfl.ai dashboard)
- **hyprlab**: `https://api.hyprlab.io/v1` + API key

### 3. Add Routes (Admin UI)
- Kind: IMAGE, Provider: vertex, Model: gemini-3-pro-image-preview
- Kind: IMAGE, Provider: bfl, Model: flux-2-pro
- Kind: IMAGE, Provider: bfl, Model: flux-2-max
- Kind: IMAGE, Provider: bfl, Model: flux-2-klein-9b
- Kind: IMAGE, Provider: hyprlab, Model: flux-2-dev
- Kind: MUSIC, Provider: vertex, Model: lyria-002

### 4. Configure Pricing (Admin UI)
- Model: gemini-3-pro-image-preview, Type: IMAGE, Price/Image: 0.02
- Model: flux-2-pro, Type: IMAGE, Price/Image: 0.04
- Model: flux-2-max, Type: IMAGE, Price/Image: 0.07
- Model: flux-2-klein-9b, Type: IMAGE, Price/Image: 0.014
- Model: lyria-002, Type: MUSIC, Price/Audio Sec: 0.01

## Commands
- Run Server: `python serve.py`
- DB Migration: Handled automatically on startup in `db.py`

## Documentation
See `instructions.txt` for comprehensive API documentation including:
- Request/response formats
- Example queries
- Error handling
- Provider configuration
- Troubleshooting guide

## API Reference Links
- [Gemini 3 Pro Image](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro-image)
- [Gemini 2.5 Flash Image](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-image)
- [Imagen 4.0](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/imagen/4-0-generate)
- [Lyria 002](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/lyria/lyria-002)
- [Black Forest Labs FLUX.2 API](https://api.bfl.ai) - Sign up for API key at dashboard
- [HyperLab API](https://docs.hyprlab.io/browse-models/model-list/google/image)
