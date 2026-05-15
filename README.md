# Admin Bud-E — AI Middleware & Admin Dashboard

A unified middleware proxy that sits between AI clients (like School Bud-E) and multiple AI providers. Manages users, billing, routing, and provides a web admin dashboard.

## Features

### Chat & Language Models
- **Google Vertex AI**: Gemini 2.5 Flash, Gemini 2.5 Pro, and other Gemini models
- **OpenAI-compatible providers**: Any provider with OpenAI-style API
- **Streaming & non-streaming** support
- **Auto-routing** with priority-based failover between providers
- **VLM support**: Vision-language models with image/PDF input

### Image Generation
- **Google Vertex AI**:
  - `gemini-3-pro-image-preview` — multimodal text+image generation
  - `imagen-4.0-generate-001` — high-quality text-to-image
  - `imagen-4.0-fast-generate-001` — faster generation
  - `imagen-4.0-ultra-generate-001` — maximum quality
- **Black Forest Labs FLUX.2**:
  - `flux-2-klein-4b` — fastest, sub-second (~$0.014/image)
  - `flux-2-klein-9b` — better prompt understanding (~$0.014/image)
  - `flux-2-pro` — production-grade (~$0.03-0.05/image)
  - `flux-2-max` — maximum quality (~$0.07/image)
  - Supports reference images for style transfer (up to 4 refs for klein, 8 for pro/max)
- **HyperLab / OpenAI-compatible**:
  - `nano-banana`, `nano-banana-pro`, `nano-banana-2`
  - `flux-2-dev`, `seedream-4.5`
- **Features**: Image editing with reference images, negative prompts, seed for reproducibility, aspect ratios

### Music Generation
- **Google Vertex AI Lyria 3 Pro** (`lyria-3-pro-preview`):
  - Full songs with vocals and lyrics (up to 184 seconds / ~3 minutes)
  - MP3 output at 44.1kHz / 192kbps
  - Supports lyrics with `[Verse]`, `[Chorus]`, `[Bridge]` tags
  - Caption/style guidance
  - Multi-language: English, German, Spanish, French, Hindi, Japanese, Korean, Portuguese
- **Google Vertex AI Lyria 3 Clip** (`lyria-3-clip-preview`):
  - 30-second instrumental clips
- **Google Vertex AI Lyria 2** (`lyria-002`):
  - Instrumental music, 32.8s WAV at 48kHz (legacy)

### Speech
- **TTS** (Text-to-Speech): Google Vertex AI, OpenAI-compatible
- **ASR** (Speech-to-Text): Whisper via OpenAI-compatible providers

### Admin Dashboard
- **Users**: Create, manage API keys, set credit budgets
- **Projects**: Group users, set allowances, Common Pool for shared credits
- **Providers**: Configure upstream AI services (Vertex, BFL, HyperLab, etc.)
- **Routes**: Priority-based routing with automatic failover
- **Pricing**: Per-model pricing (tokens, images, audio seconds)
- **Usage & Billing**: Detailed usage logs, credit ledger
- **Maintenance**: Database backup/restore, batch user creation

---

## Quick Start

### Prerequisites
- Python 3.11+
- A Google Cloud project with Vertex AI enabled (for Gemini/Imagen/Lyria)
- Optional: BFL API key (for FLUX.2), HyperLab API key

### 1. Install

```bash
git clone https://github.com/LAION-AI/Admin_Bud-E.git
cd Admin_Bud-E
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# Google Vertex AI
VERTEX_REGION=europe-west4
VERTEX_PROJECT_ID=your-project-id
VERTEX_SA_JSON=/path/to/service-account.json

# Optional: Gemini API key (for AI Studio)
GEMINI_API_KEY=your-api-key

# Server
HOST=0.0.0.0
PORT=8787
ADMIN_PASSWORD=your-admin-password
```

### 3. Run

```bash
python serve.py
```

The server starts on `http://localhost:8787`. The admin dashboard is at `http://localhost:8787/admin`.

The Vertex AI proxy runs internally on port 8001 (auto-started by `serve.py`).

### 4. Configure Providers (Admin UI)

Open the admin dashboard and add providers:

| Provider | Base URL | Notes |
|----------|----------|-------|
| `vertex` | `http://127.0.0.1:8001` | Local Vertex proxy (auto-started) |
| `bfl` | `https://api.bfl.ai` | Black Forest Labs FLUX.2 |
| `hyprlab` | `https://api.hyprlab.io/v1` | HyperLab (nano-banana, etc.) |

### 5. Configure Routes (Admin UI)

Add routes for each service type:

| Kind | Provider | Model | Priority |
|------|----------|-------|----------|
| LLM | vertex | gemini-2.5-flash | 1 |
| IMAGE | vertex | gemini-3-pro-image-preview | 1 |
| IMAGE | bfl | flux-2-pro | 2 |
| MUSIC | vertex | lyria-3-pro-preview | 1 |
| TTS | vertex | (default) | 1 |
| ASR | vertex | (default) | 1 |

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat/VLM (streaming & non-streaming) |
| `/v1/images/generations` | POST | Image generation |
| `/v1/audio/generations` | POST | Music generation |
| `/v1/audio/speech` | POST | Text-to-Speech |
| `/v1/audio/transcriptions` | POST | Speech-to-Text |

### Image Generation

```bash
curl -X POST http://localhost:8787/v1/images/generations \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "model": "gemini-3-pro-image-preview",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json"
  }'
```

With reference image (editing):
```json
{
  "prompt": "Make the sky more dramatic",
  "model": "gemini-3-pro-image-preview",
  "input_images": ["data:image/jpeg;base64,..."],
  "response_format": "b64_json"
}
```

### Music Generation (Lyria 3 Pro)

```bash
curl -X POST http://localhost:8787/v1/audio/generations \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lyria-3-pro-preview",
    "prompt": "Upbeat pop song about summer with catchy chorus",
    "lyrics": "[Verse]\nSunshine on my face...\n[Chorus]\nSummer days are here..."
  }'
```

Response includes generated audio (MP3), lyrics, and description:
```json
{
  "data": [{"b64_json": "...", "mime_type": "audio/mpeg", "duration_seconds": 120.5}],
  "lyrics": "Generated or echoed lyrics...",
  "description": "Song structure description..."
}
```

---

## Architecture

```
Client (BUD-E App)
    │
    ▼
Admin Bud-E Middleware (FastAPI, port 8787)
    │  ├── User auth & billing
    │  ├── Route selection & failover
    │  └── Request transformation
    │
    ├──► Vertex AI Proxy (port 8001) ──► Google Cloud
    ├──► Black Forest Labs API ──► BFL Cloud
    └──► HyperLab / Other OpenAI-compatible APIs
```

### Key Files
- `serve.py` — Entry point, starts both middleware and Vertex proxy
- `main.py` — FastAPI app with all routes (chat, image, music, TTS, ASR)
- `vertex_openai_proxy.py` — Translates OpenAI-format to Vertex AI native APIs
- `providers.py` — Provider communication, image/music forwarding
- `models.py` — Database schema (SQLAlchemy)
- `billing.py` — Credit charging functions
- `admin.py` — Admin API endpoints
- `security.py` — Authentication
- `db.py` — Database setup and migrations

---

## Deployment

### Docker

```bash
docker-compose up -d
```

### Systemd Service

```bash
# Copy to server
scp -r . user@server:/opt/admin-bud-e/

# Create service file
sudo cat > /etc/systemd/system/admin-bud-e.service << EOF
[Unit]
Description=Admin Bud-E Middleware
After=network.target

[Service]
WorkingDirectory=/opt/admin-bud-e
ExecStart=/usr/bin/python3 serve.py
Restart=always
User=www-data
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable admin-bud-e
sudo systemctl start admin-bud-e
```

### Running Forever (Simple)

```bash
nohup python serve.py > buddy.out 2>&1 &
# or use the included script:
bash run_forever.sh
```

---

## License

Apache 2.0 — Made by [LAION](https://laion.ai) and collaborators.
