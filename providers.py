# providers.py
import os, json, httpx
from typing import AsyncGenerator, Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import ProviderEndpoint

TIMEOUT = int(os.getenv("TIMEOUT_SECONDS", "60"))
IMAGE_TIMEOUT = int(os.getenv("IMAGE_TIMEOUT_SECONDS", "180"))  # Longer timeout for image generation
MUSIC_TIMEOUT = int(os.getenv("MUSIC_TIMEOUT_SECONDS", "300"))  # Even longer for music generation

async def _get_provider(session: AsyncSession, name: str) -> dict:
    q = await session.execute(select(ProviderEndpoint).where(ProviderEndpoint.name==name))
    pe = q.scalar_one_or_none()
    if pe:
        return {"base_url": pe.base_url, "api_key": pe.api_key}
    # Fallback: .env
    if name == "openai_compat":
        return {"base_url": os.getenv("OPENAI_BASE_URL"), "api_key": os.getenv("OPENAI_API_KEY")}
    if name == "gemini":
        # Gemini nutzt kein base_url
        return {"base_url": None, "api_key": os.getenv("GEMINI_API_KEY")}
    if name == "tts_provider":
        return {"base_url": os.getenv("TTS_PROVIDER_BASE_URL"), "api_key": os.getenv("TTS_PROVIDER_API_KEY")}
    if name == "asr_provider":
        return {"base_url": os.getenv("ASR_PROVIDER_BASE_URL"), "api_key": os.getenv("ASR_PROVIDER_API_KEY")}
    return {"base_url": None, "api_key": None}

async def openai_chat_stream(session: AsyncSession, payload: dict) -> AsyncGenerator[str, None]:
    prov = await _get_provider(session, "openai_compat")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(f"{prov['base_url']}/chat/completions",
                              headers={"Authorization": f"Bearer {prov['api_key']}",
                                       "Content-Type": "application/json"},
                              json=payload)
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line and line.startswith("data: "):
                yield line[6:]

async def gemini_stream(session: AsyncSession, contents: list, system_instruction: dict | None, generation_config: dict | None) -> AsyncGenerator[str, None]:
    prov = await _get_provider(session, "gemini")
    model = "gemini-2.5-pro"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={prov['api_key']}"
    body = {"contents": contents}
    if system_instruction: body["systemInstruction"] = system_instruction
    if generation_config: body["generationConfig"] = generation_config

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(url, headers={"Content-Type":"application/json"}, json=body)
        r.raise_for_status()
        async for line in r.aiter_lines():
            if line and line.startswith("data: "):
                yield line[6:]

async def tts_forward(session: AsyncSession, model: str, text: str, provider: str) -> bytes:
    # Get the provider's base_url and api_key from the database / .env
    prov_details = await _get_provider(session, provider)
    base_url = prov_details.get("base_url")
    api_key = prov_details.get("api_key")

    if not base_url or not api_key:
        raise ValueError(f"Provider '{provider}' is missing base_url or api_key in configuration.")

    # --- helpers -------------------------------------------------------------
    def _parse_model_options(raw: str) -> dict:
        """
        Allow passing options via the Routes 'model' column, e.g.:
          - "auto"
          - "en-US-Chirp3-HD-Achernar"
          - "voice=en-US-Wavenet-D"
          - "voice=de-DE-Chirp3-HD-Achernar;lang=de-DE;format=mp3"
        Returns a dict possibly containing: voice_name, language_code, audio_format
        """
        out = {}
        if not raw:
            return out
        s = raw.strip()
        if s.lower() == "auto":
            return out
        # key=value;key=value ... OR just a bare voice name
        if ("=" not in s) and (";" not in s):
            out["voice_name"] = s
            return out
        for part in s.split(";"):
            if not part.strip():
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k in ("voice", "voice_name"):
                    out["voice_name"] = v
                elif k in ("lang", "language", "language_code"):
                    out["language_code"] = v
                elif k in ("format", "audio_format"):
                    out["audio_format"] = v
        return out

    clean_base = base_url.rstrip("/")

    # ---------------- FISH AUDIO (unchanged) --------------------------------
    if "fish.audio" in base_url or "fish" in provider.lower():
        payload = {
            "text": text,
            "reference_id": model,   # you already use model to carry reference_id
            "normalize": True,
            "format": "mp3",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        final_url = clean_base  # Fish uses the base URL directly

    # ---------------- VERTEX PROXY (new branch) -----------------------------
    # Detect by provider name OR by your proxy base_url (e.g. http://localhost:8001/v1)
    elif "vertex" in provider.lower() or "localhost:8001" in clean_base:
        # --- Vertex proxy expects: {"text", optional "voice_name", optional "language_code", "audio_format"} ---
        opts = _parse_model_options(model or "")

        voice_name = opts.get("voice_name")
        language_code = opts.get("language_code")  # user may omit this � we�ll infer from voice_name

        # If user passed a full Google voice name (e.g., "de-DE-Chirp3-HD-Achernar"
        # or "en-US-Wavenet-D") but NO language_code, auto-derive it from the prefix.
        if voice_name and not language_code:
            import re
            m = re.match(r"^([a-z]{2,3}-[A-Z]{2})-", voice_name)
            if m:
                language_code = m.group(1)

        payload = {
            "text": text,
            **({"voice_name": voice_name} if voice_name else {}),
            **({"language_code": language_code} if language_code else {}),  # leave out ? proxy will detect from text
            "audio_format": opts.get("audio_format", "mp3"),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        final_url = f"{clean_base}/audio/speech"


    # ---------------- DEFAULT (OpenAI-compatible) ---------------------------
    else:
        payload = {
            "model": model,
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        final_url = f"{clean_base}/audio/speech"

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(final_url, headers=headers, json=payload)
        r.raise_for_status()
        return r.content

async def asr_forward(session: AsyncSession, model:str, file_bytes: bytes, filename: str, provider: str) -> dict:
    prov = await _get_provider(session, provider)
    base_url = prov.get("base_url", "")
    api_key = prov.get("api_key")

    # --- START OF NEW ROBUST LOGIC ---
    # This logic now intelligently constructs the final URL.

    # Clean up the base URL by removing trailing slashes.
    clean_base_url = base_url.rstrip('/')

    # If the user accidentally included '/chat/completions' in the base URL,
    # we intelligently replace it with the correct path.
    if '/chat/completions' in clean_base_url:
        final_url = clean_base_url.replace('/chat/completions', '/audio/transcriptions')
    else:
        # Otherwise, we just append the correct path as normal.
        final_url = f"{clean_base_url}/audio/transcriptions"
    # --- END OF NEW ROBUST LOGIC ---

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        files = {"file": (filename, file_bytes, "audio/wav")}
        data = {"model": model}

        # Use the newly constructed final_url
        r = await client.post(
            final_url,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data
        )
        r.raise_for_status()
        return r.json()


# ==============================================================================
# Image Generation Forwarding
# ==============================================================================

# Aspect ratio translation helpers
_SIZE_TO_ASPECT = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1280x896": "16:9",
    "896x1280": "9:16",
    "2048x2048": "1:1",
    "512x512": "1:1",
    "256x256": "1:1",
}


def _size_to_aspect_ratio(size: str) -> str:
    """Convert OpenAI-style size (e.g., '1024x1024') to aspect ratio (e.g., '1:1')."""
    if not size:
        return "1:1"
    size = size.strip().lower()
    if size in _SIZE_TO_ASPECT:
        return _SIZE_TO_ASPECT[size]
    # Try to parse WxH and compute ratio
    try:
        if "x" in size:
            w, h = map(int, size.split("x"))
            from math import gcd
            g = gcd(w, h)
            return f"{w // g}:{h // g}"
    except Exception:
        pass
    return "1:1"


async def image_forward(
    session: AsyncSession,
    model: str,
    prompt: str,
    provider: str,
    n: int = 1,
    size: str = "1024x1024",
    input_images: Optional[List[Any]] = None,
    response_format: str = "b64_json",
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    enhance_prompt: bool = True,
) -> Dict[str, Any]:
    """
    Forward image generation request to an OpenAI-compatible provider.

    Supports providers like:
    - HyperLab (https://api.hyprlab.io/v1)
    - Any OpenAI-compatible image API

    Args:
        session: Database session
        model: Model name (e.g., 'flux-2-dev', 'seedream-4.5', 'nano-banana')
        prompt: Text prompt for image generation
        provider: Provider name from database
        n: Number of images to generate (1-4)
        size: Image size (e.g., '1024x1024')
        input_images: Optional reference images for editing (list of base64 or data URLs)
        response_format: 'b64_json' or 'url'
        negative_prompt: What to avoid in the image
        seed: For reproducibility
        enhance_prompt: Whether to use prompt enhancement

    Returns:
        OpenAI-compatible response dict with 'data' array containing generated images

    Raises:
        ValueError: If provider configuration is missing
        httpx.HTTPStatusError: If the API request fails
    """
    prov = await _get_provider(session, provider)
    base_url = prov.get("base_url", "")
    api_key = prov.get("api_key")

    if not base_url:
        raise ValueError(
            f"Provider '{provider}' has no base_url configured. "
            f"Add it via Admin → Providers or set in environment."
        )
    if not api_key:
        raise ValueError(
            f"Provider '{provider}' has no api_key configured. "
            f"Add it via Admin → Providers or set in environment."
        )

    # Construct the endpoint URL
    clean_base = base_url.rstrip("/")

    # Handle various base URL formats
    if "/images/generations" in clean_base:
        final_url = clean_base
    elif clean_base.endswith("/v1"):
        final_url = f"{clean_base}/images/generations"
    elif "/v1" in clean_base:
        # e.g., https://api.hyprlab.io/v1/chat/completions -> https://api.hyprlab.io/v1/images/generations
        idx = clean_base.rfind("/v1")
        final_url = clean_base[:idx + 3] + "/images/generations"
    else:
        final_url = f"{clean_base}/v1/images/generations"

    # Build request body
    # OpenAI-compatible format with extensions for various providers
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": min(max(1, n), 4),
        "response_format": response_format,
    }

    # Size handling - some providers use 'size', others use 'aspect_ratio'
    aspect_ratio = _size_to_aspect_ratio(size)

    # HyperLab-style providers prefer aspect_ratio
    if "hyprlab" in base_url.lower() or "hypr" in provider.lower():
        body["aspect_ratio"] = aspect_ratio
    else:
        # Standard OpenAI uses size
        body["size"] = size

    # Optional parameters
    if input_images:
        # HyperLab uses 'input_images', some providers use 'image'
        body["input_images"] = input_images

    if negative_prompt:
        body["negative_prompt"] = negative_prompt

    if seed is not None:
        body["seed"] = int(seed)

    # Provider-specific adjustments
    if "hyprlab" in base_url.lower():
        # HyperLab-specific options
        body["output_format"] = "png"
        if enhance_prompt is not None:
            body["enhance_prompt"] = enhance_prompt

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT) as client:
            r = await client.post(final_url, headers=headers, json=body)

            if r.status_code >= 400:
                error_text = r.text
                try:
                    error_json = r.json()
                    error_msg = error_json.get("error", {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", error_text[:500])
                    else:
                        error_msg = str(error_msg)[:500]
                except Exception:
                    error_msg = error_text[:500]

                raise ImageGenerationError(
                    f"Provider '{provider}' returned {r.status_code}: {error_msg}",
                    status_code=r.status_code,
                    provider=provider,
                    model=model,
                    error_detail=error_msg
                )

            result = r.json()

            # Normalize response format
            if "data" not in result:
                # Some providers return differently - try to normalize
                if "images" in result:
                    result["data"] = result["images"]
                elif "output" in result:
                    result["data"] = result["output"]

            return result

    except httpx.TimeoutException as e:
        raise ImageGenerationError(
            f"Timeout after {IMAGE_TIMEOUT}s waiting for provider '{provider}'",
            status_code=504,
            provider=provider,
            model=model,
            error_detail=str(e)
        )
    except httpx.RequestError as e:
        raise ImageGenerationError(
            f"Network error connecting to provider '{provider}': {str(e)}",
            status_code=503,
            provider=provider,
            model=model,
            error_detail=str(e)
        )


async def image_forward_vertex(
    session: AsyncSession,
    model: str,
    prompt: str,
    provider: str,
    n: int = 1,
    size: str = "1024x1024",
    input_images: Optional[List[Any]] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    enhance_prompt: bool = True,
) -> Dict[str, Any]:
    """
    Forward image generation request to Vertex proxy (localhost:8001).

    This is used when provider is 'vertex' or 'gemini' and routes to the
    local vertex_openai_proxy.py service.

    Args:
        session: Database session
        model: Model name (e.g., 'gemini-3-pro-image-preview', 'imagen-4.0-generate-001')
        prompt: Text prompt
        provider: Provider name
        n: Number of images
        size: Image size
        input_images: Reference images for editing
        negative_prompt: What to avoid
        seed: For reproducibility
        enhance_prompt: Use prompt enhancement

    Returns:
        OpenAI-compatible response dict
    """
    prov = await _get_provider(session, provider)
    base_url = prov.get("base_url", "")

    # Default to local vertex proxy
    if not base_url:
        base_url = os.getenv("VERTEX_PROXY_URL", "http://127.0.0.1:8001")

    clean_base = base_url.rstrip("/")

    # Ensure we hit the images endpoint
    if "/images/generations" in clean_base:
        final_url = clean_base
    elif clean_base.endswith("/v1"):
        final_url = f"{clean_base}/images/generations"
    else:
        final_url = f"{clean_base}/v1/images/generations"

    # Build request body
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": min(max(1, n), 4),
        "size": size,
        "response_format": "b64_json",
    }

    if input_images:
        body["input_images"] = input_images

    if negative_prompt:
        body["negative_prompt"] = negative_prompt

    if seed is not None:
        body["seed"] = int(seed)

    if enhance_prompt is not None:
        body["enhance_prompt"] = enhance_prompt

    headers = {"Content-Type": "application/json"}

    # Add API key if configured (proxy may not need it)
    api_key = prov.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT) as client:
            r = await client.post(final_url, headers=headers, json=body)

            if r.status_code >= 400:
                error_text = r.text
                try:
                    error_json = r.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
                        if isinstance(error_detail, dict):
                            error_msg = error_detail.get("message", str(error_detail))
                        else:
                            error_msg = str(error_detail)
                    else:
                        error_msg = error_text[:500]
                except Exception:
                    error_msg = error_text[:500]

                raise ImageGenerationError(
                    f"Vertex proxy returned {r.status_code}: {error_msg}",
                    status_code=r.status_code,
                    provider=provider,
                    model=model,
                    error_detail=error_msg
                )

            return r.json()

    except httpx.TimeoutException:
        raise ImageGenerationError(
            f"Timeout after {IMAGE_TIMEOUT}s waiting for Vertex proxy",
            status_code=504,
            provider=provider,
            model=model
        )
    except httpx.RequestError as e:
        raise ImageGenerationError(
            f"Error connecting to Vertex proxy: {str(e)}",
            status_code=503,
            provider=provider,
            model=model,
            error_detail=str(e)
        )


# ==============================================================================
# Music Generation Forwarding
# ==============================================================================

async def music_forward(
    session: AsyncSession,
    model: str,
    prompt: str,
    provider: str,
    negative_prompt: Optional[str] = None,
    n: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Forward music generation request to Vertex proxy (Lyria).

    Currently only Vertex AI Lyria is supported for music generation.
    Routes to the local vertex_openai_proxy.py service.

    Args:
        session: Database session
        model: Model name (e.g., 'lyria-002')
        prompt: Text prompt describing desired music
        provider: Provider name
        negative_prompt: What to avoid (e.g., 'vocals')
        n: Number of clips (1-4)
        seed: For reproducibility

    Returns:
        Response dict with 'data' array containing generated audio clips
    """
    prov = await _get_provider(session, provider)
    base_url = prov.get("base_url", "")

    # Default to local vertex proxy
    if not base_url:
        base_url = os.getenv("VERTEX_PROXY_URL", "http://127.0.0.1:8001")

    clean_base = base_url.rstrip("/")

    # Ensure we hit the audio generations endpoint
    if "/audio/generations" in clean_base:
        final_url = clean_base
    elif clean_base.endswith("/v1"):
        final_url = f"{clean_base}/audio/generations"
    else:
        final_url = f"{clean_base}/v1/audio/generations"

    # Build request body
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json",
    }

    if negative_prompt:
        body["negative_prompt"] = negative_prompt

    # seed and n are mutually exclusive for Lyria
    if seed is not None:
        body["seed"] = int(seed)
    else:
        body["n"] = min(max(1, n), 4)

    headers = {"Content-Type": "application/json"}

    # Add API key if configured
    api_key = prov.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=MUSIC_TIMEOUT) as client:
            r = await client.post(final_url, headers=headers, json=body)

            if r.status_code >= 400:
                error_text = r.text
                try:
                    error_json = r.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
                        if isinstance(error_detail, dict):
                            error_msg = error_detail.get("message", str(error_detail))
                        else:
                            error_msg = str(error_detail)
                    else:
                        error_msg = error_text[:500]
                except Exception:
                    error_msg = error_text[:500]

                raise MusicGenerationError(
                    f"Music provider returned {r.status_code}: {error_msg}",
                    status_code=r.status_code,
                    provider=provider,
                    model=model,
                    error_detail=error_msg
                )

            return r.json()

    except httpx.TimeoutException:
        raise MusicGenerationError(
            f"Timeout after {MUSIC_TIMEOUT}s waiting for music generation",
            status_code=504,
            provider=provider,
            model=model
        )
    except httpx.RequestError as e:
        raise MusicGenerationError(
            f"Error connecting to music provider: {str(e)}",
            status_code=503,
            provider=provider,
            model=model,
            error_detail=str(e)
        )


# ==============================================================================
# Custom Exceptions for Better Error Handling
# ==============================================================================

class ImageGenerationError(Exception):
    """Exception raised when image generation fails."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_detail: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.model = model
        self.error_detail = error_detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "image_generation_failed",
            "message": self.message,
            "status_code": self.status_code,
            "provider": self.provider,
            "model": self.model,
            "detail": self.error_detail
        }


class MusicGenerationError(Exception):
    """Exception raised when music generation fails."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_detail: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.model = model
        self.error_detail = error_detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": "music_generation_failed",
            "message": self.message,
            "status_code": self.status_code,
            "provider": self.provider,
            "model": self.model,
            "detail": self.error_detail
        }


# ==============================================================================
# Black Forest Labs (BFL) FLUX.2 Image Generation
# ==============================================================================

# BFL Model endpoints mapping
BFL_MODEL_ENDPOINTS = {
    "flux-2-klein-4b": "/v1/flux-2-klein-4b",
    "flux-2-klein-9b": "/v1/flux-2-klein-9b",
    "flux-2-max": "/v1/flux-2-max",
    "flux-2-pro": "/v1/flux-2-pro",
    # Aliases for convenience
    "flux-2-klein": "/v1/flux-2-klein-9b",  # Default to 9b for better quality
    "flux-2": "/v1/flux-2-pro",  # Default to pro
}

BFL_BASE_URL = "https://api.bfl.ai"


def _parse_bfl_input_image(img: Any, index: int = 0) -> Optional[str]:
    """
    Parse an input image for BFL API.
    BFL accepts URLs or base64 strings directly.

    Args:
        img: Image data (data URL, base64 string, or dict with data/mime_type)
        index: Image index for logging

    Returns:
        Base64 string or URL suitable for BFL API, or None if invalid
    """
    if not img:
        return None

    # Data URL format: data:image/png;base64,...
    if isinstance(img, str):
        if img.startswith("data:"):
            try:
                # Extract base64 part from data URL
                _, b64 = img.split(",", 1)
                return b64
            except Exception:
                return None
        elif img.startswith("http://") or img.startswith("https://"):
            # URL - BFL accepts these directly
            return img
        elif len(img) > 100:
            # Assume it's already base64
            return img

    # Dict format: {data: ..., mime_type: ...}
    if isinstance(img, dict):
        b64 = img.get("data") or img.get("b64_json") or img.get("base64")
        if b64:
            # Strip data URL prefix if present
            if isinstance(b64, str) and b64.startswith("data:"):
                try:
                    _, b64 = b64.split(",", 1)
                except Exception:
                    pass
            return b64

    return None


async def image_forward_bfl(
    session: AsyncSession,
    model: str,
    prompt: str,
    provider: str,
    n: int = 1,
    size: str = "1024x1024",
    input_images: Optional[List[Any]] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    safety_tolerance: int = 2,
    output_format: str = "png",
) -> Dict[str, Any]:
    """
    Forward image generation request to Black Forest Labs FLUX.2 API.

    Supports:
    - FLUX.2 [klein] (4B, 9B): Fast, sub-second generation
    - FLUX.2 [max]: Maximum quality, complex instruction following
    - FLUX.2 [pro]: Production-grade, balance of speed and quality

    For image editing, provide input_images - up to 4 for klein, up to 8 for max/pro.

    Args:
        session: Database session
        model: Model name (e.g., 'flux-2-pro', 'flux-2-max', 'flux-2-klein-9b')
        prompt: Text prompt for image generation
        provider: Provider name (should be 'bfl' or 'blackforestlabs')
        n: Number of images (currently BFL returns 1 per request)
        size: Image size (e.g., '1024x1024') - converted to width/height
        input_images: Optional reference images for editing (list of base64 or data URLs)
        negative_prompt: Not directly supported by FLUX.2, but kept for API compatibility
        seed: For reproducibility
        safety_tolerance: 0 (strict) to 5 (permissive), default 2
        output_format: 'png' or 'jpeg'

    Returns:
        OpenAI-compatible response dict with 'data' array containing generated images

    Raises:
        ImageGenerationError: If the API request fails
    """
    prov = await _get_provider(session, provider)
    api_key = prov.get("api_key")

    # BFL uses its own base URL, but allow override from provider config
    base_url = prov.get("base_url")
    if not base_url or "bfl" not in base_url.lower():
        base_url = BFL_BASE_URL

    if not api_key:
        raise ImageGenerationError(
            f"Provider '{provider}' has no api_key configured. "
            f"Add your Black Forest Labs API key via Admin → Providers.",
            status_code=401,
            provider=provider,
            model=model
        )

    # Determine endpoint based on model
    model_lower = model.lower().strip()
    endpoint_path = BFL_MODEL_ENDPOINTS.get(model_lower)

    if not endpoint_path:
        # Try to match partial model name
        for key, path in BFL_MODEL_ENDPOINTS.items():
            if key in model_lower or model_lower in key:
                endpoint_path = path
                break

    if not endpoint_path:
        # Default to flux-2-pro
        endpoint_path = "/v1/flux-2-pro"

    final_url = f"{base_url.rstrip('/')}{endpoint_path}"

    # Parse size to width/height
    width, height = 1024, 1024
    if size and "x" in size.lower():
        try:
            parts = size.lower().split("x")
            width = int(parts[0])
            height = int(parts[1])
            # BFL requires multiples of 16 and max 4MP
            width = max(64, min(2048, (width // 16) * 16))
            height = max(64, min(2048, (height // 16) * 16))
            # Ensure total pixels <= 4MP
            while width * height > 4_000_000:
                width = int(width * 0.9)
                height = int(height * 0.9)
                width = (width // 16) * 16
                height = (height // 16) * 16
        except Exception:
            pass

    # Build request body
    body: Dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "output_format": output_format.lower() if output_format else "png",
    }

    if seed is not None:
        body["seed"] = int(seed)

    if safety_tolerance is not None:
        body["safety_tolerance"] = max(0, min(5, int(safety_tolerance)))

    # Add input images for editing (BFL uses input_image, input_image_2, etc.)
    if input_images:
        max_images = 4 if "klein" in model_lower else 8
        for i, img in enumerate(input_images[:max_images]):
            parsed = _parse_bfl_input_image(img, i)
            if parsed:
                if i == 0:
                    body["input_image"] = parsed
                else:
                    body[f"input_image_{i + 1}"] = parsed

    headers = {
        "x-key": api_key,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT) as client:
            r = await client.post(final_url, headers=headers, json=body)

            if r.status_code >= 400:
                error_text = r.text
                try:
                    error_json = r.json()
                    error_msg = error_json.get("error", error_json.get("message", error_text[:500]))
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                except Exception:
                    error_msg = error_text[:500]

                raise ImageGenerationError(
                    f"BFL FLUX.2 API returned {r.status_code}: {error_msg}",
                    status_code=r.status_code,
                    provider=provider,
                    model=model,
                    error_detail=error_msg
                )

            result = r.json()

            # BFL response format varies:
            # - Immediate result: {sample: "url or base64", ...}
            # - Async: {id: "...", status: "pending"} -> need to poll

            # Handle immediate result
            images_data = []

            if "sample" in result:
                # Direct result - could be URL or base64
                sample = result["sample"]
                if sample.startswith("http"):
                    # Fetch the image
                    img_resp = await client.get(sample)
                    if img_resp.status_code == 200:
                        import base64
                        b64 = base64.b64encode(img_resp.content).decode("utf-8")
                        images_data.append({
                            "b64_json": b64,
                            "revised_prompt": result.get("prompt", prompt)
                        })
                else:
                    # Already base64
                    images_data.append({
                        "b64_json": sample,
                        "revised_prompt": result.get("prompt", prompt)
                    })

            elif "id" in result:
                # Async result - poll for completion
                task_id = result["id"]
                poll_url = f"{base_url.rstrip('/')}/v1/get_result"

                max_polls = 60  # Max ~2 minutes of polling
                poll_interval = 2.0  # seconds

                for _ in range(max_polls):
                    import asyncio
                    await asyncio.sleep(poll_interval)

                    poll_resp = await client.get(
                        poll_url,
                        params={"id": task_id},
                        headers={"x-key": api_key}
                    )

                    if poll_resp.status_code >= 400:
                        continue

                    poll_result = poll_resp.json()
                    status = poll_result.get("status", "").lower()

                    if status == "ready":
                        sample = poll_result.get("result", {}).get("sample") or poll_result.get("sample")
                        if sample:
                            if sample.startswith("http"):
                                img_resp = await client.get(sample)
                                if img_resp.status_code == 200:
                                    import base64
                                    b64 = base64.b64encode(img_resp.content).decode("utf-8")
                                    images_data.append({
                                        "b64_json": b64,
                                        "revised_prompt": poll_result.get("prompt", prompt)
                                    })
                            else:
                                images_data.append({
                                    "b64_json": sample,
                                    "revised_prompt": poll_result.get("prompt", prompt)
                                })
                        break

                    elif status in ("failed", "error"):
                        error_msg = poll_result.get("error", "Generation failed")
                        raise ImageGenerationError(
                            f"BFL FLUX.2 generation failed: {error_msg}",
                            status_code=500,
                            provider=provider,
                            model=model,
                            error_detail=error_msg
                        )

                if not images_data:
                    raise ImageGenerationError(
                        "BFL FLUX.2 generation timed out waiting for result",
                        status_code=504,
                        provider=provider,
                        model=model
                    )

            # If we got data in another format, try to parse it
            if not images_data and "data" in result:
                for item in result["data"]:
                    if isinstance(item, dict):
                        b64 = item.get("b64_json") or item.get("base64")
                        if b64:
                            images_data.append({
                                "b64_json": b64,
                                "revised_prompt": item.get("revised_prompt", prompt)
                            })

            if not images_data:
                raise ImageGenerationError(
                    "BFL FLUX.2 returned no image data",
                    status_code=502,
                    provider=provider,
                    model=model,
                    error_detail=str(result)[:500]
                )

            return {
                "created": int(__import__("time").time()),
                "data": images_data,
                "model": model,
            }

    except httpx.TimeoutException:
        raise ImageGenerationError(
            f"Timeout after {IMAGE_TIMEOUT}s waiting for BFL FLUX.2",
            status_code=504,
            provider=provider,
            model=model
        )
    except httpx.RequestError as e:
        raise ImageGenerationError(
            f"Network error connecting to BFL FLUX.2: {str(e)}",
            status_code=503,
            provider=provider,
            model=model,
            error_detail=str(e)
        )