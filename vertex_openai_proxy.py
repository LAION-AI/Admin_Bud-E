#!/usr/bin/env python3
"""
Vertex AI (Gemini) ↔ OpenAI-compatible proxy server
- /v1/audio/transcriptions  (Whisper-compatible)
- /v1/chat/completions      (OpenAI Chat Completions; supports text + base64 images/PDFs)
- /v1/responses             (minimal OpenAI Responses compatibility)
- /v1/models                (tiny models listing for clients)
- /admin/settings (GET/POST) to view/change default region/model/project/SA JSON

NEW (TTS):
- /v1/audio/speech          → Google Cloud Text-to-Speech (incl. Chirp 3 HD), returns audio (MP3/WAV/OGG)
- /v1/audio/speech/voices   → List available voices (cached)
- Auto-language/auto-voice (optional): if no voice_name/language_code, detect from text (langdetect)
"""

import os, base64, mimetypes, uuid, time, json, threading
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

# ===========
# Defaults
# ===========

DEFAULTS = {
    "REGION":       os.getenv("VERTEX_REGION",      "europe-west4"),
    "MODEL":        os.getenv("VERTEX_MODEL_NAME",  "gemini-2.0-flash-001"),
    "PROJECT_ID":   os.getenv("VERTEX_PROJECT_ID",  "gemini-school-471209"),
    "SA_JSON":      os.getenv("VERTEX_SA_JSON",     "/etc/gcp/gemini-school-471209-3972614b5ec6.json"),
}

SUPPORTED_AUDIO = {
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
}

def _sse_chat_delta(model_name: str, delta_text: str, idx: int = 0, finish_reason: Optional[str] = None) -> str:
    evt = {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": _now_unix(),
        "model": model_name,
        "choices": [{
            "index": idx,
            "delta": ({"content": delta_text} if delta_text else {}),
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

def _sse_chat_final(model_name: str, usage: Optional[Dict[str, int]], idx: int = 0, finish_reason: str = "stop") -> str:
    # Final chunk that includes OpenAI-shaped usage (if available)
    evt = {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion.chunk",
        "created": _now_unix(),
        "model": model_name,
        "choices": [{
            "index": idx,
            "delta": {},
            "finish_reason": finish_reason,
        }],
    }
    if usage:
        evt["usage"] = usage
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

def normalize_audio_mime(filename: str, content_type: Optional[str]) -> str:
    ct = (content_type or "").lower().strip()
    if ct and ct != "application/octet-stream" and ct.startswith("audio/"):
        if ct in {"audio/x-m4a", "audio/mp4a-latm"}:
            return "audio/mp4"
        return ct
    ext = Path(filename or "").suffix.lower()
    if ext in SUPPORTED_AUDIO:
        return SUPPORTED_AUDIO[ext]
    guess, _ = mimetypes.guess_type(filename or "")
    if guess and guess.startswith("audio/"):
        return guess
    return "audio/wav"

def set_usage_headers(resp: Response, usage: Optional[object]):
    if not usage:
        return
    # Dict style (AI Studio/REST)
    if isinstance(usage, dict):
        if "promptTokenCount" in usage: resp.headers["X-Usage-Prompt-Tokens"] = str(usage["promptTokenCount"])
        if "candidatesTokenCount" in usage: resp.headers["X-Usage-Candidates-Tokens"] = str(usage["candidatesTokenCount"])
        if "totalTokenCount" in usage: resp.headers["X-Usage-Total-Tokens"] = str(usage["totalTokenCount"])
        if "cachedContentTokenCount" in usage: resp.headers["X-Usage-Cached-Content-Tokens"] = str(usage["cachedContentTokenCount"])
        return
    # Vertex SDK usage object
    pt = getattr(usage, "prompt_token_count", None)
    ct = getattr(usage, "candidates_token_count", None)
    tt = getattr(usage, "total_token_count", None)
    cc = getattr(usage, "cached_content_token_count", None)
    if pt is not None: resp.headers["X-Usage-Prompt-Tokens"] = str(pt)
    if ct is not None: resp.headers["X-Usage-Candidates-Tokens"] = str(ct)
    if tt is not None: resp.headers["X-Usage-Total-Tokens"] = str(tt)
    if cc is not None: resp.headers["X-Usage-Cached-Content-Tokens"] = str(cc)

# ====================
# Lazy dep management
# ====================

def _ensure(pkg: str, mod: Optional[str] = None):
    try:
        __import__(mod or pkg.replace("-", "_"))
    except ImportError:
        import subprocess, sys as _s
        subprocess.check_call([_s.executable, "-m", "pip", "install", pkg])
        __import__(mod or pkg.replace("-", "_"))

_ensure("google-cloud-aiplatform", "google.cloud.aiplatform")

# --- TTS deps (NEW) ---
_ensure("google-cloud-texttospeech", "google.cloud.texttospeech")
_ensure("langdetect", "langdetect")
from google.cloud import texttospeech_v1 as tts
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42
# ----------------------

# Vertex SDK
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# ===========
# App setup
# ===========

app = FastAPI(title="Vertex Gemini ↔ OpenAI-compatible proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

STATE = {
    "region":     DEFAULTS["REGION"],
    "model":      DEFAULTS["MODEL"],
    "project_id": DEFAULTS["PROJECT_ID"],
    "sa_json":    DEFAULTS["SA_JSON"],
}
_STATE_LOCK = threading.Lock()

_MODEL_CACHE: Dict[Tuple[str, str, str, str], GenerativeModel] = {}

def _init_vertex(region: str, project_id: str, sa_json: str):
    if sa_json and os.path.isfile(sa_json):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            sa_json, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        vertexai.init(project=project_id, location=region, credentials=creds)
    else:
        vertexai.init(project=project_id, location=region)

def _get_model(model_name: str) -> GenerativeModel:
    key = (STATE["region"], STATE["project_id"], STATE["sa_json"], model_name)
    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m
    _init_vertex(STATE["region"], STATE["project_id"], STATE["sa_json"])
    gm = GenerativeModel(model_name)
    _MODEL_CACHE[key] = gm
    return gm

def _clear_model_cache():
    _MODEL_CACHE.clear()

# =========================
# Helpers: OpenAI payloads
# =========================

def _now_unix() -> int:
    return int(time.time())

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def _parse_data_url(url: str) -> Tuple[bytes, str]:
    if not url.startswith("data:"):
        raise ValueError("Only data: URLs are supported for images/files.")
    head, b64 = url.split(",", 1)
    if ";base64" not in head:
        raise ValueError("Only base64-encoded data: URLs are supported.")
    mime = head[5:].split(";")[0].strip() or "application/octet-stream"
    return base64.b64decode(b64), mime

def _collect_text_and_binary_from_messages(messages: List[Dict[str, Any]]) -> Tuple[str, List[Part]]:
    text_blocks: List[str] = []
    binaries: List[Part] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                text_blocks.append(content)
            continue
        if isinstance(content, list):
            line_chunks: List[str] = []
            for part in content:
                ptype = part.get("type")
                if ptype in {"text", "input_text"}:
                    t = part.get("text") or part.get("input_text") or ""
                    if t:
                        line_chunks.append(t)
                elif ptype in {"image_url", "input_image"}:
                    data_url = None
                    if "image_url" in part and isinstance(part["image_url"], dict):
                        data_url = part["image_url"].get("url")
                    elif "image_url" in part and isinstance(part["image_url"], str):
                        data_url = part["image_url"]
                    if data_url:
                        blob, mime = _parse_data_url(data_url)
                        binaries.append(Part.from_data(mime_type=mime, data=blob))
                elif ptype in {"input_file", "file"}:
                    b64 = part.get("data")
                    mime = part.get("mime_type") or "application/octet-stream"
                    if b64:
                        blob = base64.b64decode(b64 if isinstance(b64, (bytes, bytearray)) else str(b64).encode("utf-8"))
                        binaries.append(Part.from_data(mime_type=mime, data=blob))
            if line_chunks:
                text_blocks.append("\n".join(line_chunks))
    text_prompt = "\n\n".join([t for t in text_blocks if t]).strip()
    return text_prompt, binaries

def _gen_config_from_openai_params(body: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if "temperature" in body and body["temperature"] is not None:
        cfg["temperature"] = float(body["temperature"])
    if "top_p" in body and body["top_p"] is not None:
        cfg["top_p"] = float(body["top_p"])
    if "max_tokens" in body and body["max_tokens"] is not None:
        cfg["max_output_tokens"] = int(body["max_tokens"])
    # Add more mappings (e.g., top_k) here if you like.
    return cfg

def _usage_dict_from_vertex(usage_obj: Any) -> Dict[str, int]:
    if not usage_obj:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pt = getattr(usage_obj, "prompt_token_count", None)
    ct = getattr(usage_obj, "candidates_token_count", None)
    tt = getattr(usage_obj, "total_token_count", None)
    # Prefer exact: completion = total - prompt (when both present). Else use candidates.
    completion = None
    if tt is not None and pt is not None:
        completion = int(tt) - int(pt)
    elif ct is not None:
        completion = int(ct)
    return {
        "prompt_tokens": int(pt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(tt or ((pt or 0) + (ct or 0))),
    }

def _approx_usage_via_count_tokens(gm: GenerativeModel, prompt_parts: List[Any], completion_text: str) -> Dict[str, int]:
    # Fallback if streaming usage_metadata is not exposed by the SDK.
    try:
        pin = gm.count_tokens(prompt_parts)
        pin_total = int(getattr(pin, "total_tokens", 0) or 0)
    except Exception:
        pin_total = 0
    try:
        pout = gm.count_tokens([completion_text] if completion_text else [""])
        pout_total = int(getattr(pout, "total_tokens", 0) or 0)
    except Exception:
        pout_total = 0
    return {
        "prompt_tokens": pin_total,
        "completion_tokens": pout_total,
        "total_tokens": pin_total + pout_total,
    }

# ==========================
# OpenAI-compatible routes
# ==========================

@app.get("/v1/models")
def list_models():
    models = {STATE["model"], *[k[3] for k in _MODEL_CACHE.keys()]}
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": _now_unix(), "owned_by": "google-vertex"} for m in sorted(models)],
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    response: Response,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),  # json | text | srt | vtt
    temperature: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
    translate: Optional[str] = Form(None),
):
    """
    Whisper-compatible endpoint that returns:
    - text transcript in requested format
    - and (for JSON format) OpenAI-style `usage` with prompt/completion/total tokens.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        mime = normalize_audio_mime(file.filename, file.content_type)

        transcribe_guard = (
            "ROLE: You are a strict speech-to-text transcriber.\n"
            "TASK: TRANSCRIBE VERBATIM the spoken words in the provided audio.\n"
            "RULES: Do NOT answer questions, do NOT summarize, do NOT add prefixes like 'Assistant:' or 'User:'. "
            "If 'translate' is true, translate to English; otherwise keep original language. "
            "Return ONLY the transcript text (or the requested caption format)."
        )

        parts = []
        if prompt:
            parts.append(f"Context: {prompt}")
        if language:
            parts.append(f"Audio language hint: {language}.")
        if translate and str(translate).strip().lower() in {"1", "true", "yes", "on"}:
            parts.append("Translate to English; otherwise transcribe verbatim.")
        if response_format.lower() == "srt":
            parts.append("Return a valid SRT file (no commentary).")
        elif response_format.lower() == "vtt":
            parts.append("Return a valid WebVTT starting with 'WEBVTT' (no commentary).")
        else:
            parts.append("Return plain text only (no prefixes, no commentary).")
        instr = " ".join(parts) if parts else "Transcribe the audio; return plain text only."

        model_name = model or STATE["model"]
        gm = _get_model(model_name)

        gen_cfg = {
            "temperature": 0.0,
            "top_p": 0.0,
            "response_mime_type": "text/plain",
        }

        # Generate
        res = gm.generate_content(
            [transcribe_guard, instr, Part.from_data(mime_type=mime, data=audio_bytes)],
            generation_config=gen_cfg,
        )

        # Extract text
        transcript = getattr(res, "text", "") or ""
        if not transcript:
            try:
                cand = res.candidates[0]
                ps = getattr(cand, "content", None)
                if ps and getattr(ps, "parts", None):
                    transcript = "".join([getattr(p, "text", "") or "" for p in ps.parts])
            except Exception:
                transcript = ""
        t_strip = transcript.lstrip()
        if t_strip.lower().startswith("assistant:"):
            transcript = t_strip.split(":", 1)[1].lstrip()

        # Usage: headers + JSON usage (OpenAI-style)
        usage_meta = getattr(res, "usage_metadata", None)
        set_usage_headers(response, usage_meta)

        # OpenAI-style usage dict (fallback to count_tokens if missing)
        try:
            usage_dict = _usage_dict_from_vertex(usage_meta) if usage_meta else None
        except Exception:
            usage_dict = None
        if usage_dict is None:
            try:
                prompt_parts = [transcribe_guard, instr, Part.from_data(mime_type=mime, data=audio_bytes)]
                usage_dict = _approx_usage_via_count_tokens(gm, prompt_parts, transcript)
            except Exception:
                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Format-specific returns
        fmt = response_format.lower()
        if fmt == "text":
            return PlainTextResponse(transcript, status_code=200, headers=response.headers)
        if fmt == "srt":
            return PlainTextResponse(transcript, status_code=200, media_type="application/x-subrip", headers=response.headers)
        if fmt == "vtt":
            out = transcript if transcript.strip().lower().startswith("webvtt") else "WEBVTT\n\n" + transcript
            return PlainTextResponse(out, status_code=200, media_type="text/vtt", headers=response.headers)

        # JSON (OpenAI Whisper-compatible)
        return JSONResponse({"text": transcript, "usage": usage_dict}, status_code=200, headers=response.headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex error in /v1/audio/transcriptions: {e}")

@app.post("/v1/chat/completions")
def chat_completions(body: Dict[str, Any] = Body(...)):
    """
    OpenAI Chat Completions compatibility (+ SSE when stream=true).
    """
    try:
        model_name = body.get("model") or STATE["model"]
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="Missing 'messages'.")

        text_prompt, binaries = _collect_text_and_binary_from_messages(messages)
        system_texts = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        ]
        anti_prefix = "Reply directly without any speaker labels or prefixes (no 'Assistant:' or 'User:')."

        # Build final prompt parts for Gemini
        prompt_parts: List[Any] = []
        preamble = " ".join([anti_prefix] + [s for s in system_texts if s])
        if preamble:
            prompt_parts.append(preamble)
        if text_prompt:
            prompt_parts.append(text_prompt)
        prompt_parts.extend(binaries)

        gm = _get_model(model_name)
        gen_cfg = _gen_config_from_openai_params(body)

        # STREAMING
        if body.get("stream"):
            include_usage = True
            so = body.get("stream_options") or {}
            if "include_usage" in so:
                include_usage = bool(so.get("include_usage"))

            stream = gm.generate_content(
                prompt_parts,
                generation_config=gen_cfg if gen_cfg else None,
                stream=True,
            )

            # We also buffer text so that, if needed, we can approximate output tokens.
            out_buf: List[str] = []

            def iter_sse():
                try:
                    for ch in stream:
                        txt = getattr(ch, "text", None)
                        if not txt:
                            try:
                                cand = ch.candidates[0]
                                ps = getattr(cand, "content", None)
                                if ps and getattr(ps, "parts", None):
                                    txt = "".join([(getattr(p, "text", "") or "") for p in ps.parts])
                            except Exception:
                                txt = ""
                        if txt:
                            t = txt.lstrip()
                            if t.lower().startswith("assistant:"):
                                txt = t.split(":", 1)[1].lstrip()
                            out_buf.append(txt)
                            yield _sse_chat_delta(model_name, txt)

                    # After stream ends, attach usage in a final chunk (OpenAI style)
                    usage_dict: Optional[Dict[str, int]] = None
                    if include_usage:
                        try:
                            usage = getattr(stream, "usage_metadata", None)
                            if usage:
                                usage_dict = _usage_dict_from_vertex(usage)
                        except Exception:
                            usage_dict = None
                        if usage_dict is None:
                            # Fallback approximate usage if SDK didn't expose usage on stream
                            try:
                                usage_dict = _approx_usage_via_count_tokens(gm, prompt_parts, "".join(out_buf))
                            except Exception:
                                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                    # Final chunk with finish_reason and (optionally) usage
                    yield _sse_chat_final(model_name, usage_dict, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                except Exception:
                    # Send an error-shaped finalization to keep clients happy
                    yield _sse_chat_final(model_name, None, finish_reason="error")
                    yield "data: [DONE]\n\n"

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Note: we can't set token-usage headers at the very end of a stream.
            }
            return StreamingResponse(iter_sse(), media_type="text/event-stream", headers=headers)

        # NON-STREAMING
        res = gm.generate_content(prompt_parts, generation_config=gen_cfg if gen_cfg else None)
        output_text = getattr(res, "text", None) or ""
        if output_text:
            t = output_text.lstrip()
            if t.lower().startswith("assistant:"):
                output_text = t.split(":", 1)[1].lstrip()

        usage = getattr(res, "usage_metadata", None)
        usage_dict = _usage_dict_from_vertex(usage)

        return {
            "id": _new_id("chatcmpl"),
            "object": "chat.completion",
            "created": _now_unix(),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }],
            "usage": usage_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex error: {e}")

@app.post("/v1/responses")
def responses_api(body: Dict[str, Any] = Body(...)):
    """
    Minimal compatibility with OpenAI 'Responses' API.
    """
    try:
        model_name = body.get("model") or STATE["model"]
        gm = _get_model(model_name)

        input_ = body.get("input")
        parts: List[Any] = []
        binaries: List[Part] = []
        text_prompt = ""

        if isinstance(input_, str):
            text_prompt = input_
        elif isinstance(input_, list):
            text_prompt, binaries = _collect_text_and_binary_from_messages([{"role":"user","content":input_}])
        else:
            raise HTTPException(status_code=400, detail="Unsupported 'input' type.")

        if text_prompt:
            parts.append(text_prompt)
        parts.extend(binaries)

        gen_cfg = _gen_config_from_openai_params(body)
        res = gm.generate_content(parts, generation_config=gen_cfg if gen_cfg else None)
        output_text = getattr(res, "text", None) or ""
        usage = getattr(res, "usage_metadata", None)
        usage_dict = _usage_dict_from_vertex(usage)

        return {
            "id": _new_id("resp"),
            "object": "response",
            "created": _now_unix(),
            "model": model_name,
            "output": [{"type": "output_text", "text": output_text}],
            "usage": usage_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex error: {e}")

# ====================
# Image Generation
# ====================

# Aspect ratio translation: OpenAI size -> Vertex aspectRatio
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
    """Convert OpenAI-style size (e.g., '1024x1024') to Vertex aspect ratio (e.g., '1:1')."""
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


def _parse_input_image(img: Any) -> Optional[Dict[str, Any]]:
    """
    Parse an input image from various formats into Gemini inlineData format.

    Supports:
    - data:image/...;base64,... URLs
    - Plain base64 strings
    - Dict with {data: base64, mime_type: ...}
    """
    if not img:
        return None

    # Data URL format
    if isinstance(img, str) and img.startswith("data:"):
        try:
            head, b64 = img.split(",", 1)
            mime = head[5:].split(";")[0].strip() or "image/png"
            return {"inlineData": {"mimeType": mime, "data": b64}}
        except Exception:
            return None

    # Plain base64 string (assume PNG)
    if isinstance(img, str) and len(img) > 100:
        return {"inlineData": {"mimeType": "image/png", "data": img}}

    # Dict format {data: ..., mime_type: ...}
    if isinstance(img, dict):
        b64 = img.get("data") or img.get("b64_json") or img.get("base64")
        mime = img.get("mime_type") or img.get("mimeType") or "image/png"
        if b64:
            return {"inlineData": {"mimeType": mime, "data": b64}}

    return None


@app.post("/v1/images/generations")
async def generate_images(body: Dict[str, Any] = Body(...)):
    """
    OpenAI-compatible image generation via Vertex AI.

    Supports two model families:
    1. Gemini models (gemini-3-pro-image-preview, gemini-2.5-flash-image):
       - Multimodal: text + images in -> text + images out
       - Use for image editing, style transfer, etc.

    2. Imagen models (imagen-4.0-generate-001, imagen-4.0-fast-generate-001, etc.):
       - Text-to-image only
       - Higher quality for pure generation

    Request body (OpenAI-compatible with extensions):
    {
        "model": "gemini-3-pro-image-preview",  // or "imagen-4.0-generate-001"
        "prompt": "A serene mountain landscape at sunset",
        "n": 1,                          // Number of images (1-4 for Imagen)
        "size": "1024x1024",             // Translated to aspectRatio
        "response_format": "b64_json",   // or "url" (always returns b64 for Vertex)
        "quality": "standard",           // "standard" or "hd" (maps to model variant)

        // Extensions for image editing (Gemini only):
        "input_images": [                // Reference images for editing
            "data:image/png;base64,...",
            {"data": "base64...", "mime_type": "image/jpeg"}
        ],

        // Imagen-specific options:
        "enhance_prompt": true,          // Use LLM prompt enhancement
        "negative_prompt": "blurry",     // What to avoid
        "seed": 12345                    // For reproducibility
    }

    Response (OpenAI-compatible):
    {
        "created": 1234567890,
        "data": [
            {
                "b64_json": "base64...",
                "revised_prompt": "Enhanced prompt used"  // If enhance_prompt=true
            }
        ],
        "model": "gemini-3-pro-image-preview",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 1290,  // ~1290 tokens per image for Gemini
            "total_tokens": 1390
        }
    }
    """
    try:
        model_name = body.get("model") or "gemini-3-pro-image-preview"
        prompt = body.get("prompt", "")
        n = min(max(1, int(body.get("n", 1))), 4)  # Clamp to 1-4
        size = body.get("size", "1024x1024")
        response_format = body.get("response_format", "b64_json")
        input_images = body.get("input_images") or []
        enhance_prompt = body.get("enhance_prompt", True)
        negative_prompt = body.get("negative_prompt")
        seed = body.get("seed")

        aspect_ratio = _size_to_aspect_ratio(size)

        # Determine which API to use based on model name
        model_lower = model_name.lower()

        if "imagen" in model_lower:
            # Use Imagen :predict API
            return await _imagen_generate(
                model_name=model_name,
                prompt=prompt,
                n=n,
                aspect_ratio=aspect_ratio,
                enhance_prompt=enhance_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
            )
        else:
            # Use Gemini generateContent API (supports image editing)
            return await _gemini_image_generate(
                model_name=model_name,
                prompt=prompt,
                input_images=input_images,
                n=n,
                aspect_ratio=aspect_ratio,
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "image_generation_failed",
                "message": f"Vertex AI image generation error: {str(e)}",
                "model": body.get("model"),
                "type": "server_error"
            }
        )

async def _gemini_image_generate(
    model_name: str,
    prompt: str,
    input_images: List[Any],
    n: int,
    aspect_ratio: str,
) -> Dict[str, Any]:
    """
    Generate images using Gemini's Vertex AI endpoint via Service Account.
    Replaces the AI Studio API path to use the Cloud Service Account instead.
    """
    import httpx
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleAuthRequest

    # 1. Use the Service Account configuration from the global STATE
    sa_json = STATE["sa_json"]
    project_id = STATE["project_id"]
    region = STATE["region"]

    # 2. Authenticate using the Service Account JSON (same logic as Imagen)
    try:
        creds = service_account.Credentials.from_service_account_file(
            sa_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(GoogleAuthRequest())
        access_token = creds.token
    except Exception as e:
        raise HTTPException(
            503,
            detail={"error": "auth_failed", "message": f"Service Account Auth Failed: {str(e)}"}
        )

    # 3. Use the Cloud Vertex Endpoint (NOT the AI Studio URL)
    #    Global endpoint uses aiplatform.googleapis.com without region prefix
    if region == "global":
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_name}:generateContent"
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_name}:generateContent"

    # 4. Build parts: text prompt + optional input images
    parts = []
    for img in input_images:
        parsed = _parse_input_image(img)
        if parsed:
            parts.append(parsed)
    if prompt:
        parts.append({"text": prompt})

    request_body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio}
        }
    }

    # 5. Make the request using the Bearer Token
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json=request_body,
        )

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.json())

        result = resp.json()

    # 6. Parse response (Extract images and token usage)
    images_data = []
    text_response = ""
    try:
        candidates = result.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            for part in content.get("parts", []):
                if "text" in part:
                    text_response += part["text"]
                elif "inlineData" in part:
                    inline = part["inlineData"]
                    images_data.append({
                        "b64_json": inline.get("data", ""),
                        "mime_type": inline.get("mimeType", "image/png"),
                        "revised_prompt": text_response or prompt
                    })
    except Exception as e:
        raise HTTPException(502, detail={"error": "parse_error", "message": str(e)})

    usage = result.get("usageMetadata", {})
    return {
        "created": _now_unix(),
        "data": images_data,
        "model": model_name,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }
    }


async def _imagen_generate(
    model_name: str,
    prompt: str,
    n: int,
    aspect_ratio: str,
    enhance_prompt: bool,
    negative_prompt: Optional[str],
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    Generate images using Vertex AI Imagen :predict API.
    Text-to-image only, higher quality for pure generation.
    """
    import httpx
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleAuthRequest

    # Get credentials
    sa_json = STATE["sa_json"]
    project_id = STATE["project_id"]
    region = STATE["region"]

    if not sa_json or not os.path.isfile(sa_json):
        raise HTTPException(
            503,
            detail={
                "error": "missing_credentials",
                "message": "Vertex AI service account JSON not configured",
                "hint": "Set VERTEX_SA_JSON environment variable or configure via /admin/settings"
            }
        )

    # Get access token
    try:
        creds = service_account.Credentials.from_service_account_file(
            sa_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(GoogleAuthRequest())
        access_token = creds.token
    except Exception as e:
        raise HTTPException(
            503,
            detail={
                "error": "auth_failed",
                "message": f"Failed to authenticate with Google Cloud: {str(e)}",
                "hint": "Check your service account JSON file"
            }
        )

    if region == "global":
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_name}:predict"
    else:
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_name}:predict"

    # Build request
    instance = {"prompt": prompt}
    parameters = {
        "sampleCount": n,
        "aspectRatio": aspect_ratio,
        "addWatermark": True,
    }

    if enhance_prompt is not None:
        parameters["enhancePrompt"] = bool(enhance_prompt)

    if negative_prompt:
        parameters["negativePrompt"] = negative_prompt

    if seed is not None:
        parameters["seed"] = int(seed)
        # Note: seed and sampleCount > 1 are mutually exclusive
        if n > 1:
            parameters["sampleCount"] = 1

    request_body = {
        "instances": [instance],
        "parameters": parameters
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )

        if resp.status_code >= 400:
            error_body = resp.text
            try:
                error_json = resp.json()
                error_detail = error_json.get("error", {})
                error_msg = error_detail.get("message", error_body[:500])
                error_status = error_detail.get("status", "UNKNOWN")
            except Exception:
                error_msg = error_body[:500]
                error_status = "UNKNOWN"

            raise HTTPException(
                status_code=resp.status_code,
                detail={
                    "error": "imagen_api_error",
                    "message": error_msg,
                    "status": error_status,
                    "model": model_name,
                    "http_status": resp.status_code
                }
            )

        result = resp.json()

    # Parse predictions
    predictions = result.get("predictions", [])
    if not predictions:
        raise HTTPException(
            502,
            detail={
                "error": "no_images_generated",
                "message": "Imagen did not return any images. The prompt may have been blocked by safety filters.",
                "model": model_name,
                "raw_response": str(result)[:500]
            }
        )

    images_data = []
    for pred in predictions:
        b64 = pred.get("bytesBase64Encoded", "")
        mime = pred.get("mimeType", "image/png")
        revised = pred.get("prompt", prompt)  # Enhanced prompt if available

        if b64:
            images_data.append({
                "b64_json": b64,
                "mime_type": mime,
                "revised_prompt": revised
            })

    if not images_data:
        # Check for filtered content
        filtered_reasons = [p.get("raiFilteredReason") for p in predictions if p.get("raiFilteredReason")]
        if filtered_reasons:
            raise HTTPException(
                400,
                detail={
                    "error": "content_filtered",
                    "message": "Images were filtered by safety systems",
                    "reasons": filtered_reasons,
                    "model": model_name
                }
            )

        raise HTTPException(
            502,
            detail={
                "error": "no_images_in_response",
                "message": "Imagen response contained no image data",
                "model": model_name
            }
        )

    return {
        "created": _now_unix(),
        "data": images_data,
        "model": model_name,
        "usage": {
            "prompt_tokens": len(prompt.split()),  # Approximate
            "completion_tokens": len(images_data) * 1000,  # Placeholder
            "total_tokens": len(prompt.split()) + len(images_data) * 1000,
        }
    }


# ====================
# Music Generation (Lyria 2 + Lyria 3 Pro)
# ====================
@app.post("/v1/audio/generations")
async def generate_music(body: Dict[str, Any] = Body(...)):
    """
    Music generation via Vertex AI Lyria.

    Supports:
    - Lyria 2 (lyria-002): instrumental music, 32.8s WAV, predict API
    - Lyria 3 Pro (lyria-3-pro-preview): full songs with lyrics, up to 184s MP3, interactions API
    - Lyria 3 Clip (lyria-3-clip-preview): 30s clips, interactions API

    Lyria 3 Pro extra fields:
    - lyrics: str — user-provided lyrics (use [Verse], [Chorus], [Bridge] tags)
    - caption: str — song description / style guidance
    """
    import httpx
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleAuthRequest

    try:
        model_name = body.get("model", "lyria-3-pro-preview")
        prompt = body.get("prompt", "")
        negative_prompt = body.get("negative_prompt")
        lyrics = body.get("lyrics")
        caption = body.get("caption")
        n = min(max(1, int(body.get("n", 1))), 4)
        seed = body.get("seed")

        if not prompt and not lyrics and not caption:
            raise HTTPException(400, detail="prompt, lyrics, or caption is required for music generation")

        sa_json = STATE["sa_json"]
        project_id = STATE["project_id"]
        region = STATE["region"]

        if not sa_json or not os.path.isfile(sa_json):
            raise HTTPException(503, detail="Vertex service account JSON not configured")

        # 1. Get Access Token
        creds = service_account.Credentials.from_service_account_file(
            sa_json, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(GoogleAuthRequest())
        access_token = creds.token

        is_lyria3 = model_name.startswith("lyria-3")

        if is_lyria3:
            # ─── Lyria 3 Pro / Clip: Interactions API ───
            url = f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/global/interactions"

            # Build input array
            input_parts = []

            # Combine prompt, caption, and lyrics into a rich text prompt
            text_parts = []
            if caption:
                text_parts.append(f"Style: {caption}")
            if prompt:
                text_parts.append(prompt)
            if lyrics:
                text_parts.append(f"\nLyrics:\n{lyrics}")
            if negative_prompt:
                text_parts.append(f"\nAvoid: {negative_prompt}")

            combined_text = "\n".join(text_parts) if text_parts else prompt
            input_parts.append({"type": "text", "text": combined_text})

            request_body = {
                "model": model_name,
                "input": input_parts,
            }

            print(f"\n[LYRIA3 DEBUG] Request to Google: {json.dumps(request_body, indent=2)}")

            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    json=request_body,
                )

                print(f"[LYRIA3 DEBUG] Google Status: {resp.status_code}")

                if resp.status_code >= 400:
                    error_text = resp.text[:1000]
                    print(f"[LYRIA3 DEBUG] Error Body: {error_text}")
                    try:
                        raise HTTPException(status_code=resp.status_code, detail=resp.json())
                    except Exception:
                        raise HTTPException(status_code=resp.status_code, detail=error_text)

                result = resp.json()

            print(f"[LYRIA3 DEBUG] Response status: {result.get('status')}")

            # Parse Lyria 3 response: outputs[] with type=audio and type=text
            outputs = result.get("outputs", [])
            audio_data = []
            generated_lyrics = ""
            description = ""
            total_duration = 0.0

            for out in outputs:
                out_type = out.get("type", "")
                if out_type == "audio":
                    b64 = out.get("data", "")
                    mime = out.get("mime_type", "audio/mpeg")
                    if b64:
                        # Estimate duration from base64 size (MP3 ~192kbps = 24KB/s)
                        raw_size = len(b64) * 3 / 4
                        duration = raw_size / 24000 if "mpeg" in mime or "mp3" in mime else raw_size / 96000
                        duration = min(duration, 184.0)
                        total_duration += duration
                        audio_data.append({
                            "b64_json": b64,
                            "mime_type": mime,
                            "duration_seconds": round(duration, 1),
                        })
                elif out_type == "text":
                    text = out.get("text", "")
                    # First text output is usually lyrics, second is description
                    if not generated_lyrics:
                        generated_lyrics = text
                    else:
                        description = text

            if not audio_data:
                print(f"[LYRIA3 DEBUG] No audio in outputs. Full response: {json.dumps(result, indent=2)}")
                raise HTTPException(502, detail={
                    "error": "no_audio_in_response",
                    "message": "Lyria 3 response contained no audio data.",
                    "status": result.get("status"),
                    "outputs_count": len(outputs),
                })

            response = {
                "created": _now_unix(),
                "data": audio_data,
                "model": model_name,
                "usage": {"duration_seconds": round(total_duration, 1), "clip_count": len(audio_data)},
            }
            if generated_lyrics:
                response["lyrics"] = generated_lyrics
            if description:
                response["description"] = description

            return response

        else:
            # ─── Lyria 2: Predict API (legacy) ───
            if region == "global":
                url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_name}:predict"
            else:
                url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_name}:predict"

            instance: Dict[str, Any] = {"prompt": prompt}
            if negative_prompt:
                instance["negative_prompt"] = negative_prompt

            parameters: Dict[str, Any] = {}
            if seed is not None:
                instance["seed"] = int(seed)
            else:
                parameters["sample_count"] = n

            request_body = {"instances": [instance]}
            if parameters:
                request_body["parameters"] = parameters

            print(f"\n[LYRIA2 DEBUG] Request to Google: {json.dumps(request_body, indent=2)}")

            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    json=request_body,
                )

                print(f"[LYRIA2 DEBUG] Google Status: {resp.status_code}")

                if resp.status_code >= 400:
                    print(f"[LYRIA2 DEBUG] Error Body: {resp.text}")
                    raise HTTPException(status_code=resp.status_code, detail=resp.json())

                result = resp.json()

            predictions = result.get("predictions", [])
            if not predictions:
                print(f"[LYRIA2 DEBUG] Empty predictions. Full Response: {json.dumps(result, indent=2)}")

            audio_data = []
            total_duration = 0.0

            for pred in predictions:
                b64 = pred.get("audioContent") or pred.get("bytesBase64Encoded")
                mime = pred.get("mimeType", "audio/wav")
                if b64:
                    duration = 32.8
                    total_duration += duration
                    audio_data.append({"b64_json": b64, "mime_type": mime, "duration_seconds": duration})
                else:
                    filter_reason = pred.get("raiFilteredReason")
                    if filter_reason:
                        print(f"[LYRIA2 DEBUG] Content was filtered! Reason: {filter_reason}")

            if not audio_data:
                raise HTTPException(502, detail={
                    "error": "no_audio_in_response",
                    "message": "Lyria response contained no audio data.",
                    "debug_predictions": predictions[:1]
                })

            return {
                "created": _now_unix(),
                "data": audio_data,
                "model": model_name,
                "usage": {"duration_seconds": total_duration, "clip_count": len(audio_data)}
            }

    except Exception as e:
        print(f"[LYRIA DEBUG] Exception: {str(e)}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=502, detail=str(e))



# ====================
# Admin configuration
# ====================

@app.get("/admin/settings")
def get_settings():
    with _STATE_LOCK:
        return dict(STATE)

@app.post("/admin/settings")
def set_settings(body: Dict[str, Any] = Body(...)):
    """
    Change defaults at runtime. Any of: region, model, project_id, sa_json
    """
    changed = False
    with _STATE_LOCK:
        for k in ("region","model","project_id","sa_json"):
            if k in body and body[k] and body[k] != STATE[k]:
                STATE[k] = body[k]
                changed = True
        if changed:
            _clear_model_cache()
            _clear_voices_cache()  # TTS cache reset if settings change
    return {"ok": True, "changed": changed, "state": dict(STATE)}

# ============
# Root ping
# ============

@app.get("/")
def root():
    return {
        "ok": True,
        "msg": "Vertex Gemini ↔ OpenAI proxy is up.",
        "region": STATE["region"],
        "model": STATE["model"],
        "project_id": STATE["project_id"],
    }

# =========================
# TTS (Google Cloud TTS)  |
# =========================

# --- small voice cache (per settings) ---
_VOICES_CACHE: Dict[str, Dict[str, Any]] = {}
_VOICES_CACHE_TTL_SEC = 1800  # 30 min

def _voices_cache_key() -> str:
    return f"{STATE['region']}|{STATE['project_id']}|{STATE['sa_json']}"

def _clear_voices_cache():
    _VOICES_CACHE.clear()

def _tts_endpoint_for_region() -> str:
    # Region-gerechte Endpoints (ggf. global fallback)
    region = (STATE["region"] or "").lower()
    if region.startswith("europe-"):
        return "eu-texttospeech.googleapis.com"
    if region.startswith("us-"):
        return "us-texttospeech.googleapis.com"
    return "texttospeech.googleapis.com"  # global

def _get_tts_client():
    from google.oauth2 import service_account
    creds = None
    if STATE["sa_json"] and os.path.isfile(STATE["sa_json"]):
        creds = service_account.Credentials.from_service_account_file(
            STATE["sa_json"], scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    endpoint = _tts_endpoint_for_region()
    return tts.TextToSpeechClient(client_options={"api_endpoint": endpoint}, credentials=creds)

def _list_voices(language_code: Optional[str] = None) -> List[tts.Voice]:
    key = _voices_cache_key()
    now = time.time()
    entry = _VOICES_CACHE.get(key)
    if entry and now - entry["ts"] < _VOICES_CACHE_TTL_SEC:
        voices = entry["voices"]
    else:
        client = _get_tts_client()
        voices = list(client.list_voices().voices)
        _VOICES_CACHE[key] = {"ts": now, "voices": voices}

    if language_code:
        lc = language_code.lower()
        v = [v for v in voices if any(l.lower().startswith(lc) for l in v.language_codes)]
        if v:
            return v
    return voices

def _detect_language_code(text: str) -> str:
    try:
        lang = detect(text or "")
    except Exception:
        lang = "en"
    mapping = {
        "en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "it": "it-IT",
        "pt": "pt-BR", "nl": "nl-NL", "pl": "pl-PL", "ru": "ru-RU",
        "sv": "sv-SE", "da": "da-DK", "fi": "fi-FI", "no": "nb-NO",
        "tr": "tr-TR", "ja": "ja-JP", "ko": "ko-KR",
        "zh-cn": "cmn-CN", "zh-tw": "cmn-TW",
        "ar": "ar-XA", "hi": "hi-IN",
    }
    return mapping.get(lang.lower(), "en-US")

def _choose_voice(language_code: str, prefer: str = "chirp") -> Optional[str]:
    voices = _list_voices(language_code)
    names = [v.name for v in voices]
    # Preferenzen: Chirp 3 HD → Neural2 → Standard
    if prefer == "chirp":
        cand = [n for n in names if "Chirp" in n and "HD" in n]
        if cand: return sorted(cand)[0]
    if prefer in {"neural","chirp"}:
        cand = [n for n in names if "Neural2" in n]
        if cand: return sorted(cand)[0]
    cand = [n for n in names if "Standard" in n]
    if cand: return sorted(cand)[0]
    return names[0] if names else None

def _audio_encoding_and_mime(fmt: str) -> Tuple[tts.AudioEncoding, str, str]:
    fmt = (fmt or "mp3").lower()
    enc_map = {
        "mp3": tts.AudioEncoding.MP3,
        "wav": tts.AudioEncoding.LINEAR16,
        "linear16": tts.AudioEncoding.LINEAR16,
        "ogg": tts.AudioEncoding.OGG_OPUS,
        "opus": tts.AudioEncoding.OGG_OPUS,
        "mulaw": tts.AudioEncoding.MULAW,
        "alaw": tts.AudioEncoding.ALAW,
    }
    audio_encoding = enc_map.get(fmt, tts.AudioEncoding.MP3)
    mime_map = {
        tts.AudioEncoding.MP3: "audio/mpeg",
        tts.AudioEncoding.LINEAR16: "audio/wav",
        tts.AudioEncoding.OGG_OPUS: "audio/ogg",
        tts.AudioEncoding.MULAW: "audio/basic",
        tts.AudioEncoding.ALAW: "audio/basic",
    }
    mime = mime_map[audio_encoding]
    ext = "mp3" if audio_encoding == tts.AudioEncoding.MP3 else ("wav" if audio_encoding == tts.AudioEncoding.LINEAR16 else "audio")
    return audio_encoding, mime, ext

@app.get("/v1/audio/speech/voices")
def list_voices_api(language_code: Optional[str] = None):
    """
    Optional helper to inspect available voices.
    GET /v1/audio/speech/voices?language_code=de-DE
    """
    try:
        voices = _list_voices(language_code)
        data = []
        for v in voices:
            data.append({
                "name": v.name,
                "language_codes": list(v.language_codes),
                "ssml_gender": tts.SsmlVoiceGender(v.ssml_gender).name if v.ssml_gender is not None else "SSML_VOICE_GENDER_UNSPECIFIED",
                "natural_sample_rate_hertz": getattr(v, "natural_sample_rate_hertz", None),
            })
        return {"voices": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS voices error: {e}")

@app.post("/v1/audio/speech")
def tts_synthesize(body: Dict[str, Any] = Body(...)):
    """
    Synthesize speech via Google Cloud Text-to-Speech.

    JSON body:
    {
      "text": "Hello world",             # or "ssml": "<speak>...</speak>" (ignored by Chirp 3 HD)
      "ssml": "<speak>...</speak>",
      "voice_name": "en-US-Chirp-HD-F",  # optional; if omitted -> auto language + auto voice
      "language_code": "en-US",          # optional; if omitted -> auto detect from text/ssml
      "prefer": "chirp",                 # "chirp" | "neural" | "standard" (default: chirp)
      "audio_format": "mp3",             # mp3 | wav | ogg | opus | mulaw | alaw
      "speaking_rate": 1.0,              # ignored for Chirp
      "pitch": 0.0,                      # ignored for Chirp
      "effects_profile_id": []           # optional
    }

    Returns: binary audio with proper Content-Type, or JSON error on failure.
    """
    try:
        text = (body.get("text") or "").strip()
        ssml = (body.get("ssml") or "").strip()
        if not text and not ssml:
            raise HTTPException(status_code=400, detail="Provide 'text' or 'ssml'.")

        prefer = (body.get("prefer") or "chirp").lower().strip()
        if prefer not in {"chirp", "neural", "standard"}:
            prefer = "chirp"

        voice_name = body.get("voice_name")
        language_code = body.get("language_code")
        audio_format = (body.get("audio_format") or "mp3").lower()

        # Auto language from content if not provided
        if not language_code:
            sample = ssml if ssml else text
            language_code = _detect_language_code(sample)

        # Auto voice selection if not provided
        if not voice_name:
            if prefer == "neural":
                voice_name = _choose_voice(language_code, prefer="neural") or _choose_voice(language_code, prefer="chirp")
            elif prefer == "standard":
                voice_name = _choose_voice(language_code, prefer="standard")
            else:
                voice_name = _choose_voice(language_code, prefer="chirp") or _choose_voice(language_code, prefer="neural")
        if not voice_name:
            raise HTTPException(status_code=404, detail=f"No voice found for language_code={language_code}")

        audio_encoding, mime, ext = _audio_encoding_and_mime(audio_format)

        # Build TTS request
        if ssml:
            synthesis_input = tts.SynthesisInput(ssml=ssml)
        else:
            synthesis_input = tts.SynthesisInput(text=text)

        voice_params = tts.VoiceSelectionParams(language_code=language_code, name=voice_name)

        cfg_kwargs = {"audio_encoding": audio_encoding}
        # Chirp 3 HD: speaking_rate/pitch werden ignoriert – setze sie nur bei Nicht-Chirp
        if not ("Chirp" in voice_name and "HD" in voice_name):
            if "speaking_rate" in body and body["speaking_rate"] is not None:
                cfg_kwargs["speaking_rate"] = float(body["speaking_rate"])
            if "pitch" in body and body["pitch"] is not None:
                cfg_kwargs["pitch"] = float(body["pitch"])
        if "effects_profile_id" in body and body["effects_profile_id"]:
            cfg_kwargs["effects_profile_id"] = list(body["effects_profile_id"])

        audio_config = tts.AudioConfig(**cfg_kwargs)

        client = _get_tts_client()
        resp = client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_config)

        data = resp.audio_content or b""
        if not data:
            # Explizit JSON-Fehler zurückgeben, damit niemand ein leeres „MP3“ speichert
            raise HTTPException(status_code=502, detail="Empty audio from Text-to-Speech.")

        headers = {
            "Content-Type": mime,
            "Content-Disposition": f'inline; filename="speech.{ext}"',
            "X-TTS-Voice-Name": voice_name,
            "X-TTS-Language-Code": language_code,
            "X-TTS-Endpoint": _tts_endpoint_for_region(),
        }
        return Response(content=data, media_type=mime, headers=headers)

    except HTTPException:
        # Immer JSON bei Fehlern (keine Binärantwort → kein „defektes MP3“)
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS error: {e}")
