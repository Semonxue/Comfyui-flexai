"""OpenAIVideoNode - OpenAI-compatible video generation (ComfyUI FlexAI plugin).

Features:
  - Text-to-video, first-frame, first/last-frame and multi-subject reference
    (Seedance 2.x `content[]` roles) via `/v1/videos`, plus reference video
    (`videoExtend` on 2.5) and reference audio.
  - Geometry is expressed the way modern video models actually take it:
    a resolution tier (`resolution`: 480p/720p/1080p/4k + super_* upscaling)
    + an aspect ratio (`ratio`: 16:9 / 9:16 / 1:1 / 4:3 / 3:4 / 21:9 / adaptive)
    + `duration`. No pixel size is required; the model decides the pixels.
  - Seedance family rules are enforced per model id (2.0, 2.0 fast/mini, 2.5):
    allowed resolution tiers, duration range, 2.5's forced `ratio=adaptive` for
    frame-driven runs, `output_format` and `videoExtend` availability.
  - Async job handling: create -> poll status -> download content (MP4).
  - Tolerant response parsing, so third-party OpenAI-compatible proxies work too:
      * OpenAI/Sora style:   {"id": "video_123", "status": "queued"|"in_progress"|"completed"}
      * Direct URL style:    {"url"/"video_url"/"download_url": "https://.../out.mp4"}
      * Wrapped style:       {"code": 0, "data": {"status": "SUCCESS", "video_url": "..."}}
  - Returns a native ComfyUI `VIDEO` output (works with SaveVideo / Get Video
    Components / any VIDEO consumer), instead of materializing frames, so long
    clips don't blow up RAM.
  - Retries transient errors (429/5xx) and re-polls `/content` until the file is
    actually available.
  - Detailed request/response logging into debug.log when `debug` is enabled.

Only registered when `comfy_api.latest.io.Video` exists (VIDEO output type needs a
modern ComfyUI); older installs keep using the image/text nodes.
"""
from __future__ import annotations

import os
import re
import time
import base64
import importlib.util
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

import provider_config

# -- Dynamically load utility modules (same pattern as the image/text nodes) --
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_UTILS_DIR = os.path.join(_PLUGIN_ROOT, 'nodes', 'utils')


def _load_utils_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTILS_DIR, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_images_module = _load_utils_module('images')
_openai_standard_module = _load_utils_module('openai_standard')
_model_manager_module = _load_utils_module('model_manager')

tensor_to_pil = _images_module.tensor_to_pil
debug_log = _openai_standard_module.debug_log
log_api_interaction = _openai_standard_module.log_api_interaction
get_models = _model_manager_module.get_models
add_model = _model_manager_module.add_model

load_dotenv(os.path.join(_PLUGIN_ROOT, '.env'), override=True)

_MODEL_KEY = "flexai_video_models"
_DEFAULT_MODEL = "doubao-seedance-2-0-260128"

# ---- v3 schema / VIDEO type (deferred so the module still imports on old ComfyUI) ----
try:
    from comfy_api.latest import io as _v3_io
    from comfy_api.latest import VideoFromFile as _VideoFromFile
    _V3_AVAILABLE = hasattr(_v3_io, "Video")
except ImportError:
    _v3_io = None
    _VideoFromFile = None
    _V3_AVAILABLE = False

# Video geometry is NOT expressed in output pixels by modern video APIs: the
# model takes a RESOLUTION TIER + an ASPECT RATIO and decides the exact pixel
# pair itself (which may be an unusual number, e.g. 480p 16:9 -> ~854x480).
# Seedance / Volcengine Ark example payload:
#     {"model": "...", "content": [...], "ratio": "16:9",
#      "resolution": "720p", "duration": 8, "generate_audio": true}
# So this node sends `ratio` + `resolution` (+ `duration`) and only derives a
# nominal "WxH" for legacy OpenAI/Sora-style channels that require `size`.
_VIDEO_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]
_VIDEO_RESOLUTIONS = ["480p", "720p", "1080p", "4k", "super_720p", "super_1080p", "super_4k"]

_DEFAULT_RATIO = "16:9"
_DEFAULT_RESOLUTION = "720p"
_DEFAULT_DURATION = 8

# Nominal short side per resolution tier (the shorter side equals this value).
_RES_SHORT_SIDE = {
    "480p": 480, "720p": 720, "1080p": 1080, "4k": 2160,
    "super_720p": 720, "super_1080p": 1080, "super_4k": 2160,
}

# --- Seedance 2.x capabilities -------------------------------------------------
# Values below follow the gateway documentation for /v1/videos: the wire format
# uses upper-case resolution tokens (720P / 4K / SUPER_720P ...), while the node
# keeps lower-case combo values for readability.
_RESOLUTION_WIRE = {
    "480p": "480P",
    "720p": "720P",
    "1080p": "1080P",
    "4k": "4K",
    "super_720p": "SUPER_720P",
    "super_1080p": "SUPER_1080P",
    "super_4k": "SUPER_4K",
}
_SUPER_RESOLUTIONS = ["super_720p", "super_1080p", "super_4k"]

_SD25 = "seedance-2.5"
_SD20 = "seedance-2.0"
_SD20_FAST = "seedance-2.0-fast"
_SD20_MINI = "seedance-2.0-mini"
_LEGACY = "legacy"

_FAMILY_CAPS = {
    _SD25: {
        "label": "Seedance 2.5",
        "resolutions": ["480p", "720p", "1080p"],
        "duration": (4, 30),
        "default_ratio": "adaptive",
        "output_format": True,
        "video_extend": True,
        # 2.5 requires ratio=adaptive whenever a first/last frame is involved.
        "frames_need_adaptive": True,
    },
    _SD20: {
        "label": "Seedance 2.0",
        "resolutions": ["480p", "720p", "1080p", "4k"] + _SUPER_RESOLUTIONS,
        "duration": (4, 15),
        "default_ratio": "16:9",
        "output_format": False,
        "video_extend": False,
        "frames_need_adaptive": False,
    },
    _SD20_FAST: {
        "label": "Seedance 2.0 Fast",
        "resolutions": ["480p", "720p"] + _SUPER_RESOLUTIONS,
        "duration": (4, 15),
        "default_ratio": "16:9",
        "output_format": False,
        "video_extend": False,
        "frames_need_adaptive": False,
    },
    _SD20_MINI: {
        "label": "Seedance 2.0 Mini",
        "resolutions": ["480p", "720p"] + _SUPER_RESOLUTIONS,
        "duration": (4, 15),
        "default_ratio": "16:9",
        "output_format": False,
        "video_extend": False,
        "frames_need_adaptive": False,
    },
    _LEGACY: {
        "label": "non-Seedance / legacy channel",
        "resolutions": _VIDEO_RESOLUTIONS,
        "duration": (1, 60),
        "default_ratio": "16:9",
        "output_format": False,
        "video_extend": False,
        "frames_need_adaptive": False,
    },
}


def _seedance_family(model: str) -> str:
    """Map a model id to a Seedance capability family (2.0+ only, else legacy)."""
    name = (model or "").lower()
    if "seedance" not in name and "doubao" not in name:
        return _LEGACY
    # Anchor on the "seedance" token: date suffixes like -250528 must not be
    # mistaken for "2.5".
    # Anchor on the "seed"/"seedance" token: date suffixes like -250528 must
    # not be mistaken for "2.5", and both "seedance-2-0" and the shorter
    # gateway alias "seed-2-0" must be recognised.
    has_25 = bool(re.search(r"seed(?:ance)?[-_.\s]?2[-_.\s]?5", name))
    has_20 = bool(re.search(r"seed(?:ance)?[-_.\s]?2[-_.\s]?0", name))
    if has_25:
        return _SD25
    if has_20:
        if "fast" in name:
            return _SD20_FAST
        if "mini" in name:
            return _SD20_MINI
        return _SD20
    return _LEGACY


def _caps_for(model: str) -> dict:
    return _FAMILY_CAPS[_seedance_family(model)]


def _wire_resolution(resolution: str, caps: dict) -> str:
    """Validate the resolution against the model family and return its wire token."""
    value = (resolution or _DEFAULT_RESOLUTION).strip().lower()
    allowed = caps["resolutions"]
    if value not in allowed:
        raise ValueError(
            f"resolution '{resolution}' is not supported by {caps['label']}. "
            f"Supported: {', '.join(allowed)}."
        )
    return _RESOLUTION_WIRE[value]


def _clamp_duration(duration, caps: dict, debug: bool) -> int:
    low, high = caps["duration"]
    value = max(1, int(duration or _DEFAULT_DURATION))
    if value < low or value > high:
        clamped = min(max(value, low), high)
        if debug:
            debug_log(
                f"[video] duration {value}s is outside {caps['label']} range "
                f"{low}-{high}s, using {clamped}s."
            )
        return clamped
    return value

_RATIO_WH = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
}

# Transient HTTP statuses worth retrying.
_RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)

# Terminal job states.
_DONE_STATES = {"completed", "complete", "succeeded", "success", "done", "finished"}
_FAILED_STATES = {"failed", "failure", "error", "cancelled", "canceled", "expired"}
_RUNNING_STATES = {"queued", "pending", "in_progress", "in-progress", "processing", "running", "generating"}


def _provider_names():
    return provider_config.get_provider_display_names() or ["default"]


def _model_names():
    return get_models(_MODEL_KEY)


# ---------------------------------------------------------------- URL helpers

def _normalize_base_url(base_url: str) -> str:
    """Normalize a provider base_url to a `/v1`-rooted API root.

    ``https://api.openai.com/v1``      -> unchanged
    ``https://api.example.com``        -> ``https://api.example.com/v1``
    ``https://api.example.com/v1/``    -> trailing slash removed
    """
    base = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    if not re.search(r"/v\d+$", base):
        base = base + "/v1"
    return base


def _videos_url(base_url: str) -> str:
    return f"{_normalize_base_url(base_url)}/videos"


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _clean_size(size: str) -> str:
    """Strip the descriptive ``(16:9)`` annotation, returning e.g. ``1280x720``."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", (size or "").strip()).strip()


def _parse_wh(size: str) -> Optional[Tuple[int, int]]:
    """Parse a ``WxH`` string (annotation allowed) into ``(w, h)``."""
    m = re.match(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*(?:\([^)]*\))?\s*$", size or "")
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    return (w, h) if w > 0 and h > 0 else None


def _canonical_size(ratio: str, resolution: str) -> Optional[str]:
    """Nominal ``WxH`` for (ratio, resolution), or ``None`` for adaptive/unknown.

    The shorter side equals the resolution number, rounded to an even value —
    the same rule ByteDance's own integration uses. This is only a *nominal*
    size (sent to legacy OpenAI-compatible channels and used to pre-resize the
    reference image); the model itself decides the real output pixels.
    """
    wh = _RATIO_WH.get(ratio)
    short = _RES_SHORT_SIDE.get(resolution)
    if not wh or not short:
        return None
    rw, rh = wh
    if rw >= rh:
        w, h = round(short * rw / rh), short
    else:
        w, h = short, round(short * rh / rw)
    w, h = w - w % 2, h - h % 2
    return f"{w}x{h}"


def _ratio_from_wh(w: int, h: int) -> str:
    """Nearest supported aspect ratio for a pixel pair."""
    aspect = w / h
    return min(_RATIO_WH, key=lambda k: abs(_RATIO_WH[k][0] / _RATIO_WH[k][1] - aspect))


def _resolution_from_wh(w: int, h: int) -> str:
    """Nearest supported resolution tier for a pixel pair (by shorter side)."""
    short = min(w, h)
    return min(_RES_SHORT_SIDE, key=lambda k: abs(_RES_SHORT_SIDE[k] - short))


def _geometry(ratio: str, resolution: str, custom_size: str) -> Tuple[str, str, Optional[str], Optional[Tuple[int, int]]]:
    """Resolve the request geometry.

    Returns ``(ratio, resolution, nominal_size, target_wh)``. ``custom_size``
    (explicit ``WxH``) is a legacy escape hatch: when given, the ratio and
    resolution tiers are derived from it so both request styles stay coherent.
    """
    custom = _clean_size(custom_size)
    wh = _parse_wh(custom) if custom else None
    if wh:
        w, h = wh
        resolved_ratio = ratio if ratio in _RATIO_WH else _ratio_from_wh(w, h)
        if resolved_ratio == "adaptive":
            resolved_ratio = _ratio_from_wh(w, h)
        resolved_res = resolution if resolution in _RES_SHORT_SIDE else _resolution_from_wh(w, h)
        return resolved_ratio, resolved_res, f"{w}x{h}", wh

    if ratio == "adaptive":
        return "adaptive", resolution, None, None

    size = _canonical_size(ratio, resolution)
    return ratio, resolution, size, (_parse_wh(size) if size else None)


# ------------------------------------------------------- tolerant JSON probing

def _find_job_id(payload: Any) -> Optional[str]:
    """Best-effort lookup of a video job/task id in a response payload."""
    for key in ("id", "video_id", "videoId", "task_id", "taskId", "request_id", "requestId", "job_id"):
        value = _deep_get(payload, key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return None


def _deep_get(obj: Any, key: str) -> Any:
    """Recursively look up ``key`` in nested dicts/lists (breadth-first)."""
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for value in obj.values():
            found = _deep_get(value, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _deep_get(item, key)
            if found is not None:
                return found
    return None


def _looks_like_video_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value.lower().startswith(("http://", "https://")):
        return None
    # Accept anything that could be a media link; skip obvious JSON/API endpoints.
    if "/videos/" in value and value.rstrip("/").endswith(("retrieve", "content")):
        return value
    return value


def _find_video_url(payload: Any) -> Optional[str]:
    """Find a download URL for the generated video in a payload."""
    for key in ("video_url", "videoUrl", "download_url", "downloadUrl", "url", "src", "file_url", "fileUrl", "output"):
        value = _deep_get(payload, key)
        if isinstance(value, list):
            value = value[0] if value else None
        url = _looks_like_video_url(value)
        if url:
            return url
    # Some providers return {"data": [{"url": ...}]} with a different key casing.
    return None


def _find_embedded_b64(payload: Any) -> Optional[bytes]:
    """Some proxies return the video inline as base64 instead of a URL."""
    for key in ("b64_json", "video_base64", "base64", "data_base64"):
        value = _deep_get(payload, key)
        if isinstance(value, str) and len(value) > 1024:
            if value.startswith("data:"):
                value = value.split(",", 1)[-1]
            try:
                return base64.b64decode(value)
            except Exception:
                continue
    return None


def _error_message(payload: Any) -> str:
    """Extract a human-readable error message from an API payload."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or err.get("detail")
            if msg:
                return str(msg)
        elif isinstance(err, str) and err:
            return err
        for key in ("message", "msg", "detail", "error_message"):
            if isinstance(payload.get(key), str) and payload[key]:
                return payload[key]
    return ""


def _status_of(payload: Any) -> str:
    status = _deep_get(payload, "status") or _deep_get(payload, "state") or _deep_get(payload, "task_status")
    return str(status).strip().lower() if status is not None else ""


def _progress_of(payload: Any) -> Optional[int]:
    value = _deep_get(payload, "progress")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _poll_delay(response: Optional[requests.Response], poll_interval: int) -> float:
    """Sleep interval between polls, honouring the server's poll hint.

    The OpenAI SDK reads an ``openai-poll-after-ms`` header to adapt its polling
    cadence; we do the same but never wait longer than the user's poll_interval.
    """
    try:
        hint_ms = int((response.headers or {}).get("openai-poll-after-ms", 0))
    except (TypeError, ValueError):
        hint_ms = 0
    if hint_ms > 0:
        return max(0.5, min(float(poll_interval), hint_ms / 1000.0))
    return float(poll_interval)


# --------------------------------------------------------------- HTTP helpers

def _http_json(method: str, url: str, *, api_key: str, debug: bool, timeout: int = 120,
               json_body: Optional[dict] = None, data: Optional[dict] = None,
               files: Optional[dict] = None, retries: int = 3) -> Tuple[int, Any, requests.Response]:
    """Perform an HTTP request and return ``(status, json_or_text, response)``.

    Retries transient failures with exponential backoff. Never raises for HTTP
    error codes: callers decide what a non-2xx means.
    """
    headers = _auth_headers(api_key)
    delay = 2.0
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=json_body,
                data=data,
                files=files,
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if debug:
                debug_log(f"[video] {method} {url} raised {type(exc).__name__}: {exc} (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc

        if response.status_code in _RETRYABLE_STATUS and attempt < retries:
            if debug:
                debug_log(f"[video] {method} {url} -> HTTP {response.status_code}, retrying (attempt {attempt}/{retries})")
            time.sleep(delay)
            delay *= 2
            continue

        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        return response.status_code, payload, response

    raise RuntimeError(f"Request to {url} failed after {retries} attempts: {last_exc}")


def _image_to_png(image, target_wh: Optional[Tuple[int, int]], debug: bool,
                  label: str = "reference", max_side: int = 6000) -> Optional[bytes]:
    """Convert a ComfyUI IMAGE tensor into PNG bytes.

    ``target_wh`` forces an exact frame size (needed for first/last frames that
    must match the output pixel pair). Reference images are NOT resized to the
    output size — they are subjects, not frames — but very large ones are
    downscaled to stay inside Seedance's 6000px-per-side limit.
    """
    if image is None:
        return None
    from PIL import Image as _Image

    tensor = image
    if getattr(tensor, "ndim", 3) == 4:
        tensor = tensor[0]
    if getattr(tensor, "ndim", 3) != 3:
        raise ValueError(f"image input must be (H,W,C) or (B,H,W,C), got shape {tuple(tensor.shape)}")

    pil_img = tensor_to_pil(tensor)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if target_wh and pil_img.size != target_wh:
        if debug:
            debug_log(f"[video] {label}: resizing {pil_img.size} -> {target_wh} to match the output frame.")
        pil_img = pil_img.resize(target_wh, _Image.Resampling.LANCZOS)
    elif max(pil_img.size) > max_side:
        scale = max_side / float(max(pil_img.size))
        new_size = (max(2, int(pil_img.size[0] * scale) // 2 * 2), max(2, int(pil_img.size[1] * scale) // 2 * 2))
        if debug:
            debug_log(f"[video] {label}: downscaling {pil_img.size} -> {new_size} (max side {max_side}).")
        pil_img = pil_img.resize(new_size, _Image.Resampling.LANCZOS)

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    if debug:
        debug_log(f"[video] {label}: encoded {pil_img.size}, {buf.getbuffer().nbytes} bytes.")
    return buf.getvalue()


# Seedance content roles (Volcengine Ark / tu-zi gateway).
_ROLE_FIRST = "first_frame"
_ROLE_LAST = "last_frame"
_ROLE_REF = "reference_image"

# `refer_model` values understood by the gateway.
_REFER_MODES = ["auto", "first_frame", "first_last_frame", "reference"]

_REFER_TEXT = "textToVideo"
_REFER_IMAGE = "imageToVideo"
_REFER_FIRST_LAST = "firstAndLastFrame"
_REFER_REFERENCE = "referToVideo"


def _plan_media(images, reference_mode: str, video_ref: str, audio_ref: str,
                target_wh: Optional[Tuple[int, int]], caps: dict,
                debug: bool) -> Tuple[str, list, Optional[str]]:
    """Decide ``refer_model``, the ``content[]`` media items and any ratio override.

    Seedance 2.x semantics (gateway docs):
      * no media                  -> textToVideo
      * exactly 1 image           -> imageToVideo  (role first_frame)
      * exactly 2 images, FLF     -> firstAndLastFrame (first_frame / last_frame)
      * 2+ images or extra media  -> referToVideo  (role reference_image)
      * exactly 1 video (2.5)     -> videoExtend (ratio=adaptive)
    Seedance 2.5 additionally requires ``ratio=adaptive`` for imageToVideo and
    firstAndLastFrame, which is returned as the ratio override.
    """
    present = [img for img in images if img is not None]
    mode = (reference_mode or "auto").strip().lower()
    if mode not in _REFER_MODES:
        mode = "auto"
    video_url = (video_ref or "").strip() or None
    audio_url = (audio_ref or "").strip() or None

    media: list = []
    ratio_override: Optional[str] = None

    if mode == "auto":
        if present and len(present) == 1 and not video_url and not audio_url:
            mode = "first_frame"
        elif present or video_url or audio_url:
            mode = "reference"
        else:
            mode = "text"

    if video_url and mode in ("auto", "reference"):
        if not present and not audio_url and caps["video_extend"]:
            # Seedance 2.5 video extension: exactly one reference video.
            ratio_override = "adaptive"
            media.append(("video", "reference_video", video_url))
            return "videoExtend", media, ratio_override
        media.append(("video", "reference_video", video_url))
    if audio_url:
        media.append(("audio", "reference_audio", audio_url))

    if not present:
        if media:
            return "referToVideo", media, ratio_override
        return "textToVideo", [], ratio_override

    # Seedance 2.5 forces ratio=adaptive for frame-driven runs and derives the
    # output shape from the frame; resizing the first frame to a fixed ratio
    # would silently crop it, so leave the frame untouched in that case.
    frame_target = None if caps["frames_need_adaptive"] else target_wh

    if mode == "first_frame":
        if len(present) > 1 and debug:
            debug_log(f"[video] first_frame mode ignores {len(present) - 1} extra image(s).")
        media.insert(0, ("image", _ROLE_FIRST, _image_to_png(present[0], frame_target, debug, "first frame")))
    elif mode == "first_last_frame":
        if len(present) == 1:
            media.insert(0, ("image", _ROLE_FIRST, _image_to_png(present[0], frame_target, debug, "first frame")))
        else:
            if len(present) > 2 and debug:
                debug_log(f"[video] first_last_frame mode ignores {len(present) - 2} extra image(s).")
            media.insert(0, ("image", _ROLE_LAST, _image_to_png(present[1], target_wh, debug, "last frame")))
            media.insert(0, ("image", _ROLE_FIRST, _image_to_png(present[0], target_wh, debug, "first frame")))
    else:  # reference
        for index, img in enumerate(present):
            media.append(("image", _ROLE_REF, _image_to_png(img, None, debug, f"reference {index + 1}")))

    image_roles = [role for kind, role, _ in media if kind == "image"]
    if "reference_image" in image_roles or video_url or audio_url:
        refer_model = "referToVideo"
    elif image_roles == [_ROLE_FIRST, _ROLE_LAST]:
        refer_model = "firstAndLastFrame"
    elif image_roles == [_ROLE_FIRST]:
        refer_model = "imageToVideo"
    else:
        refer_model = "referToVideo"

    if caps["frames_need_adaptive"] and refer_model in ("imageToVideo", "firstAndLastFrame"):
        ratio_override = "adaptive"

    # Keep images first so a prompt like "the first image is the character"
    # still lines up with content order.
    media = [item for item in media if item[0] == "image"] + [
        item for item in media if item[0] != "image"
    ]
    return refer_model, media, ratio_override


# ------------------------------------------------------------- core pipeline

def _create_job(prov, model: str, prompt: str, ratio: str, resolution: str,
                size: Optional[str], duration: int, generate_audio: Optional[bool],
                watermark: Optional[bool], media_items: Optional[list],
                refer_model: Optional[str], output_format: Optional[str],
                debug: bool, legacy: bool = False) -> Tuple[int, Any]:
    """POST /v1/videos. Returns ``(status_code, payload)``.

    Flat generation parameters work for Seedance-style gateways (they read
    ``ratio``/``resolution``/``duration``) and OpenAI/Sora-style channels (they
    read ``prompt``/``size``/``seconds`` and ignore the rest). Unknown-field
    rejections fall back to a minimal OpenAI body.
    """
    url = _videos_url(prov.base_url)
    fields: Dict[str, Any] = {"model": model}
    if prompt:
        fields["prompt"] = prompt
    if ratio and ratio != "adaptive":
        fields["ratio"] = ratio
    elif ratio == "adaptive":
        fields["ratio"] = "adaptive"
    if resolution:
        fields["resolution"] = resolution
    if size and legacy:
        fields["size"] = size
    if duration:
        # Seedance-style: integer `duration`.
        fields["duration"] = int(duration)
        if legacy:
            # OpenAI/Sora-style: string `seconds`.
            fields["seconds"] = str(int(duration))
    if generate_audio is not None:
        fields["generate_audio"] = bool(generate_audio)
    if watermark is not None:
        fields["watermark"] = bool(watermark)
    if refer_model:
        fields["refer_model"] = refer_model
    if output_format:
        fields["output_format"] = output_format

    request_summary = dict(fields)
    if media_items:
        request_summary["content"] = [
            f"<{role or kind}: {len(payload) if isinstance(payload, bytes) else payload[:80]}>"
            for kind, role, payload in media_items
        ]
    log_api_interaction("Video Create Request", {"url": url, **request_summary}, debug)

    status, payload = 0, None
    if media_items:
        # Seedance-native shape: prompt plus every reference medium as a
        # `content[]` item carrying an explicit role (first_frame / last_frame /
        # reference_image / reference_video / reference_audio).
        content = []
        if prompt:
            content.append({"type": "text", "text": prompt})
        for kind, role, payload_item in media_items:
            if kind == "image":
                item = {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(payload_item).decode("ascii")},
                }
            elif kind == "video":
                item = {"type": "video_url", "video_url": {"url": payload_item}}
            else:
                item = {"type": "audio_url", "audio_url": {"url": payload_item}}
            if role:
                item["role"] = role
            content.append(item)

        native = {k: v for k, v in fields.items() if k not in ("prompt", "size", "seconds")}
        native.pop("input_reference", None)
        native["content"] = content
        attempts = [native]

        # Legacy single-image shapes: top-level `input_reference` (JSON data URI
        # / multipart file), used by OpenAI/Sora-style channels.
        image_only = [item for item in media_items if item[0] == "image"]
        if len(media_items) == 1 and len(image_only) == 1:
            only_png = image_only[0][2]
            single = dict(fields)
            single["input_reference"] = "data:image/png;base64," + base64.b64encode(only_png).decode("ascii")
            attempts.append(single)

        for index, body in enumerate(attempts):
            status, payload, _ = _http_json("POST", url, api_key=prov.api_key, debug=debug, timeout=180, json_body=body)
            if status < 400:
                return status, payload
            if status in (401, 403, 404, 429):
                return status, payload
            if debug:
                debug_log(f"[video] Media create rejected (HTTP {status}) on attempt {index + 1}.")

        if len(media_items) == 1 and len(image_only) == 1:
            # OpenAI-style multipart upload (`input_reference` as a file part).
            return _http_json(
                "POST", url, api_key=prov.api_key, debug=debug, timeout=180,
                data={k: v for k, v in fields.items() if k != "content"},
                files={"input_reference": ("first_frame.png", image_only[0][2], "image/png")},
            )[:2]
        raise RuntimeError(
            f"Video API rejected the media reference ({len(media_items)} item(s)); "
            f"last HTTP {status}: {_error_message(payload) or payload}"
        )

    # No file: full body first, then progressively simpler / more native bodies
    # so both modern (ratio/resolution) and legacy (size/seconds) gateways work.
    attempts = [fields]
    if resolution and resolution.lower() != resolution.upper():
        upper = dict(fields)
        upper["resolution"] = resolution.upper()
        attempts.append(upper)
    native_text = {k: v for k, v in fields.items() if k not in ("prompt", "size", "seconds", "count")}
    if prompt:
        native_text["content"] = [{"type": "text", "text": prompt}]
    attempts.append(native_text)
    minimal = {"model": model}
    if prompt:
        minimal["prompt"] = prompt
    if size:
        minimal["size"] = size
    if duration:
        minimal["seconds"] = str(int(duration))
    attempts.append(minimal)

    tried = set()
    for body in attempts:
        key = tuple(sorted((k, str(v)) for k, v in body.items()))
        if key in tried:
            continue
        tried.add(key)
        status, payload, _ = _http_json("POST", url, api_key=prov.api_key, debug=debug, timeout=180, json_body=body)
        if status < 400:
            return status, payload
        if status in (401, 403, 404, 429):
            return status, payload
        if debug:
            debug_log(f"[video] JSON create rejected (HTTP {status}); trying the next body shape.")

    # Last resort: form-data (some gateways only parse multipart).
    status, payload, _ = _http_json("POST", url, api_key=prov.api_key, debug=debug, timeout=180, data=fields)
    return status, payload


def _poll_job(prov, job_id: str, *, poll_interval: int, deadline: float, debug: bool) -> Any:
    """Poll GET /v1/videos/{id} until terminal. Returns the final payload."""
    url = f"{_videos_url(prov.base_url)}/{job_id}"
    last_payload: Any = None
    while time.time() < deadline:
        status, payload, response = _http_json("GET", url, api_key=prov.api_key, debug=debug, timeout=60)
        if status >= 400:
            message = _error_message(payload) or f"HTTP {status}"
            if status in (401, 403):
                raise RuntimeError(f"Video job {job_id} is not accessible: {message}")
            if debug:
                debug_log(f"[video] Poll {job_id} -> HTTP {status}: {message}")
            time.sleep(_poll_delay(response, poll_interval))
            continue

        last_payload = payload
        state = _status_of(payload)
        progress = _progress_of(payload)
        if debug:
            debug_log(f"[video] Poll {job_id}: status={state or '?'} progress={progress if progress is not None else '?'}")

        if state in _DONE_STATES:
            return payload
        if state in _FAILED_STATES:
            raise RuntimeError(f"Video generation failed: {_error_message(payload) or state}")
        if state and state not in _RUNNING_STATES:
            # Unknown state: if a URL is already present, treat it as done.
            if _find_video_url(payload) or _find_embedded_b64(payload):
                return payload
            if debug:
                debug_log(f"[video] Unknown status '{state}', continuing to poll.")
        time.sleep(_poll_delay(response, poll_interval))

    raise TimeoutError(
        f"Video job {job_id} did not finish before the timeout. "
        f"Last status: {_status_of(last_payload) or 'unknown'}. "
        "Increase 'timeout' or check the provider dashboard."
    )


def _download_bytes(url: str, api_key: str, *, base_url: str, debug: bool, deadline: float,
                    poll_interval: int) -> bytes:
    """Download the finished video, retrying while the file is not ready yet.

    The Authorization header is only attached when the URL points at the
    provider itself, so an API key is never leaked to a third-party CDN host.
    """
    from urllib.parse import urlparse

    same_host = urlparse(url).netloc == urlparse(base_url).netloc
    headers = _auth_headers(api_key) if same_host else {}
    headers["Accept"] = "video/mp4, application/octet-stream"
    last_status: Optional[int] = None
    while time.time() < deadline:
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=(30, 300))
        except requests.exceptions.RequestException as exc:
            if debug:
                debug_log(f"[video] Download raised {type(exc).__name__}: {exc}")
            time.sleep(poll_interval)
            continue

        last_status = response.status_code
        content_type = (response.headers.get("Content-Type") or "").lower()
        if response.status_code in (401, 403):
            # Auth failures never resolve by retrying.
            snippet = ""
            try:
                snippet = response.text[:300]
            except Exception:
                pass
            response.close()
            raise RuntimeError(f"Video download rejected (HTTP {response.status_code}): {snippet}")
        if response.status_code >= 400 or "application/json" in content_type or "text/html" in content_type:
            snippet = ""
            try:
                snippet = response.text[:300]
            except Exception:
                pass
            if debug:
                debug_log(f"[video] Download not ready (HTTP {response.status_code}, {content_type}): {snippet}")
            response.close()
            time.sleep(poll_interval)
            continue

        data = response.content
        response.close()
        if len(data) < 1024:
            if debug:
                debug_log(f"[video] Download returned only {len(data)} bytes, retrying.")
            time.sleep(poll_interval)
            continue
        if debug:
            debug_log(f"[video] Downloaded {len(data)} bytes ({content_type or 'unknown type'}).")
        return data

    raise TimeoutError(f"Could not download the generated video before the timeout (last HTTP {last_status}).")


def _generate_video(prov, model: str, prompt: str, ratio: str, resolution: str,
                    size: Optional[str], duration: int, generate_audio: Optional[bool],
                    watermark: Optional[bool], media_items: Optional[list],
                    refer_model: Optional[str], output_format: Optional[str], *,
                    poll_interval: int, timeout: int, debug: bool,
                    legacy: bool = False) -> bytes:
    """Full create -> poll -> download flow. Returns raw MP4 bytes."""
    deadline = time.time() + max(int(timeout), poll_interval)
    status, payload = _create_job(
        prov, model, prompt, ratio, resolution, size, duration,
        generate_audio, watermark, media_items, refer_model, output_format,
        debug, legacy=legacy,
    )
    log_api_interaction("Video Create Response", payload, debug)

    if status >= 400:
        message = _error_message(payload) or f"HTTP {status}"
        raise RuntimeError(f"Video creation request failed (HTTP {status}): {message}")

    # Case 1: the response already carries the video payload.
    embedded = _find_embedded_b64(payload)
    if embedded:
        return embedded
    direct_url = _find_video_url(payload)
    create_state = _status_of(payload)
    if direct_url and (create_state in _DONE_STATES or not _find_job_id(payload)):
        if debug:
            debug_log(f"[video] Create response already contains the video URL: {direct_url[:120]}")
        return _download_bytes(direct_url, prov.api_key, base_url=prov.base_url, debug=debug,
                               deadline=deadline, poll_interval=poll_interval)

    job_id = _find_job_id(payload)
    if not job_id:
        raise RuntimeError(
            "Video API response did not contain a job id or a video URL. "
            f"Payload: {str(payload)[:500]}"
        )
    if debug:
        debug_log(f"[video] Job created: {job_id}")

    # If a job id and a URL were returned together, poll once for status anyway.
    final_payload = _poll_job(prov, job_id, poll_interval=poll_interval, deadline=deadline, debug=debug)
    log_api_interaction("Video Job Final Response", final_payload, debug)

    embedded = _find_embedded_b64(final_payload)
    if embedded:
        return embedded
    video_url = _find_video_url(final_payload)
    if video_url:
        return _download_bytes(video_url, prov.api_key, base_url=prov.base_url, debug=debug,
                               deadline=deadline, poll_interval=poll_interval)

    # Standard OpenAI: fetch the binary from /videos/{id}/content.
    content_url = f"{_videos_url(prov.base_url)}/{job_id}/content"
    if debug:
        debug_log(f"[video] Downloading content from {content_url}")
    return _download_bytes(content_url, prov.api_key, base_url=prov.base_url, debug=debug,
                           deadline=deadline, poll_interval=poll_interval)


# ---- V3 schema node ----

if _V3_AVAILABLE:

    class OpenAIVideoNode(_v3_io.ComfyNode):
        """OpenAI-compatible video generation (V3 schema).

        Text-to-video with no image attached, image-to-video (first frame) when
        an image is connected — same implicit mode switching as the image node.
        """

        @classmethod
        def define_schema(cls):
            providers = _provider_names()
            models = _model_names()
            return _v3_io.Schema(
                node_id="flexai:openai_video",
                display_name="OpenAI Video (FlexAI)",
                category="flexai",
                description=(
                    "Video generation through an OpenAI-compatible /v1/videos gateway "
                    "(Seedance 2.0/2.5 today, other model families via the legacy path). "
                    "Sends ratio + resolution + duration + generate_audio; Seedance family "
                    "rules are applied automatically (upper-case tiers, SUPER_* upscaling, "
                    "duration clamping, 2.5 auto ratio=adaptive), other models keep the "
                    "permissive parameter set. Up to 4 reference images as first_frame / "
                    "last_frame / reference_image, plus reference video (videoExtend on 2.5) "
                    "and reference audio. Polls the async job and returns a native VIDEO output."
                ),
                inputs=[
                    _v3_io.Combo.Input(
                        "provider",
                        options=providers,
                        default=providers[0],
                        tooltip="API provider (configured in .env).",
                    ),
                    _v3_io.Combo.Input(
                        "model",
                        options=models,
                        default=next(
                            (mm for mm in models if _seedance_family(mm) != _LEGACY),
                            models[0] if models else _DEFAULT_MODEL,
                        ),
                        tooltip=(
                            "Video model. Seedance 2.0 / 2.0 fast / 2.0 mini / 2.5 get the "
                            "per-family resolution, duration and ratio rules enforced; other "
                            "OpenAI-style channels fall back to permissive settings."
                        ),
                    ),
                    _v3_io.String.Input(
                        "custom_model",
                        default="",
                        multiline=False,
                        optional=True,
                        tooltip="Override the model and save it to the dropdown.",
                    ),
                    _v3_io.String.Input(
                        "prompt",
                        default="A cinematic drone shot over a misty mountain range.",
                        multiline=True,
                        optional=True,
                    ),
                    _v3_io.Image.Input("image", optional=True),
                    _v3_io.Image.Input("image_2", optional=True),
                    _v3_io.Image.Input("image_3", optional=True),
                    _v3_io.Image.Input("image_4", optional=True),
                    _v3_io.Combo.Input(
                        "reference_mode",
                        options=_REFER_MODES,
                        default="auto",
                        optional=True,
                        tooltip=(
                            "How the connected images are used. auto: 1 image -> first frame "
                            "(imageToVideo), 2+ images -> multi-subject reference "
                            "(referToVideo, role reference_image). first_frame: only slot 1 as "
                            "first frame. first_last_frame: slot 1 = first frame, slot 2 = last "
                            "frame. reference: every image is a reference subject."
                        ),
                    ),
                    _v3_io.Combo.Input(
                        "ratio",
                        options=_VIDEO_RATIOS,
                        default=_DEFAULT_RATIO,
                        optional=True,
                        tooltip=(
                            "Aspect ratio sent as `ratio` (Seedance) — the model picks the "
                            "actual pixels. `adaptive` lets the model derive it from the "
                            "reference image."
                        ),
                    ),
                    _v3_io.Combo.Input(
                        "resolution",
                        options=_VIDEO_RESOLUTIONS,
                        default=_DEFAULT_RESOLUTION,
                        optional=True,
                        tooltip=(
                            "Resolution tier sent as `resolution` (wire format is upper case, e.g. 720P). "
                            "Seedance 2.0: 480p/720p/1080p/4k + super_* (super resolution output); "
                            "2.0 fast/mini: 480p/720p + super_*; 2.5: 480p/720p/1080p. "
                            "Unsupported values are rejected with the allowed list."
                        ),
                    ),
                    _v3_io.String.Input(
                        "custom_size",
                        default="",
                        multiline=False,
                        optional=True,
                        tooltip=(
                            "Legacy override, e.g. '1792x1024'. Only needed for channels that "
                            "require a pixel `size` (Sora-style). When set, ratio/resolution are "
                            "derived from it."
                        ),
                    ),
                    _v3_io.Int.Input(
                        "duration",
                        default=_DEFAULT_DURATION,
                        min=1,
                        max=60,
                        optional=True,
                        tooltip=(
                            "Clip length in seconds, sent as integer `duration` (Seedance) and "
                            "string `seconds` (Sora-style). Seedance 2.0: 4-15s, 2.5: 4-30s; "
                            "sora-2: 4/8/12."
                        ),
                    ),
                    _v3_io.Boolean.Input(
                        "generate_audio",
                        default=True,
                        optional=True,
                        tooltip="Sent as `generate_audio` (Seedance native audio track).",
                    ),
                    _v3_io.Boolean.Input(
                        "watermark",
                        default=False,
                        optional=True,
                        tooltip="Sent as `watermark` (model-side watermark).",
                    ),
                    _v3_io.Combo.Input(
                        "output_format",
                        options=["auto", "mp4", "mov"],
                        default="auto",
                        optional=True,
                        tooltip=(
                            "Container format sent as `output_format`. Seedance 2.5 only "
                            "(mp4 or mov); 'auto' omits the field and lets the model decide."
                        ),
                    ),
                    _v3_io.String.Input(
                        "reference_video",
                        default="",
                        multiline=False,
                        optional=True,
                        tooltip=(
                            "Optional reference video: an HTTP(S) URL or asset:// id (data URIs are "
                            "not supported for video). Seedance 2.5 alone also accepts it as "
                            "videoExtend input (exactly one clip, ratio becomes adaptive). "
                            "Seedance 2.0/2.5 accept it as a multimodal reference."
                        ),
                    ),
                    _v3_io.String.Input(
                        "reference_audio",
                        default="",
                        multiline=False,
                        optional=True,
                        tooltip=(
                            "Optional reference audio: HTTP(S) URL, asset:// id, or an "
                            "mp3/wav data URI. Sent as a content item with role reference_audio."
                        ),
                    ),
                    _v3_io.Int.Input(
                        "poll_interval",
                        default=5,
                        min=1,
                        max=60,
                        optional=True,
                        tooltip="Seconds between status polls (shortened if the server sends a poll hint).",
                    ),
                    _v3_io.Int.Input(
                        "timeout",
                        default=900,
                        min=30,
                        max=7200,
                        optional=True,
                        tooltip=(
                            "Overall timeout in seconds for polling + download. "
                            "Video renders often take 1-5 minutes; pro/1080p can take longer."
                        ),
                    ),
                    _v3_io.Boolean.Input("debug", default=False, optional=True),
                ],
                outputs=[_v3_io.Video.Output()],
            )

        @classmethod
        def execute(
            cls,
            provider: str,
            model: str,
            custom_model: str = "",
            prompt: str = "",
            image=None,
            image_2=None,
            image_3=None,
            image_4=None,
            reference_mode: str = "auto",
            ratio: str = _DEFAULT_RATIO,
            resolution: str = _DEFAULT_RESOLUTION,
            custom_size: str = "",
            duration: int = _DEFAULT_DURATION,
            generate_audio: bool = True,
            watermark: bool = False,
            output_format: str = "auto",
            reference_video: str = "",
            reference_audio: str = "",
            poll_interval: int = 5,
            timeout: int = 900,
            debug: bool = False,
        ):
            try:
                final_model = custom_model.strip() if custom_model and custom_model.strip() else model
                if custom_model and custom_model.strip():
                    add_model(custom_model.strip(), _MODEL_KEY)

                caps = _caps_for(final_model)
                explicit_size = (custom_size or "").strip()
                ratio_value, resolution_value, nominal_size, target_wh = _geometry(
                    (ratio or caps.get("default_ratio") or _DEFAULT_RATIO).strip(),
                    (resolution or _DEFAULT_RESOLUTION).strip(),
                    explicit_size,
                )
                # Family-specific validations (Seedance 2.0 / 2.0 fast|mini / 2.5).
                _wire_resolution(resolution_value, caps)  # raises for unsupported tiers
                wire_resolution = (
                    resolution_value
                    if _seedance_family(final_model) == _LEGACY
                    else _RESOLUTION_WIRE[resolution_value]
                )
                duration_value = _clamp_duration(duration, caps, debug)

                # `size` is a legacy OpenAI/Sora-style field. Seedance-style
                # gateways take ratio+resolution and would only see a confusing
                # extra key, so send it only when pixels were explicitly pinned
                # or the model is Sora-family.
                legacy = bool(explicit_size) or ("sora" in final_model.lower())
                send_size = nominal_size if legacy else None

                output_format_value = (output_format or "auto").strip().lower()
                if output_format_value in ("", "auto"):
                    output_format_value = None
                elif not caps["output_format"]:
                    raise ValueError(
                        f"output_format is only supported by Seedance 2.5, not {caps['label']}."
                    )

                reference_mode_value = (reference_mode or "auto").strip().lower()
                refer_model, media_items, ratio_override = _plan_media(
                    [image, image_2, image_3, image_4],
                    reference_mode_value,
                    reference_video,
                    reference_audio,
                    target_wh,
                    caps,
                    debug,
                )
                if ratio_override:
                    ratio_value = ratio_override
                if media_items and not prompt.strip():
                    raise ValueError("A prompt is required when reference media is connected.")

                prov = provider_config.get_provider_by_name(provider)
                if not prov:
                    raise ValueError(f"Provider '{provider}' not found.")
                if not prov.api_key or prov.api_key.startswith("your_key"):
                    raise ValueError("API key is not configured.")

                if debug:
                    debug_log(
                        f"[video] provider={provider} model={final_model} "
                        f"family={caps['label']} ratio={ratio_value} "
                        f"resolution={wire_resolution} size={send_size or 'not sent'} "
                        f"duration={duration_value}s audio={generate_audio} "
                        f"refer_model={refer_model} output_format={output_format_value or 'auto'} "
                        f"media={[(kind, role) for kind, role, _ in (media_items or [])] or 'none'}"
                    )

                video_bytes = _generate_video(
                    prov,
                    final_model,
                    prompt,
                    ratio_value,
                    wire_resolution,
                    send_size,
                    duration_value,
                    bool(generate_audio),
                    bool(watermark),
                    media_items,
                    refer_model,
                    output_format_value,
                    poll_interval=int(poll_interval or 5),
                    timeout=int(timeout or 900),
                    debug=debug,
                    legacy=legacy,
                )
                if not video_bytes:
                    raise RuntimeError("Video generation returned no data.")

                return _v3_io.NodeOutput(_VideoFromFile(BytesIO(video_bytes)))
            except Exception as exc:
                if debug:
                    import traceback
                    debug_log(f"[video] Error: {exc}\n{traceback.format_exc()}")
                # Video has no "error placeholder" equivalent; surface a clear error.
                raise RuntimeError(f"FlexAI video generation failed: {exc}") from exc


__all__ = ["OpenAIVideoNode"] if _V3_AVAILABLE else []
