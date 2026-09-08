"""OpenAIImageNode - A unified node for image generation and editing (ComfyUI FlexAI Plugin).

Features:
  - Dual-mode operation: Generation (images.generate) and Editing (images.edit).
  - Smart detection: Automatically selects the mode based on image input.
  - Edit mode: Supports editing 1-4 images (leveraging multi-image capabilities).
  - Generation mode: Generates images from text prompts.
  - Error handling: Provides a user-friendly error image on failure instead of crashing.
  - Uses the modern OpenAI Python SDK (>=1.0).
  - Supports both base64 and URL response formats.
  - Enhanced debugging: Detailed API request/response logging and error analysis.
  - V3 schema (io.Combo): editable size dropdown. For nano-banana / gemini-*-image-*
    the selected preset is translated into Gemini imageConfig (imageSize + aspectRatio)
    and passed via extra_body, so 2K/4K actually takes effect instead of silently
    falling back to 1K.
  - Falls back to the legacy INPUT_TYPES-based class if ComfyUI doesn't expose
    `comfy_api.latest.io`.
"""
from __future__ import annotations
import os
import sys
import base64
import re
import requests
import importlib.util
from io import BytesIO
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

import provider_config

# -- Dynamically load utility modules --
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

pil_to_tensor = _images_module.pil_to_tensor
tensor_to_pil = _images_module.tensor_to_pil
pil_to_base64 = _images_module.pil_to_base64
ensure_client = _openai_standard_module.ensure_client
chat_complete = _openai_standard_module.chat_complete
debug_log = _openai_standard_module.debug_log
_truncate_base64_in_dict = _openai_standard_module._truncate_base64_in_dict
log_api_interaction = _openai_standard_module.log_api_interaction
get_models = _model_manager_module.get_models
add_model = _model_manager_module.add_model

load_dotenv(os.path.join(_PLUGIN_ROOT, '.env'), override=True)

_MODEL_KEY = "flexai_image_models"


def download_image_from_url(url: str, timeout: int = 30, debug: bool = False) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image object."""
    if debug:
        debug_log(f"Downloading image from URL: {url[:100]}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        if debug:
            debug_log("Skipping SSL verification for image download.")
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if debug:
            debug_log(f"Image download completed: {len(response.content)} bytes.")
        return img
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to download image from {url}: {e}"
        if debug:
            debug_log(error_msg)
        raise ValueError(error_msg) from e


# ---- v3 schema import (deferred so module still loads on older ComfyUI) ----
try:
    from comfy_api.latest import io as _v3_io
    _V3_AVAILABLE = True
except ImportError:
    _v3_io = None
    _V3_AVAILABLE = False


# Models that need imageConfig (Gemini native protocol) instead of OpenAI size.
_IMAGE_CONFIG_MODELS = ("nano-banana", "gemini-3-pro-image", "gemini-2.5-flash-image")

# Size presets for the dropdown.
# Each preset is "<WxH>(<aspect-ratio>,<resolution-level>)".
# The "(...)" suffix is purely descriptive — it's stripped before parsing
# or forwarding to the API.
#
# 1K sizes are the 11 hardcoded resolutions recognized by the upstream
# billing expression (size 1K tier). 2K = 1K * 2, 4K = 1K * 4 per side.
_SIZE_PRESETS = [
    # 1:1
    "1254x1254(1:1,1K)",
    "2508x2508(1:1,2K)",
    "5016x5016(1:1,4K)",
    # 2:3 / 3:2
    "1024x1536(2:3,1K)",
    "2048x3072(2:3,2K)",
    "4096x6144(2:3,4K)",
    "1536x1024(3:2,1K)",
    "3072x2048(3:2,2K)",
    "6144x4096(3:2,4K)",
    # 3:4 / 4:3
    "1086x1448(3:4,1K)",
    "2172x2896(3:4,2K)",
    "4344x5792(3:4,4K)",
    "1448x1086(4:3,1K)",
    "2896x2172(4:3,2K)",
    "5792x4344(4:3,4K)",
    # 4:5 / 5:4
    "1122x1402(4:5,1K)",
    "2244x2804(4:5,2K)",
    "4488x5608(4:5,4K)",
    "1402x1122(5:4,1K)",
    "2804x2244(5:4,2K)",
    "5608x4488(5:4,4K)",
    # 16:9 / 9:16
    "1672x941(16:9,1K)",
    "3344x1882(16:9,2K)",
    "6688x3764(16:9,4K)",
    "941x1672(9:16,1K)",
    "1882x3344(9:16,2K)",
    "3764x6688(9:16,4K)",
    # 21:9 / 9:21
    "1915x821(21:9,1K)",
    "3830x1642(21:9,2K)",
    "7660x3284(21:9,4K)",
    "821x1915(9:21,1K)",
    "1642x3830(9:21,2K)",
    "3284x7660(9:21,4K)",
]


def _needs_image_config(model: str) -> bool:
    m = (model or "").lower()
    return any(tok in m for tok in _IMAGE_CONFIG_MODELS)


def _parse_size_to_image_config(size: str):
    """Map a `WxH` size string into {imageSize, aspectRatio}.

    Returns None if size cannot be parsed.

    Resolution level comes from the user's annotation when present
    (e.g. "1024x1024 (2K)" -> 2K, ignoring the actual pixel count). This
    matches the upstream billing behaviour where the size string is matched
    against a whitelist before falling back to a pixel-based bucket.

    For free-form input without an annotation, resolution is bucketed by
    total pixel count to match the upstream billing tiers:
        <= 1048576 px  (= 1024x1024) -> 1K
        <= 4194304 px  (= 2048x2048) -> 2K
        >  4194304 px                 -> 4K
    """
    if not size:
        return None
    raw = size.strip()
    # Extract trailing "(<annotation>)" if present.
    # Annotation format can be "<level>" (legacy, e.g. "1K") or
    # "<ratio>,<level>" (new, e.g. "1:1,1K").
    ann_match = re.search(r"\(([^)]+)\)\s*$", raw)
    annotation_raw = ann_match.group(1).strip() if ann_match else None

    # Determine resolution level from annotation.
    # New format: "(1:1,1K)" → level = "1K"
    # Legacy format: "(1K)" → level = "1K"
    resolution_level = None
    if annotation_raw:
        parts = annotation_raw.split(",")
        resolution_level = parts[-1].strip().upper()

    # Strip the annotation for further parsing.
    s = re.sub(r"\s*\([^)]*\)\s*$", "", raw).lower().replace(" ", "")

    # Pure K-spec input ("1K", "2K", "4K").
    if s in {"512", "1k", "2k", "4k"}:
        return {"imageSize": s.upper() if len(s) > 1 else s, "aspectRatio": "1:1"}

    m = re.match(r"^(\d+)[x\*:](\d+)$", s)
    if not m:
        return None

    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None

    # Honour the user's annotation if it looks like a resolution level.
    if resolution_level in {"1K", "2K", "4K"}:
        image_size = resolution_level
    else:
        # Fall back to pixel-based bucketing.
        pixels = w * h
        if pixels <= 1048576:
            image_size = "1K"
        elif pixels <= 4194304:
            image_size = "2K"
        else:
            image_size = "4K"

    ratio = w / h
    ratios = {
        "1:1": 1.0,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "3:2": 3 / 2,
        "2:3": 2 / 3,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "21:9": 21 / 9,
    }
    aspect_ratio = min(ratios.keys(), key=lambda k: abs(ratios[k] - ratio))
    return {"imageSize": image_size, "aspectRatio": aspect_ratio}


def _provider_names():
    return provider_config.get_provider_display_names() or ["default"]


def _model_names():
    return get_models(_MODEL_KEY)


# ---- Module-level helpers used by both the V3 and legacy execute paths ----

def _preprocess_images_for_edit(images, debug):
    byte_streams = []
    for i, tensor in enumerate(images):
        if debug:
            debug_log(f"Preprocessing image {i+1}/{len(images)} for editing.")
        pil_img = tensor_to_pil(tensor[0] if tensor.ndim == 4 else tensor)
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        byte_stream = BytesIO()
        pil_img.save(byte_stream, format='PNG')
        byte_stream.seek(0)
        byte_streams.append(byte_stream)
    return byte_streams


def _process_image_api_response(response, debug):
    try:
        response_dict = response.model_dump() if hasattr(response, "model_dump") else vars(response)
        log_api_interaction("Full Native Image API Response", response_dict, debug)
    except Exception as e:
        if debug:
            debug_log(f"Could not serialize native response for debug log: {e}")

    if not response.data:
        raise ValueError("API response contained no data.")

    image_data_list = []
    for item in response.data:
        image_data = item.b64_json or item.url
        if image_data:
            image_data_list.append(image_data)
            if debug:
                source = "b64_json" if item.b64_json else "url"
                debug_log(f"Received image data from '{source}'.")
        else:
            if debug:
                debug_log("Skipping item with no b64_json or url.")

    if not image_data_list:
        raise ValueError("API response did not contain any valid b64_json or a URL.")

    return _data_list_to_pils(image_data_list, debug)


def _run_native_mode(client, model, prompt, images, size, debug):
    image_config = None
    if _needs_image_config(model):
        image_config = _parse_size_to_image_config(size) if size else None
        if image_config and debug:
            debug_log(f"imageConfig for {model}: {image_config} (parsed from size='{size}')")
        elif size and debug:
            debug_log(f"Could not parse size='{size}' into imageConfig; falling back to model default.")

    if images:
        if debug:
            debug_log(f"Native mode: editing {len(images)} image(s).")
        image_files = _preprocess_images_for_edit(images, debug)
        try:
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": 1,
                "response_format": "b64_json",
                "image_bytes": image_files[0].getbuffer().nbytes if image_files else 0,
            }
            if image_config:
                params["imageConfig"] = image_config
            log_api_interaction("Native Image Edit Request", params, debug)

            if len(image_files) > 1 and debug:
                debug_log("Warning: Native edit mode only uses the first image.")

            edit_kwargs = dict(
                model=model,
                image=image_files[0],
                prompt=prompt,
                size=size,
                response_format="b64_json",
                n=1,
            )
            if image_config:
                edit_kwargs["extra_body"] = {"imageConfig": image_config}
            response = client.images.edit(**edit_kwargs)
        finally:
            for f in image_files:
                f.close()
    else:
        params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
        }
        if image_config:
            params["imageConfig"] = image_config
        log_api_interaction("Native Image Generate Request", params, debug)

        gen_kwargs = dict(
            model=model,
            prompt=prompt,
            size=size,
            response_format="b64_json",
            n=1,
        )
        if image_config:
            gen_kwargs["extra_body"] = {"imageConfig": image_config}
        response = client.images.generate(**gen_kwargs)

    return _process_image_api_response(response, debug)


def _run_chat_mode(client, model, prompt, images, size, streaming, debug):
    prompt_text = prompt or ""
    if size:
        prompt_text = f"{prompt_text}\n\nImage size: {size}"
    content = [{"type": "text", "text": prompt_text}]
    for i, tensor in enumerate(images):
        if debug:
            debug_log(f"Encoding image {i+1}/{len(images)} for chat.")
        pil_img = tensor_to_pil(tensor[0] if tensor.ndim == 4 else tensor)
        pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        base64_url = pil_to_base64(pil_img)
        content.append({"type": "image_url", "image_url": {"url": base64_url}})
    messages = [{"role": "user", "content": content}]

    response_data = chat_complete(
        client,
        model=model,
        messages=messages,
        stream=streaming,
        temperature=0.7,
        top_p=1.0,
        seed=None,
        include_usage=True,
        max_tokens=4000,
        debug=debug,
    )
    log_api_interaction("Full Chat Mode API Response", response_data, debug)

    image_data_list = _extract_image_from_chat_response(response_data, debug)
    if not image_data_list:
        raise ValueError("No image data found in chat response.")
    return _data_list_to_pils(image_data_list, debug)


def _extract_image_from_chat_response(response, debug):
    if not response:
        return []
    image_data_list = []

    if "images" in response and isinstance(response["images"], list):
        for image_item in response["images"]:
            url = None
            if isinstance(image_item, dict):
                url = image_item.get("image_url", {}).get("url") or image_item.get("url")
            elif isinstance(image_item, str):
                url = image_item
            if url:
                if "base64," in url:
                    image_data_list.append(url.split("base64,", 1)[1])
                else:
                    image_data_list.append(url)

    content = response.get("content", "")
    if not content:
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (IndexError, KeyError):
            pass

    if content:
        url_pattern = r'!\[.*?\]\((https?://[^\s"\'\)]+)\)|(https?://[^\s"\'\)]+)'
        found_urls = re.findall(url_pattern, content)
        for url_tuple in found_urls:
            url = url_tuple[0] or url_tuple[1]
            if url:
                image_data_list.append(url)
        b64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
        found_b64 = re.findall(b64_pattern, content)
        image_data_list.extend(found_b64)

    unique_image_data = list(dict.fromkeys(image_data_list))
    if debug:
        debug_log(f"Found {len(unique_image_data)} unique image data items in chat response.")
    return unique_image_data


def _data_list_to_pils(data_list, debug):
    pil_images = []
    for i, data in enumerate(data_list):
        try:
            if debug:
                debug_log(f"Processing image data {i+1}/{len(data_list)}...")
            if data.startswith("http"):
                if debug:
                    debug_log("Data is a URL, downloading...")
                img = download_image_from_url(data, debug=debug)
            else:
                if debug:
                    debug_log("Data is base64, decoding...")
                img = Image.open(BytesIO(base64.b64decode(data)))
            pil_images.append(img)
        except Exception as e:
            if debug:
                debug_log(f"Failed to process image data item {i+1}: {e}")
            continue
    return pil_images


def _create_error_image(error_msg):
    img = Image.new('RGB', (512, 512), color='#300000')
    draw = ImageDraw.Draw(img)
    try:
        font_path = "Arial.ttf" if sys.platform == "win32" else "/System/Library/Fonts/Helvetica.ttc"
        font = ImageFont.truetype(font_path, 18)
    except IOError:
        font = ImageFont.load_default()
    lines = []
    words = error_msg.split()
    line = ""
    for word in words:
        if len(line + " " + word) < 50:
            line += " " + word
        else:
            lines.append(line.strip())
            line = word
    lines.append(line.strip())
    draw.text((10, 10), "FlexAI Node Error:", fill='#ff4444', font=font)
    for i, line in enumerate(lines[:20]):
        draw.text((10, 40 + i * 20), line, fill='white', font=font)
    return pil_to_tensor(img)


# ---- V3 schema node (preferred when available) ----

if _V3_AVAILABLE:

    class OpenAIImageNode(_v3_io.ComfyNode):
        """OpenAI-compatible image generation/editing (V3 schema).

        Same node ID and display name as before (`flexai:openai_image` /
        "OpenAI Image (FlexAI)"), so existing workflows stay connected.
        Uses io.Combo so `size` renders as an editable dropdown.
        """

        @classmethod
        def define_schema(cls):
            providers = _provider_names()
            models = _model_names()
            return _v3_io.Schema(
                node_id="flexai:openai_image",
                display_name="OpenAI Image (FlexAI)",
                category="flexai",
                description=(
                    "OpenAI-compatible image generation/editing. Supports nano-banana, "
                    "gpt-image-1, dall-e-3, etc. For nano-banana / gemini-*-image-* the "
                    "selected size is translated into Gemini's imageConfig "
                    "(imageSize + aspectRatio) so 2K/4K actually takes effect."
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
                        default=models[0] if models else "dall-e-3",
                        tooltip="Image model.",
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
                        default="A cute cat in watercolor.",
                        multiline=True,
                        optional=True,
                    ),
                    _v3_io.Image.Input("image_1", optional=True),
                    _v3_io.Image.Input("image_2", optional=True),
                    _v3_io.Image.Input("image_3", optional=True),
                    _v3_io.Image.Input("image_4", optional=True),
                    _v3_io.Combo.Input(
                        "size",
                        options=_SIZE_PRESETS,
                        default="1254x1254(1:1,1K)",
                        optional=True,
                        tooltip=(
                            "Output size preset including aspect ratio. "
                            "Format: '<WxH>(<ratio>,<level>)', e.g. '1254x1254(1:1,1K)'. "
                            "For nano-banana / gemini-*-image-*, parsed into imageConfig "
                            "(imageSize + aspectRatio) automatically."
                        ),
                    ),
                    _v3_io.String.Input(
                        "custom_size",
                        default="",
                        multiline=False,
                        optional=True,
                        tooltip=(
                            "Free-form size override (takes priority over 'size'). "
                            "Examples: '3072x2048', '2048x2048', '4K', '16:9'. "
                            "Leave empty to use the 'size' preset."
                        ),
                    ),
                    _v3_io.Boolean.Input(
                        "compatibility_mode",
                        default=False,
                        optional=True,
                        tooltip=(
                            "Use the chat completions endpoint for OpenRouter-like "
                            "providers. Auto-disabled for nano-banana / "
                            "gemini-*-image-* (chat path can't carry imageConfig)."
                        ),
                    ),
                    _v3_io.Boolean.Input(
                        "streaming",
                        default=False,
                        optional=True,
                        tooltip="Enable streaming for compatibility mode.",
                    ),
                    _v3_io.Boolean.Input("debug", default=False, optional=True),
                ],
                outputs=[_v3_io.Image.Output()],
            )

        @classmethod
        def execute(
            cls,
            provider: str,
            model: str,
            custom_model: str = "",
            prompt: str = "",
            image_1=None,
            image_2=None,
            image_3=None,
            image_4=None,
            size: str = "1254x1254(1:1,1K)",
            custom_size: str = "",
            compatibility_mode: bool = False,
            streaming: bool = False,
            debug: bool = False,
        ):
            try:
                # custom_size (free-form) takes priority over the size preset.
                # Strip the "(...)" annotation from the preset for API forwarding.
                preset = (size or "").strip()
                preset_clean = re.sub(r"\s*\([^)]*\)\s*$", "", preset)
                custom = (custom_size or "").strip()
                final_size = custom or preset_clean
                if not final_size and not compatibility_mode:
                    final_size = "1024x1024"

                final_model = custom_model.strip() if custom_model and custom_model.strip() else model
                if custom_model.strip():
                    add_model(custom_model.strip(), _MODEL_KEY)

                prov = provider_config.get_provider_by_name(provider)
                if not prov:
                    raise ValueError(f"Provider '{provider}' not found.")
                if not prov.api_key or prov.api_key.startswith("your_key"):
                    raise ValueError("API key is not configured.")
                client = ensure_client(prov.api_key, prov.base_url)

                images = [img for img in [image_1, image_2, image_3, image_4] if img is not None]

                # nano-banana / gemini-*-image-* must use the native image API
                # so imageConfig (imageSize + aspectRatio) takes effect.
                if compatibility_mode and not _needs_image_config(final_model):
                    if debug:
                        debug_log("Running in Compatibility Mode (Chat).")
                    pil_images = _run_chat_mode(client, final_model, prompt, images, final_size, streaming, debug)
                else:
                    if compatibility_mode and debug:
                        debug_log(f"Model {final_model} uses imageConfig; routing to native mode (chat path would silently drop size).")
                    if debug:
                        debug_log("Running in Native Mode (Image API).")
                    pil_images = _run_native_mode(client, final_model, prompt, images, final_size, debug)

                return _v3_io.NodeOutput(pil_to_tensor(pil_images))
            except Exception as e:
                if debug:
                    import traceback
                    debug_log(f"An error occurred: {e}\n{traceback.format_exc()}")
                return _v3_io.NodeOutput(_create_error_image(str(e)))

else:

    class OpenAIImageNode:
        """Legacy V1 node (kept for ComfyUI without comfy_api.latest).

        Same node ID and display name as the V3 version, so swapping back
        and forth keeps existing workflows connected.
        """

        @classmethod
        def INPUT_TYPES(cls):
            provider_names = _provider_names()
            models = _model_names()
            return {
                "required": {
                    "provider": (provider_names, {"default": provider_names[0]}),
                    "model": (models, {"default": models[0] if models else "dall-e-3"}),
                },
                "optional": {
                    "custom_model": ("STRING", {"default": "", "placeholder": "Enter new model (overrides selection and saves automatically)"}),
                    "prompt": ("STRING", {"multiline": True, "default": "A cute cat in watercolor."}),
                    "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                    "size": ("STRING", {
                        "default": "1254x1254(1:1,1K)",
                        "multiline": False,
                        "placeholder": "1254x1254(1:1,1K), 1024x1536(2:3,1K), 2K, 4K, 16:9, ...",
                    }),
                    "custom_size": ("STRING", {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional free-form override, e.g. 3072x2048, 16:9, 4K",
                    }),
                    "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "Ignored for nano-banana / gemini-*-image-* (chat path can't carry imageConfig)."}),
                    "streaming": ("BOOLEAN", {"default": False, "tooltip": "Enable streaming for compatibility mode."}),
                    "debug": ("BOOLEAN", {"default": False}),
                }
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "execute"
        CATEGORY = "flexai"

        def execute(self, provider, model, prompt, size="1254x1254(1:1,1K)", custom_size="", compatibility_mode=False, streaming=False, debug=False, custom_model="", **kwargs):
            try:
                preset = (size or "").strip()
                preset_clean = re.sub(r"\s*\([^)]*\)\s*$", "", preset)
                custom = (custom_size or "").strip()
                final_size = custom or preset_clean
                if not final_size and not compatibility_mode:
                    final_size = "1024x1024"

                final_model = custom_model.strip() if custom_model and custom_model.strip() else model
                if custom_model.strip():
                    add_model(custom_model.strip(), _MODEL_KEY)

                prov = provider_config.get_provider_by_name(provider)
                if not prov:
                    raise ValueError(f"Provider '{provider}' not found.")
                if not prov.api_key or prov.api_key.startswith("your_key"):
                    raise ValueError("API key is not configured.")
                client = ensure_client(prov.api_key, prov.base_url)
                images = [img for img in [kwargs.get(f"image_{i}") for i in range(1, 5)] if img is not None]

                if compatibility_mode and not _needs_image_config(final_model):
                    if debug:
                        debug_log("Running in Compatibility Mode (Chat).")
                    pil_images = _run_chat_mode(client, final_model, prompt, images, final_size, streaming, debug)
                else:
                    if compatibility_mode and debug:
                        debug_log(f"Model {final_model} uses imageConfig; routing to native mode (chat path would silently drop size).")
                    if debug:
                        debug_log("Running in Native Mode (Image API).")
                    pil_images = _run_native_mode(client, final_model, prompt, images, final_size, debug)

                return (pil_to_tensor(pil_images),)
            except Exception as e:
                if debug:
                    import traceback
                    debug_log(f"An error occurred: {e}\n{traceback.format_exc()}")
                return (self._create_error_image_legacy(str(e)),)

        def _create_error_image_legacy(self, error_msg):
            return _create_error_image(error_msg)