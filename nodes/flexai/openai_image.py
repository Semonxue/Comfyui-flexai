"""OpenAIImageNode - A unified node for image generation and editing (ComfyUI FlexAI Plugin v1.0.4).

Features:
 - Dual-mode operation: Generation (images.generate) and Editing (images.edit).
 - Smart detection: Automatically selects the mode based on image input.
 - Edit mode: Supports editing 1-4 images (leveraging multi-image capabilities).
 - Generation mode: Generates images from text prompts.
 - Error handling: Provides a user-friendly error image on failure instead of crashing.
 - Uses the modern OpenAI Python SDK (>=1.0).
 - Supports both base64 and URL response formats.
 - Enhanced debugging: Detailed API request/response logging and error analysis.
"""
from __future__ import annotations
import os
import sys
import base64
import requests
import importlib.util
from io import BytesIO
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

import provider_config

# -- Dynamically load utility modules --
# To avoid import issues in ComfyUI, load directly from file path
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

# Import functions from utility modules
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

# Load environment variables
load_dotenv(os.path.join(_PLUGIN_ROOT, '.env'), override=True)

_MODEL_KEY = "flexai_image_models"

def download_image_from_url(url: str, timeout: int = 30, debug: bool = False) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image object."""
    if debug:
        debug_log(f"Downloading image from URL: {url[:100]}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # Add verify=False to try to resolve SSLError
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


class OpenAIImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        provider_names = provider_config.get_provider_display_names() or ["default"]
        models = get_models(_MODEL_KEY)
        return {
            "required": {
                "provider": (provider_names, {"default": provider_names[0]}),
                "model": (models, {"default": models[0] if models else "dall-e-3"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "placeholder": "Enter new model (overrides selection and saves automatically)"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat in watercolor."}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "size": ("STRING", {"default": "", "placeholder": "Optional, e.g., 1024x1024"}),
                "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable compatibility mode for services like OpenRouter via chat endpoints."}),
                "streaming": ("BOOLEAN", {"default": False, "tooltip": "Enable streaming for compatibility mode."}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "flexai"

    def execute(self, provider, model, prompt, size="", compatibility_mode=False, streaming=False, debug=False, custom_model="", **kwargs):
        """Main execution function, dispatches to generation or editing based on mode."""
        try:
            # Handle empty size input
            final_size = size.strip()
            if not final_size and not compatibility_mode:
                final_size = "1024x1024"  # Default for native mode

            # Determine the final model name to use
            final_model = custom_model.strip() if custom_model and custom_model.strip() else model
            if custom_model.strip():
                add_model(custom_model.strip(), _MODEL_KEY) # If a custom model is used, save it
            
            client = self._setup_client(provider)
            images = [img for img in [kwargs.get(f"image_{i}") for i in range(1, 5)] if img is not None]

            if compatibility_mode:
                if debug: debug_log("Running in Compatibility Mode (Chat).")
                pil_images = self._run_chat_mode(client, final_model, prompt, images, final_size, streaming, debug)
            else:
                if debug: debug_log("Running in Native Mode (Image API).")
                pil_images = self._run_native_mode(client, final_model, prompt, images, final_size, debug)
            
            return (pil_to_tensor(pil_images),)

        except Exception as e:
            if debug:
                import traceback
                debug_log(f"An error occurred: {e}\n{traceback.format_exc()}")
            return (self._create_error_image(str(e)),)

    def _setup_client(self, provider_name: str):
        """Configures and returns the API client."""
        prov = provider_config.get_provider_by_name(provider_name)
        if not prov:
            raise ValueError(f"Provider '{provider_name}' not found.")
        if not prov.api_key or prov.api_key.startswith("your_key"):
            raise ValueError("API key is not configured.")
        return ensure_client(prov.api_key, prov.base_url)

    # --- Native Mode ---
    def _run_native_mode(self, client, model, prompt, images, size, debug):
        """Uses the native OpenAI images.generate or images.edit endpoints."""
        if images:
            # Edit mode
            if debug: debug_log(f"Native mode: editing {len(images)} image(s).")
            image_files = self._preprocess_images_for_edit(images, debug)
            try:
                params = {
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "n": 1,
                    "response_format": "b64_json",
                    "image_bytes": image_files[0].getbuffer().nbytes if image_files else 0
                }
                log_api_interaction("Native Image Edit Request", params, debug)

                # Note: The OpenAI SDK's images.edit 'image' parameter accepts a single file, not a list.
                # To support multi-image editing, the custom chat mode must be used.
                if len(image_files) > 1 and debug:
                    debug_log("Warning: Native edit mode only uses the first image.")
                response = client.images.edit(model=model, image=image_files[0], prompt=prompt, size=size, response_format="b64_json", n=1)
            finally:
                for f in image_files:
                    f.close()
        else:
            # Generation mode
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": 1,
                "response_format": "b64_json"
            }
            log_api_interaction("Native Image Generate Request", params, debug)
            response = client.images.generate(model=model, prompt=prompt, size=size, response_format="b64_json", n=1)
        
        return self._process_image_api_response(response, debug)

    def _preprocess_images_for_edit(self, images: List, debug: bool) -> List[BytesIO]:
        """Converts input Tensors to a list of PNG byte streams for the API call."""
        byte_streams = []
        for i, tensor in enumerate(images):
            if debug: debug_log(f"Preprocessing image {i+1}/{len(images)} for editing.")
            pil_img = tensor_to_pil(tensor[0] if tensor.ndim == 4 else tensor)
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
            
            byte_stream = BytesIO()
            pil_img.save(byte_stream, format='PNG')
            byte_stream.seek(0)
            byte_streams.append(byte_stream)
        return byte_streams

    def _process_image_api_response(self, response: Any, debug: bool) -> List[Image.Image]:
        """Processes the response from the images.generate/edit API, with multi-image support."""
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
            
        return self._data_list_to_pils(image_data_list, debug)

    # --- Compatibility (Chat) Mode ---
    def _run_chat_mode(self, client, model, prompt, images, size, streaming, debug) -> List[Image.Image]:
        """Generates or edits an image using the chat.completions endpoint."""
        messages = self._build_chat_messages(prompt, images, size, debug)
        response_data = chat_complete(client, model=model, messages=messages, stream=streaming, temperature=0.7, top_p=1.0, seed=None, include_usage=True, max_tokens=4000, debug=debug)
        
        log_api_interaction("Full Chat Mode API Response", response_data, debug)

        image_data_list = self._extract_image_from_chat_response(response_data, debug)
        if not image_data_list:
            raise ValueError("No image data found in chat response.")
            
        return self._data_list_to_pils(image_data_list, debug)

    def _build_chat_messages(self, prompt: str, images: List, size: str, debug: bool) -> List[Dict]:
        """Builds the message body for Chat mode."""
        prompt_text = prompt
        if size:
            prompt_text = f"{prompt}\n\nImage size: {size}"
        
        content = [{"type": "text", "text": prompt_text}]
        
        for i, tensor in enumerate(images):
            if debug: debug_log(f"Encoding image {i+1}/{len(images)} for chat.")
            pil_img = tensor_to_pil(tensor[0] if tensor.ndim == 4 else tensor)
            
            # Resize image to reduce token consumption
            pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            base64_url = pil_to_base64(pil_img)
            content.append({"type": "image_url", "image_url": {"url": base64_url}})
            
        return [{"role": "user", "content": content}]

    def _extract_image_from_chat_response(self, response: Dict, debug: bool) -> List[str]:
        """Robustly extracts all image data (URL or Base64) from the Chat API response."""
        if not response: return []
        
        image_data_list = []

        # First, try to extract from the `images` field (e.g., from streaming aggregation)
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

        # Second, try to extract from the aggregated content field or original response structure
        content = response.get("content", "")
        if not content:
            try:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            except (IndexError, KeyError):
                pass

        if content:
            import re
            # Match all Markdown images `![alt](url)` or direct URLs
            url_pattern = r'!\[.*?\]\((https?://[^\s"\'\)]+)\)|(https?://[^\s"\'\)]+)'
            found_urls = re.findall(url_pattern, content)
            for url_tuple in found_urls:
                url = url_tuple[0] or url_tuple[1]
                if url:
                    image_data_list.append(url)
            
            # Match all Base64 data URIs
            b64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
            found_b64 = re.findall(b64_pattern, content)
            image_data_list.extend(found_b64)

        # Deduplicate while preserving order
        unique_image_data = list(dict.fromkeys(image_data_list))
        if debug:
            debug_log(f"Found {len(unique_image_data)} unique image data items in chat response.")
        return unique_image_data

    # --- Utility Methods ---
    def _data_list_to_pils(self, data_list: List[str], debug: bool) -> List[Image.Image]:
        """Converts a list of URLs or Base64 data into a list of PIL images."""
        pil_images = []
        for i, data in enumerate(data_list):
            try:
                if debug: debug_log(f"Processing image data {i+1}/{len(data_list)}...")
                if data.startswith("http"):
                    if debug: debug_log("Data is a URL, downloading...")
                    img = download_image_from_url(data, debug=debug)
                else:
                    if debug: debug_log("Data is base64, decoding...")
                    img = Image.open(BytesIO(base64.b64decode(data)))
                pil_images.append(img)
            except Exception as e:
                if debug:
                    debug_log(f"Failed to process image data item {i+1}: {e}")
                    # Log the data snippet that caused the failure, but avoid logging long base64 strings
                    if len(data) > 100:
                        debug_log(f"Truncated problematic data: {data[:100]}...")
                    else:
                        debug_log(f"Problematic data: {data}")
                # When invalid data is encountered, either skip or log the error
                continue
        return pil_images

    def _create_error_image(self, error_msg: str) -> Image.Image:
        """Creates an image displaying the error message."""
        img = Image.new('RGB', (512, 512), color='#300000')
        draw = ImageDraw.Draw(img)
        try:
            # Try to load a common cross-platform font
            font_path = "Arial.ttf" if sys.platform == "win32" else "/System/Library/Fonts/Helvetica.ttc"
            font = ImageFont.truetype(font_path, 18)
        except IOError:
            font = ImageFont.load_default()

        # Simple text wrapping
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
        for i, line in enumerate(lines[:20]): # Limit number of lines
            draw.text((10, 40 + i * 20), line, fill='white', font=font)
            
        return pil_to_tensor(img)
