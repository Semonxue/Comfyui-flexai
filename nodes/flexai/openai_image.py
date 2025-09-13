"""OpenAIImageNode - 统一命名的图片生成/编辑节点 (ComfyUI FlexAI Plugin v1.0.4).

特性:
 - 双模式运行: 生成模式 (images.generate) 和编辑模式 (images.edit)
 - 智能判断: 根据是否提供图片自动选择运行模式
 - 编辑模式: 可提交1-4张图片进行编辑处理（images.edit支持多图输入）
 - 生成模式: 纯文本提示词生成图片
 - 错误处理: 安全系统拒绝时提供友好提示，生成错误图片而非异常
 - 使用现代 OpenAI Python SDK (>=1.0)
 - 支持base64和URL两种响应格式
 - 增强调试: 详细API请求响应日志和完整错误分析
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

# -- 动态加载工具模块 --
# 为了避免在ComfyUI中出现导入问题，直接通过文件路径加载
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

# 从工具模块导入函数
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

# 加载环境变量
load_dotenv(os.path.join(_PLUGIN_ROOT, '.env'), override=True)

_MODEL_KEY = "flexai_image_models"

def download_image_from_url(url: str, timeout: int = 30, debug: bool = False) -> Image.Image:
    """从URL下载图片并返回PIL Image对象"""
    if debug:
        debug_log(f"Downloading image from URL: {url[:100]}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # 增加 verify=False 尝试解决 SSLError
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
                "custom_model": ("STRING", {"default": "", "placeholder": "输入新模型(会覆盖上方选择并自动保存)"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat in watercolor."}),
                "image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "image_4": ("IMAGE",),
                "size": ("STRING", {"default": "1024x1024"}),
                "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable compatibility mode for services like OpenRouter via chat endpoints."}),
                "streaming": ("BOOLEAN", {"default": False, "tooltip": "Enable streaming for compatibility mode."}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "flexai"

    def execute(self, provider, model, prompt, size="1024x1024", compatibility_mode=False, streaming=False, debug=False, custom_model="", **kwargs):
        """主执行函数，根据模式调度图片生成或编辑"""
        try:
            # 确定最终使用的模型名称
            final_model = custom_model.strip() if custom_model and custom_model.strip() else model
            if custom_model.strip():
                add_model(custom_model.strip(), _MODEL_KEY) # 如果使用了自定义模型，则保存
            
            client = self._setup_client(provider)
            images = [img for img in [kwargs.get(f"image_{i}") for i in range(1, 5)] if img is not None]

            if compatibility_mode:
                if debug: debug_log("Running in Compatibility Mode (Chat).")
                pil_images = self._run_chat_mode(client, final_model, prompt, images, size, streaming, debug)
            else:
                if debug: debug_log("Running in Native Mode (Image API).")
                pil_images = self._run_native_mode(client, final_model, prompt, images, size, debug)
            
            return (pil_to_tensor(pil_images),)

        except Exception as e:
            if debug:
                import traceback
                debug_log(f"An error occurred: {e}\n{traceback.format_exc()}")
            return (self._create_error_image(str(e)),)

    def _setup_client(self, provider_name: str):
        """配置并返回API客户端"""
        prov = provider_config.get_provider_by_name(provider_name)
        if not prov:
            raise ValueError(f"Provider '{provider_name}' not found.")
        if not prov.api_key or prov.api_key.startswith("your_key"):
            raise ValueError("API key is not configured.")
        return ensure_client(prov.api_key, prov.base_url)

    # --- Native Mode ---
    def _run_native_mode(self, client, model, prompt, images, size, debug):
        """使用 OpenAI 原生 images.generate 或 images.edit 端点"""
        if images:
            # 编辑模式
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

                # 注意：OpenAI SDK的images.edit的image参数只接受单个文件，而不是列表
                # 如果需要支持多图编辑，需要依赖可以处理多图的自定义chat模式
                if len(image_files) > 1 and debug:
                    debug_log("Warning: Native edit mode only uses the first image.")
                response = client.images.edit(model=model, image=image_files[0], prompt=prompt, size=size, response_format="b64_json", n=1)
            finally:
                for f in image_files:
                    f.close()
        else:
            # 生成模式
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
        """将输入的Tensor转换为用于API调用的PNG字节流列表"""
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
        """处理来自 images.generate/edit API 的响应，支持多图"""
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
        """使用 chat.completions 端点生成或编辑图片"""
        messages = self._build_chat_messages(prompt, images, size, debug)
        response_data = chat_complete(client, model=model, messages=messages, stream=streaming, temperature=0.7, top_p=1.0, seed=None, include_usage=True, max_tokens=4000, debug=debug)
        
        log_api_interaction("Full Chat Mode API Response", response_data, debug)

        image_data_list = self._extract_image_from_chat_response(response_data, debug)
        if not image_data_list:
            raise ValueError("No image data found in chat response.")
            
        return self._data_list_to_pils(image_data_list, debug)

    def _build_chat_messages(self, prompt: str, images: List, size: str, debug: bool) -> List[Dict]:
        """为 Chat 模式构建消息体"""
        content = [{"type": "text", "text": f"{prompt}\n\nImage size requirement: {size}"}]
        
        for i, tensor in enumerate(images):
            if debug: debug_log(f"Encoding image {i+1}/{len(images)} for chat.")
            pil_img = tensor_to_pil(tensor[0] if tensor.ndim == 4 else tensor)
            
            # 调整图片大小以减少token消耗
            pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            base64_url = pil_to_base64(pil_img)
            content.append({"type": "image_url", "image_url": {"url": base64_url}})
            
        return [{"role": "user", "content": content}]

    def _extract_image_from_chat_response(self, response: Dict, debug: bool) -> List[str]:
        """从 Chat API 响应中稳健地提取所有图片数据（URL或Base64）"""
        if not response: return []
        
        image_data_list = []

        # 优先从 `images` 字段提取 (e.g., from streaming aggregation)
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

        # 其次，尝试从聚合后的 content 字段或原始响应结构中提取
        content = response.get("content", "")
        if not content:
            try:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            except (IndexError, KeyError):
                pass

        if content:
            import re
            # 匹配所有 Markdown 图片 `![alt](url)` 或直接的 URL
            url_pattern = r'!\[.*?\]\((https?://[^\s"\'\)]+)\)|(https?://[^\s"\'\)]+)'
            found_urls = re.findall(url_pattern, content)
            for url_tuple in found_urls:
                url = url_tuple[0] or url_tuple[1]
                if url:
                    image_data_list.append(url)
            
            # 匹配所有 Base64 数据URI
            b64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
            found_b64 = re.findall(b64_pattern, content)
            image_data_list.extend(found_b64)

        # 去重并保持顺序
        unique_image_data = list(dict.fromkeys(image_data_list))
        if debug:
            debug_log(f"Found {len(unique_image_data)} unique image data items in chat response.")
        return unique_image_data

    # --- Utility Methods ---
    def _data_list_to_pils(self, data_list: List[str], debug: bool) -> List[Image.Image]:
        """将URL或Base64数据列表转换为PIL图像列表"""
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
                    # 记录导致失败的数据片段，但避免记录过长的base64字符串
                    if len(data) > 100:
                        debug_log(f"Truncated problematic data: {data[:100]}...")
                    else:
                        debug_log(f"Problematic data: {data}")
                # 遇到无效数据时，可以选择跳过或记录错误
                continue
        return pil_images

    def _create_error_image(self, error_msg: str) -> Image.Image:
        """创建一个显示错误信息的图片"""
        img = Image.new('RGB', (512, 512), color='#300000')
        draw = ImageDraw.Draw(img)
        try:
            # 尝试加载一个常见的跨平台字体
            font_path = "Arial.ttf" if sys.platform == "win32" else "/System/Library/Fonts/Helvetica.ttc"
            font = ImageFont.truetype(font_path, 18)
        except IOError:
            font = ImageFont.load_default()

        # 简单的文本换行
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
        for i, line in enumerate(lines[:20]): # 限制行数
            draw.text((10, 40 + i * 20), line, fill='white', font=font)
            
        return pil_to_tensor(img)
