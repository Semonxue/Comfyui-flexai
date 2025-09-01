"""OpenAITextNode - 统一命名的文本生成节点.

特点:
 - 使用现代 openai>=1.x 客户端 (OpenAI)
 - 多模态: 支持可选 0-4 张参考图 (data URL image_url)
 - 内部固定 max_tokens=4096; 精简参数(去除 legacy response_language/max_tokens)
 - 可供旧 key (flexai:gentext) 兼容, 由插件入口重复映射
"""
import os
from dotenv import load_dotenv
import provider_config
import sys
import importlib.util

# Load utils modules directly by file path to avoid import issues
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_UTILS_DIR = os.path.join(_PLUGIN_ROOT, 'nodes', 'utils')

def _load_utils_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTILS_DIR, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_images_module = _load_utils_module('images')
_openai_standard_module = _load_utils_module('openai_standard')

tensor_to_pil = _images_module.tensor_to_pil
pil_to_base64 = _images_module.pil_to_base64
ensure_client = _openai_standard_module.ensure_client
build_multimodal_messages = _openai_standard_module.build_multimodal_messages
chat_complete = _openai_standard_module.chat_complete

plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(plugin_root, '.env'), override=True)

class OpenAITextNode:
    @classmethod
    def INPUT_TYPES(cls):
        provider_names = provider_config.get_provider_display_names() or ["default"]
        return {
            "required": {
                "provider": (provider_names, {"default": provider_names[0]}),
                "model": ("STRING", {"default": "gpt-4"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": False}),
                "user_prompt": ("STRING", {"default": "Describe the following images.", "multiline": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "include_usage": ("BOOLEAN", {"default": True}),
                "stream": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Only return text content
    FUNCTION = "generate_text"
    CATEGORY = "flexai"

    def generate_text(self, provider, model, temperature=0.5, system_prompt="You are a helpful assistant.",
                      user_prompt="Describe the following images.", image_1=None, image_2=None, image_3=None,
                      image_4=None, seed=0, top_p=1.0, include_usage=True, stream=True, debug=False):
        max_tokens = 4096
        include_usage = True  # 始终请求使用量 (stream 时)
        prov = provider_config.get_provider_by_name(provider)
        if prov is None:
            provider_config.load_providers(force_reload=True)
            prov = provider_config.get_provider_by_name(provider)
        if prov is None:
            raise ValueError(f"未找到 provider: {provider}")
        api_key = prov.api_key
        base_url = prov.base_url
        if not api_key or api_key.startswith("your_key"):
            raise ValueError("API 密钥未配置或仍为占位符")

        client = ensure_client(api_key, base_url)

        # 参考图转 data url
        data_urls = []
        for ref in [image_1, image_2, image_3, image_4]:
            if ref is None:
                continue
            try:
                t = ref
                if len(t.shape) == 4 and t.shape[0] >= 1:
                    t = t[0]
                if len(t.shape) != 3:
                    continue
                pil_img = tensor_to_pil(t)
                if max(pil_img.size) > 1024:
                    ratio = 1024 / max(pil_img.size)
                    pil_img = pil_img.resize((int(pil_img.size[0]*ratio), int(pil_img.size[1]*ratio)))
                data_urls.append(pil_to_base64(pil_img, fmt="JPEG"))
            except Exception:
                if debug:
                    print("[openai_text] 忽略无法处理的参考图")
                continue

        messages = build_multimodal_messages(system_prompt, user_prompt, data_urls)
        result = chat_complete(client, model=model, messages=messages, temperature=temperature, top_p=top_p,
                               max_tokens=max_tokens, seed=seed if seed > 0 else None, stream=stream,
                               include_usage=include_usage, debug=debug)
        content = result.get("content", "")
        if not content.strip():
            raise ValueError("API返回空响应。")
        return (content,)

__all__ = ["OpenAITextNode"]
