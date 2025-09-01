"""OpenAIImageNode - 统一命名的图片生成/编辑节点.

特性:
 - 双模式运行: 生成模式 (images.generate) 和编辑模式 (images.edit)
 - 智能判断: 根据是否提供图片自动选择运行模式
 - 编辑模式: 可提交1-4张图片进行编辑处理（images.edit支持多图输入）
 - 生成模式: 纯文本提示词生成图片
 - 错误处理: 安全系统拒绝时提供友好提示，生成错误图片而非异常
 - 使用现代 OpenAI Python SDK (>=1.0)
"""
from __future__ import annotations
import os
from io import BytesIO
from typing import Optional
from PIL import Image
import base64
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

pil_to_tensor = _images_module.pil_to_tensor
tensor_to_pil = _images_module.tensor_to_pil
pil_to_base64 = _images_module.pil_to_base64
ensure_client = _openai_standard_module.ensure_client
generate_image_b64 = _openai_standard_module.generate_image_b64
chat_complete = _openai_standard_module.chat_complete
build_multimodal_messages = _openai_standard_module.build_multimodal_messages

plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(plugin_root, '.env'), override=True)


class OpenAIImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        provider_names = provider_config.get_provider_display_names() or ["default"]
        return {
            "required": {
                "provider": (provider_names, {"default": provider_names[0]}),
                "model": ("STRING", {"default": "gpt-image-1"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat in watercolor."}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "size": ("STRING", {"default": "1024x1024"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "flexai"

    def generate_image(self, provider, model, prompt, image_1=None, image_2=None, 
                      image_3=None, image_4=None, size="1024x1024", debug=False):
        """生成或编辑图片
        - 如果提供了任何图片，则使用 images.edit 编辑模式（可提交1-4张图）
        - 如果没有提供图片，则使用 images.generate 生成新图片
        """
        try:
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
            
            # 收集所有提供的图片
            input_images = [image_1, image_2, image_3, image_4]
            active_images = [img for img in input_images if img is not None]
            
            if active_images:
                # 编辑模式：使用提供的图片进行编辑
                if debug:
                    print(f"[DEBUG] 使用编辑模式 (images.edit)，处理 {len(active_images)} 张图片")
                result = self._edit_images(client, model, prompt, active_images, size, debug)
            else:
                # 生成模式：纯文本生成
                if debug:
                    print("[DEBUG] 使用生成模式 (images.generate)")
                result = self._generate_image(client, model, prompt, size, debug)
            
            return (result,)
            
        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG] 错误: {e}")
                traceback.print_exc()
            
            # 返回错误信息图片而不是抛出异常
            return self._create_error_image(str(e))
    
    def _generate_image(self, client, model, prompt, size, debug):
        """生成新图片"""
        if debug:
            print(f"[DEBUG] 调用 images.generate，prompt: {prompt}")
        b64 = generate_image_b64(client, model=model, prompt=prompt, size=size, seed=None, debug=debug)
        img = Image.open(BytesIO(base64.b64decode(b64)))
        tensor = pil_to_tensor(img)
        return tensor
    
    def _edit_images(self, client, model, prompt, input_images, size, debug):
        """编辑多张图片（1-4张）"""
        try:
            # 处理输入图片，转换为文件对象列表
            image_files = []
            for i, img_tensor in enumerate(input_images, 1):
                try:
                    if img_tensor.ndim == 4 and img_tensor.shape[0] >= 1:
                        img_tensor = img_tensor[0]
                    
                    pil_img = tensor_to_pil(img_tensor)
                    
                    # 转换为RGBA格式（某些 edit API 可能需要）
                    if pil_img.mode not in ['RGB', 'RGBA']:
                        pil_img = pil_img.convert('RGB')
                    
                    # 转换为字节流
                    img_bytes = BytesIO()
                    pil_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    image_files.append(img_bytes)
                    
                    if debug:
                        print(f"[DEBUG] 处理图片 {i}: {pil_img.size} {pil_img.mode}")
                        
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] 跳过无法处理的图片 {i}: {e}")
                    continue
            
            if not image_files:
                raise RuntimeError("没有有效的图片可以处理")
            
            # 调用 OpenAI images.edit API
            if debug:
                print(f"[DEBUG] 调用 images.edit，图片数量: {len(image_files)}, prompt: {prompt}")
            
            response = client.images.edit(
                model=model,
                image=image_files,  # 传递图片文件数组
                prompt=prompt,
                size=size,
                response_format="b64_json",
                n=1
            )
            
            if debug:
                print(f"[DEBUG] images.edit 响应成功，返回 {len(response.data)} 张图片")
            
            # 解析响应
            if response.data and len(response.data) > 0:
                b64_data = response.data[0].b64_json
                img = Image.open(BytesIO(base64.b64decode(b64_data)))
                tensor = pil_to_tensor(img)
                return tensor
            else:
                raise RuntimeError("API 返回空响应")
            
        except Exception as e:
            error_msg = f"图片编辑失败: {str(e)}"
            if "safety system" in str(e).lower():
                error_msg += "\n提示: 请尝试调整提示词或使用其他图片，避免可能违反安全策略的内容"
            if debug:
                print(f"[DEBUG] {error_msg}")
            raise RuntimeError(error_msg)
        finally:
            # 清理文件对象
            for img_file in image_files if 'image_files' in locals() else []:
                try:
                    img_file.close()
                except:
                    pass

    def _create_error_image(self, error_msg):
        """创建错误信息图片"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # 创建一个简单的错误信息图片
            img = Image.new('RGB', (512, 256), color='#ff4444')
            draw = ImageDraw.Draw(img)
            
            # 尝试使用系统字体，失败则使用默认字体
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 绘制错误信息（截断长文本）
            text = error_msg[:120] + "..." if len(error_msg) > 120 else error_msg
            
            # 简单的文本换行
            words = text.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) <= 40:  # 每行大约40个字符
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # 绘制多行文本
            y = 10
            for line in lines[:8]:  # 最多8行
                draw.text((10, y), line, fill='white', font=font)
                y += 25
            
            tensor_img = pil_to_tensor(img)
            return (tensor_img,)
            
        except Exception:
            # 如果连错误图片都无法创建，返回纯色图片
            img = Image.new('RGB', (256, 256), color='red')
            tensor_img = pil_to_tensor(img)
            return (tensor_img,)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "flexai/openai"

__all__ = ["OpenAIImageNode"]
