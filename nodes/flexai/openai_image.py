"""OpenAIImageNode - 统一命名的图片生成/编辑节点 (ComfyUI FlexAI Plugin v1.0.0).

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
from io import BytesIO
from typing import Optional
from PIL import Image
import base64
import requests
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
_truncate_base64_in_dict = _openai_standard_module._truncate_base64_in_dict

plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(plugin_root, '.env'), override=True)


def download_image_from_url(url: str, timeout: int = 30, debug: bool = False) -> Image.Image:
    """从URL下载图片并返回PIL Image对象"""
    if debug:
        print(f"[DEBUG] 🌐 开始下载图片")
        print(f"[DEBUG] 🔗 URL: {url}")
        print(f"[DEBUG] ⏰ 超时设置: {timeout} 秒")
    
    # 记录下载开始时间
    import time
    start_time = time.time()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        if debug:
            print(f"[DEBUG] 📡 发送HTTP GET请求...")
            print(f"[DEBUG] 🔧 User-Agent: {headers['User-Agent']}")
        
        response = requests.get(url, headers=headers, timeout=timeout)
        request_duration = time.time() - start_time
        
        if debug:
            print(f"[DEBUG] ✅ HTTP请求完成!")
            print(f"[DEBUG] ⏱️  请求耗时: {request_duration:.2f} 秒")
            print(f"[DEBUG] 📊 HTTP状态码: {response.status_code}")
            
        response.raise_for_status()  # 如果HTTP状态不是200会抛出异常
        
        download_duration = time.time() - start_time
        
        if debug:
            print(f"[DEBUG] ✅ 下载成功!")
            print(f"[DEBUG] ⏱️  总下载耗时: {download_duration:.2f} 秒")
            print(f"[DEBUG] 📏 下载数据大小: {len(response.content):,} 字节 ({len(response.content)/1024:.1f} KB)")
            print(f"[DEBUG] 📄 Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"[DEBUG] 🔄 开始解析图片数据...")
        
        # 直接从字节数据创建PIL Image
        parse_start = time.time()
        img = Image.open(BytesIO(response.content))
        parse_duration = time.time() - parse_start
        
        if debug:
            print(f"[DEBUG] ✅ 图片解析成功!")
            print(f"[DEBUG] ⏱️  解析耗时: {parse_duration:.2f} 秒")
            print(f"[DEBUG] 🖼️  图片信息: {img.size} 像素, {img.mode} 模式")
            
            total_duration = time.time() - start_time
            print(f"[DEBUG] 🎯 下载完成，总耗时: {total_duration:.2f} 秒")
        
        return img
        
    except requests.exceptions.Timeout:
        download_duration = time.time() - start_time
        error_msg = f"下载图片超时 (>{timeout}秒): {url}"
        if debug:
            print(f"[DEBUG] ❌ 请求超时!")
            print(f"[DEBUG] ⏱️  超时前耗时: {download_duration:.2f} 秒")
            print(f"[DEBUG] 🔍 错误: {error_msg}")
        raise ValueError(error_msg)
        
    except requests.exceptions.ConnectionError:
        download_duration = time.time() - start_time
        error_msg = f"无法连接到图片URL: {url}"
        if debug:
            print(f"[DEBUG] ❌ 连接错误!")
            print(f"[DEBUG] ⏱️  失败前耗时: {download_duration:.2f} 秒")
            print(f"[DEBUG] 🔍 错误: {error_msg}")
        raise ValueError(error_msg)
        
    except requests.exceptions.HTTPError as e:
        download_duration = time.time() - start_time
        status_code = e.response.status_code if e.response else 'unknown'
        error_msg = f"HTTP错误 {status_code}: {url}"
        if debug:
            print(f"[DEBUG] ❌ HTTP错误!")
            print(f"[DEBUG] ⏱️  失败前耗时: {download_duration:.2f} 秒")
            print(f"[DEBUG] 📊 HTTP状态码: {status_code}")
            print(f"[DEBUG] 🔍 错误: {error_msg}")
            if e.response and hasattr(e.response, 'text'):
                print(f"[DEBUG] 📄 响应内容: {e.response.text[:200]}...")
        raise ValueError(error_msg)
        
    except Exception as e:
        download_duration = time.time() - start_time
        error_msg = f"下载图片失败: {e}"
        if debug:
            print(f"[DEBUG] ❌ 未知错误!")
            print(f"[DEBUG] ⏱️  失败前耗时: {download_duration:.2f} 秒")
            print(f"[DEBUG] 📋 错误类型: {type(e).__name__}")
            print(f"[DEBUG] 🔍 错误详情: {str(e)}")
        raise ValueError(error_msg)


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
                "compatibility_mode": ("BOOLEAN", {"default": False, "tooltip": "启用兼容模式：通过chat端点实现图像生成，适用于OpenRouter等第三方服务"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "flexai"

    def generate_image(self, provider, model, prompt, image_1=None, image_2=None, 
                      image_3=None, image_4=None, size="1024x1024", compatibility_mode=False, debug=False):
        """生成或编辑图片
        - compatibility_mode=False: 使用OpenAI原生端点 (images.generate/images.edit)
        - compatibility_mode=True: 使用chat端点实现，兼容OpenRouter等第三方服务
        - 如果提供了任何图片，则根据模式选择编辑方式
        - 如果没有提供图片，则根据模式选择生成方式
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
            
            # 检查模型是否可能不支持图像生成
            vision_only_patterns = ['image-preview', 'vision', 'claude-3', 'gpt-4-vision', 'gemini-pro-vision']
            is_likely_vision_only = any(pattern in model.lower() for pattern in vision_only_patterns)
            
            if compatibility_mode:
                # 兼容模式：统一使用chat端点
                if debug:
                    if is_likely_vision_only:
                        print(f"[DEBUG] ⚠️  注意: 模型 '{model}' 可能主要用于图像理解而非生成")
                    if active_images:
                        print(f"[DEBUG] 兼容模式 - 使用chat端点进行图像编辑，处理 {len(active_images)} 张图片")
                    else:
                        print("[DEBUG] 兼容模式 - 使用chat端点进行图像生成")
                result = self._chat_mode_image(client, model, prompt, active_images, size, debug)
            else:
                # 原生模式：使用OpenAI专用端点
                if active_images:
                    # 编辑模式：使用提供的图片进行编辑
                    if debug:
                        print(f"[DEBUG] 原生模式 - 使用编辑端点 (images.edit)，处理 {len(active_images)} 张图片")
                    result = self._edit_images(client, model, prompt, active_images, size, debug)
                else:
                    # 生成模式：纯文本生成
                    if debug:
                        print("[DEBUG] 原生模式 - 使用生成端点 (images.generate)")
                    result = self._generate_image(client, model, prompt, size, debug)
            
            return (result,)
            
        except Exception as e:
            if debug:
                import traceback
                # 安全地打印错误信息，避免base64数据
                error_str = str(e)
                if len(error_str) > 500:
                    error_str = f"{error_str[:250]}... [错误信息太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                print(f"[DEBUG] 错误: {error_str}")
                traceback.print_exc()
            
            # 返回错误信息图片而不是抛出异常
            return self._create_error_image(str(e))
    
    def _generate_image(self, client, model, prompt, size, debug):
        """生成新图片"""
        if debug:
            print("=" * 60)
            print(f"[DEBUG] 🎨 开始图片生成流程")
            print(f"[DEBUG] 📝 Prompt: {prompt}")
            print(f"[DEBUG] 📐 尺寸: {size}")
            print(f"[DEBUG] 🤖 模型: {model}")
            print("=" * 60)
        
        # 记录总体开始时间
        import time
        total_start = time.time()
        
        try:
            api_start = time.time()
            b64 = generate_image_b64(client, model=model, prompt=prompt, size=size, seed=None, debug=debug)
            api_duration = time.time() - api_start
            
            if debug:
                print(f"[DEBUG] ✅ generate_image_b64 调用完成，耗时: {api_duration:.2f} 秒")
                
        except Exception as e:
            api_duration = time.time() - api_start
            if debug:
                print(f"[DEBUG] ❌ generate_image_b64 调用失败，耗时: {api_duration:.2f} 秒")
                # 安全地打印错误信息，避免base64数据
                error_str = str(e)
                if len(error_str) > 500:
                    error_str = f"{error_str[:250]}... [错误信息太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                print(f"[DEBUG] 🔍 错误: {error_str}")
            raise e
        
        # 检查base64数据是否有效
        if not b64:
            if debug:
                print("[DEBUG] ❌ generate_image_b64 返回空的base64数据")
            raise ValueError("generate_image_b64 返回空的base64数据")
            
        if not isinstance(b64, str):
            if debug:
                print(f"[DEBUG] ❌ generate_image_b64 返回的不是字符串类型: {type(b64)}")
            raise ValueError(f"generate_image_b64 返回的不是字符串类型，而是: {type(b64)}")
        
        if debug:
            print(f"[DEBUG] ✅ 收到有效的base64数据，长度: {len(b64)} 字符")
            print(f"[DEBUG] 🔄 开始解码base64数据...")
        
        try:
            decode_start = time.time()
            img = Image.open(BytesIO(base64.b64decode(b64)))
            tensor = pil_to_tensor(img)
            decode_duration = time.time() - decode_start
            
            if debug:
                print(f"[DEBUG] ✅ base64解码成功!")
                print(f"[DEBUG] ⏱️  解码耗时: {decode_duration:.2f} 秒")
                print(f"[DEBUG] 🖼️  最终图片: {img.size} {img.mode}")
                
                total_duration = time.time() - total_start
                print("=" * 60)
                print(f"[DEBUG] 🎉 图片生成流程完成!")
                print(f"[DEBUG] ⏱️  总耗时: {total_duration:.2f} 秒")
                print(f"[DEBUG]    ├─ API调用: {api_duration:.2f} 秒")
                print(f"[DEBUG]    └─ 数据解码: {decode_duration:.2f} 秒")
                print("=" * 60)
                
            return tensor
            
        except Exception as e:
            decode_duration = time.time() - decode_start
            if debug:
                print(f"[DEBUG] ❌ base64解码失败!")
                print(f"[DEBUG] ⏱️  失败前耗时: {decode_duration:.2f} 秒")
                # 安全地打印错误信息，避免base64数据
                error_str = str(e)
                if len(error_str) > 500:
                    error_str = f"{error_str[:250]}... [错误信息太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                print(f"[DEBUG] 🔍 错误: {error_str}")
                print(f"[DEBUG] 📋 base64数据前100字符: {b64[:100] if len(b64) > 100 else b64}")
            raise ValueError(f"base64图像数据解码失败: {e}")
    
    def _edit_images(self, client, model, prompt, input_images, size, debug):
        """编辑多张图片（1-4张）"""
        if debug:
            print("=" * 60)
            print(f"[DEBUG] ✏️  开始图片编辑流程")
            print(f"[DEBUG] 📝 Prompt: {prompt}")
            print(f"[DEBUG] 📐 尺寸: {size}")
            print(f"[DEBUG] 🤖 模型: {model}")
            print(f"[DEBUG] 🖼️  输入图片数量: {len(input_images)}")
            print("=" * 60)
        
        # 记录总体开始时间
        import time
        total_start = time.time()
        
        try:
            # 处理输入图片，转换为文件对象列表
            if debug:
                print(f"[DEBUG] 🔄 开始处理输入图片...")
                
            process_start = time.time()
            image_files = []
            
            for i, img_tensor in enumerate(input_images, 1):
                try:
                    if debug:
                        print(f"[DEBUG] 📷 处理第 {i} 张图片...")
                        
                    if img_tensor.ndim == 4 and img_tensor.shape[0] >= 1:
                        img_tensor = img_tensor[0]
                    
                    pil_img = tensor_to_pil(img_tensor)
                    
                    # 转换为RGBA格式（某些 edit API 可能需要）
                    if pil_img.mode not in ['RGB', 'RGBA']:
                        if debug:
                            print(f"[DEBUG]    转换颜色模式: {pil_img.mode} -> RGB")
                        pil_img = pil_img.convert('RGB')
                    
                    # 转换为字节流
                    img_bytes = BytesIO()
                    pil_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    image_files.append(img_bytes)
                    
                    if debug:
                        file_size = len(img_bytes.getvalue())
                        print(f"[DEBUG]    ✅ 第 {i} 张图片处理完成: {pil_img.size} {pil_img.mode}, 文件大小: {file_size:,} 字节")
                        
                except Exception as e:
                    if debug:
                        # 安全地打印错误信息，避免base64数据
                        error_str = str(e)
                        if len(error_str) > 200:
                            error_str = f"{error_str[:100]}... [错误太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-100:]}"
                        print(f"[DEBUG]    ❌ 跳过无法处理的第 {i} 张图片: {error_str}")
                    continue
            
            process_duration = time.time() - process_start
            
            if debug:
                print(f"[DEBUG] ✅ 图片预处理完成，耗时: {process_duration:.2f} 秒")
                print(f"[DEBUG] 📊 成功处理 {len(image_files)} 张图片")
            
            if not image_files:
                if debug:
                    print(f"[DEBUG] ❌ 没有有效的图片可以处理")
                raise RuntimeError("没有有效的图片可以处理")
            
            # 调用 OpenAI images.edit API
            if debug:
                print("=" * 60)
                print(f"[DEBUG] 🚀 开始图片编辑请求")
                print(f"[DEBUG] ⏰ 请求时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"[DEBUG] 📝 提交到OpenAI Images Edit API的参数:")
                # 打印API调用参数（不包含二进制图片数据）
                import json
                debug_params = {
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "response_format": "b64_json",
                    "n": 1,
                    "image_count": len(image_files)  # 只记录图片数量，不打印二进制数据
                }
                try:
                    truncated_params = _truncate_base64_in_dict(debug_params)
                    print(json.dumps(truncated_params, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"JSON序列化失败: {e}")
                    truncated_params = _truncate_base64_in_dict(debug_params)
                    print(f"原始参数: {truncated_params}")
                print("=" * 60)
                print("[DEBUG] 📡 正在发送API请求...")
                print("[DEBUG] ⚠️  注意: OpenAI图片编辑是同步API，需要等待完整处理后返回")
                print("[DEBUG] 💡 编辑时间通常在15-90秒之间，请耐心等待...")
            
            # 记录请求开始时间
            import time
            start_time = time.time()
            
            try:
                response = client.images.edit(
                    model=model,
                    image=image_files,  # 传递图片文件数组
                    prompt=prompt,
                    size=size,
                    response_format="b64_json",
                    n=1
                )
                request_duration = time.time() - start_time
                
                if debug:
                    print(f"[DEBUG] ✅ API请求成功完成!")
                    print(f"[DEBUG] ⏱️  总耗时: {request_duration:.2f} 秒")
                    # 安全地检查response.data是否存在
                    if response.data is not None:
                        print(f"[DEBUG] 📊 返回 {len(response.data)} 张图片")
                    else:
                        print(f"[DEBUG] ⚠️  响应中没有data字段或data为空")
                    
            except Exception as api_error:
                request_duration = time.time() - start_time
                if debug:
                    print(f"[DEBUG] ❌ API请求失败!")
                    print(f"[DEBUG] ⏱️  失败前耗时: {request_duration:.2f} 秒")
                    print(f"[DEBUG] 📋 错误类型: {type(api_error).__name__}")
                    print(f"[DEBUG] 🔍 错误详情: {str(api_error)}")
                raise api_error
            
            if debug:
                print("=" * 60)
                print("[DEBUG] 📨 分析API响应数据:")
                print(f"[DEBUG] 📊 响应对象类型: {type(response)}")
                # 打印API返回的JSON数据
                import json
                try:
                    resp_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                    if isinstance(resp_dict, dict):
                        # 打印完整的响应对象JSON体
                        print("[DEBUG] 🔍 完整响应对象JSON体:")
                        complete_resp = resp_dict.copy()
                        # 为了可读性，如果base64数据太长，截取前100和后100字符
                        if 'data' in complete_resp and isinstance(complete_resp['data'], list):
                            for i, item in enumerate(complete_resp['data']):
                                if isinstance(item, dict):
                                    complete_item = item.copy()
                                    for field in ['b64_json', 'b64', 'base64']:
                                        if field in complete_item and isinstance(complete_item[field], str) and len(complete_item[field]) > 200:
                                            b64_data = complete_item[field]
                                            complete_item[field] = f"{b64_data[:100]}...{b64_data[-100:]} [完整长度: {len(b64_data)} 字符]"
                                    complete_resp['data'][i] = complete_item
                        print(json.dumps(complete_resp, ensure_ascii=False, indent=2))
                    else:
                        truncated_dict = _truncate_base64_in_dict(resp_dict)
                        print(truncated_dict)
                except Exception as e:
                    print(f"JSON序列化失败: {e}")
                    # 安全地打印响应信息，避免base64数据
                    response_str = str(response)
                    if len(response_str) > 500:
                        response_str = f"{response_str[:200]}... [响应太长，已截断，总长度: {len(response_str)} 字符] ...{response_str[-200:]}"
                    print(f"原始响应: {response_str}")
                    # 如果JSON序列化失败，尝试打印响应对象的属性
                    if hasattr(response, '__dict__'):
                        response_dict = response.__dict__
                        truncated_dict = _truncate_base64_in_dict(response_dict)
                        print(f"[DEBUG] 响应对象属性: {truncated_dict}")
                    else:
                        available_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                        print(f"[DEBUG] 响应可用属性: {available_attrs}")
                print("=" * 60)
            
            # 解析响应
            if response.data is not None and len(response.data) > 0:
                if debug:
                    print(f"[DEBUG] 🔍 开始解析响应数据...")
                    
                first_item = response.data[0]
                
                # 尝试获取base64数据，支持不同的字段名
                b64_data = None
                found_field = None
                for attr_name in ['b64_json', 'b64', 'base64']:
                    if hasattr(first_item, attr_name):
                        field_value = getattr(first_item, attr_name)
                        if field_value:
                            b64_data = field_value
                            found_field = attr_name
                            if debug:
                                print(f"[DEBUG] ✅ 在字段 '{attr_name}' 中找到base64数据")
                                print(f"[DEBUG] 📏 数据长度: {len(b64_data)} 字符")
                                print(f"[DEBUG] 🔤 数据类型: {type(b64_data)}")
                            break
                
                if not b64_data:
                    # 检查是否有URL字段，支持URL响应
                    if hasattr(first_item, 'url') and first_item.url:
                        if debug:
                            print(f"[DEBUG] 🌐 未找到base64数据，开始从URL下载")
                            print(f"[DEBUG] 🔗 图片URL: {first_item.url}")
                        
                        try:
                            # 从URL下载图片
                            download_start = time.time()
                            img = download_image_from_url(first_item.url, debug=debug)
                            download_duration = time.time() - download_start
                            
                            if debug:
                                print(f"[DEBUG] ✅ URL下载完成，耗时: {download_duration:.2f}秒")
                                print(f"[DEBUG] 🖼️  图片信息: {img.size} {img.mode}")
                            
                            tensor = pil_to_tensor(img)
                            return tensor
                        except Exception as e:
                            if debug:
                                # 安全地打印错误信息，避免base64数据
                                error_str = str(e)
                                if len(error_str) > 500:
                                    error_str = f"{error_str[:250]}... [错误太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                                print(f"[DEBUG] ❌ URL下载失败: {error_str}")
                            raise ValueError(f"从URL下载图片失败: {e}")
                    else:
                        available_attrs = [attr for attr in dir(first_item) if not attr.startswith('_')]
                        if debug:
                            print(f"[DEBUG] ❌ 未找到有效的base64或URL数据")
                            print(f"[DEBUG] 📋 可用属性: {available_attrs}")
                        raise ValueError(f"images.edit 未返回有效的base64数据或URL，可用属性: {available_attrs}")
                
                if debug:
                    print(f"[DEBUG] 🔍 开始验证并解码base64数据...")
                    
                if not isinstance(b64_data, str):
                    if debug:
                        print(f"[DEBUG] ❌ base64数据类型错误: 期望str，实际{type(b64_data)}")
                    raise ValueError(f"base64数据类型错误，期望字符串，实际: {type(b64_data)}")
                
                try:
                    decode_start = time.time()
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                    tensor = pil_to_tensor(img)
                    decode_duration = time.time() - decode_start
                    
                    if debug:
                        print(f"[DEBUG] ✅ base64解码成功!")
                        print(f"[DEBUG] ⏱️  解码耗时: {decode_duration:.2f} 秒")
                        print(f"[DEBUG] 🖼️  生成图片信息: {img.size} {img.mode}")
                        
                        total_duration = time.time() - start_time
                        print("=" * 60)
                        print(f"[DEBUG] 🎉 图片编辑完成!")
                        print(f"[DEBUG] ⏱️  总处理时间: {total_duration:.2f} 秒")
                        print(f"[DEBUG] 📦 数据来源: {found_field}")
                        print("=" * 60)
                    
                    return tensor
                except Exception as e:
                    if debug:
                        # 安全地打印错误信息，避免base64数据
                        error_str = str(e)
                        if len(error_str) > 500:
                            error_str = f"{error_str[:250]}... [错误太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                        print(f"[DEBUG] ❌ base64解码失败: {error_str}")
                        print(f"[DEBUG] 🔍 base64数据前100字符: {b64_data[:100] if len(b64_data) > 100 else b64_data}")
                    raise ValueError(f"base64图像数据解码失败: {e}")
            else:
                if debug:
                    print(f"[DEBUG] ❌ API返回空响应或无数据")
                    print(f"[DEBUG] 🔍 响应对象类型: {type(response)}")
                    # 安全地打印response.data信息
                    if hasattr(response, 'data') and response.data:
                        if isinstance(response.data, list):
                            print(f"[DEBUG] 🔍 response.data 长度: {len(response.data)} 个元素")
                        else:
                            data_str = str(response.data)
                            if len(data_str) > 200:
                                data_str = f"{data_str[:100]}... [数据太长，已截断，总长度: {len(data_str)} 字符] ...{data_str[-100:]}"
                            print(f"[DEBUG] 🔍 response.data 值: {data_str}")
                    else:
                        print(f"[DEBUG] 🔍 response.data: 无数据或为空")
                    print(f"[DEBUG] 🔍 response.data 类型: {type(response.data) if hasattr(response, 'data') else 'No data attribute'}")
                    
                    # 尝试获取响应的所有属性
                    if hasattr(response, '__dict__'):
                        print(f"[DEBUG] 🔍 响应对象属性: {list(response.__dict__.keys())}")
                    else:
                        available_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                        print(f"[DEBUG] 🔍 响应可用属性: {available_attrs}")
                
                raise RuntimeError("API 返回空响应或无数据")
            
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
    
    def _chat_mode_image(self, client, model, prompt, input_images, size, debug):
        """兼容模式：通过chat端点实现图像生成/编辑"""
        if debug:
            print("=" * 60)
            print(f"[DEBUG] 🔄 开始兼容模式图像处理")
            print(f"[DEBUG] 📝 Prompt: {prompt}")
            print(f"[DEBUG] 📐 尺寸: {size}")
            print(f"[DEBUG] 🤖 模型: {model}")
            if input_images:
                print(f"[DEBUG] 🖼️  输入图片数量: {len(input_images)} (编辑模式)")
            else:
                print(f"[DEBUG] 🎨 纯文本生成模式")
            print("=" * 60)
        
        import time
        total_start = time.time()
        
        try:
            # 构建消息内容
            message_content = []
            
            # 添加文本内容
            if input_images:
                # 编辑模式的提示词构建
                full_prompt = f"请根据以下描述对提供的图像进行编辑或修改: {prompt}\n\n生成要求：\n- 输出尺寸: {size}\n- 请生成修改后的图像"
            else:
                # 生成模式的提示词构建
                full_prompt = f"请生成一张图像: {prompt}\n\n生成要求：\n- 输出尺寸: {size}\n- 请直接生成图像"
                
            message_content.append({
                "type": "text",
                "text": full_prompt
            })
            
            # 如果有输入图片，添加到消息中
            if input_images:
                if debug:
                    print(f"[DEBUG] 🔄 处理输入图片转换为base64...")
                    
                for i, img_tensor in enumerate(input_images, 1):
                    try:
                        if debug:
                            print(f"[DEBUG] 📷 处理第 {i} 张图片...")
                            
                        if img_tensor.ndim == 4 and img_tensor.shape[0] >= 1:
                            img_tensor = img_tensor[0]
                        
                        pil_img = tensor_to_pil(img_tensor)
                        
                        # 优化图片尺寸以减少token消耗
                        max_size = 1024
                        if max(pil_img.size) > max_size:
                            ratio = max_size / max(pil_img.size)
                            new_size = tuple(int(dim * ratio) for dim in pil_img.size)
                            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                            if debug:
                                print(f"[DEBUG]    图片已缩放至: {new_size}")
                        
                        # 转换为base64 (pil_to_base64返回完整的data URI)
                        img_base64 = pil_to_base64(pil_img)
                        
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64  # 不需要再添加前缀，pil_to_base64已经包含了
                            }
                        })
                        
                        if debug:
                            print(f"[DEBUG]    ✅ 第 {i} 张图片已转换为base64")
                            
                    except Exception as e:
                        if debug:
                            # 安全地打印错误信息，避免base64数据
                            error_str = str(e)
                            if len(error_str) > 300:
                                error_str = f"{error_str[:150]}... [错误太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-150:]}"
                            print(f"[DEBUG] ❌ 处理第 {i} 张图片失败: {error_str}")
                        continue
            
            # 构建聊天消息
            messages = [{
                "role": "user",
                "content": message_content
            }]
            
            if debug:
                import json
                print("[DEBUG] 📡 发送chat请求...")
                print("[DEBUG] 📝 请求消息结构:")
                # 创建调试用的消息副本（隐藏base64数据）
                debug_messages = []
                for msg in messages:
                    debug_msg = {"role": msg["role"]}
                    if isinstance(msg["content"], list):
                        debug_content = []
                        for item in msg["content"]:
                            if item["type"] == "text":
                                debug_content.append(item)
                            elif item["type"] == "image_url":
                                debug_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": "<base64_image_data>"}
                                })
                        debug_msg["content"] = debug_content
                    else:
                        debug_msg["content"] = msg["content"]
                    debug_messages.append(debug_msg)
                print(json.dumps(debug_messages, ensure_ascii=False, indent=2))
            
            # 调用chat API
            api_start = time.time()
            response = chat_complete(
                client, 
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=1.0,
                max_tokens=4000,
                seed=None,
                stream=False,
                include_usage=True,
                debug=debug
            )
            api_duration = time.time() - api_start
            
            if debug:
                print(f"[DEBUG] ✅ Chat API调用完成，耗时: {api_duration:.2f} 秒")
            
            # 解析响应
            if response:
                if debug:
                    content = response.get("content", "")
                    print(f"[DEBUG] 📨 收到响应内容长度: {len(content) if content else 0} 字符")
                
                # 尝试多种方式解析图像数据 - 传入原始响应或处理过的响应
                raw_response = response.get("_raw_response")
                if raw_response:
                    # 使用原始响应解析
                    try:
                        raw_dict = raw_response.model_dump() if hasattr(raw_response, 'model_dump') else None
                        if raw_dict:
                            image_data = self._extract_image_from_chat_response(raw_dict, debug)
                        else:
                            image_data = self._extract_image_from_chat_response(response, debug)
                    except:
                        image_data = self._extract_image_from_chat_response(response, debug)
                else:
                    image_data = self._extract_image_from_chat_response(response, debug)
                
                if image_data:
                    if debug:
                        print(f"[DEBUG] ✅ 成功从响应中提取图像数据")
                    
                    # 转换为tensor
                    decode_start = time.time()
                    if image_data.startswith('http'):
                        # URL格式
                        pil_img = download_image_from_url(image_data, debug=debug)
                    else:
                        # Base64格式
                        img = Image.open(BytesIO(base64.b64decode(image_data)))
                        pil_img = img
                    
                    tensor = pil_to_tensor(pil_img)
                    decode_duration = time.time() - decode_start
                    
                    total_duration = time.time() - total_start
                    if debug:
                        print("=" * 60)
                        print(f"[DEBUG] 🎉 兼容模式图像处理完成!")
                        print(f"[DEBUG] ⏱️  总耗时: {total_duration:.2f} 秒")
                        print(f"[DEBUG]    ├─ API调用: {api_duration:.2f} 秒")
                        print(f"[DEBUG]    └─ 图像解析: {decode_duration:.2f} 秒")
                        print(f"[DEBUG] 🖼️  最终图片: {pil_img.size} {pil_img.mode}")
                        print("=" * 60)
                    
                    return tensor
                else:
                    if debug:
                        print(f"[DEBUG] ❌ 未能从响应中提取图像数据")
                        print(f"[DEBUG] 📄 原始响应内容: {content[:500]}...")
                        print(f"[DEBUG] 💡 提示: 模型 '{model}' 可能不支持图像生成，仅支持图像理解")
                    
                    # 创建一个包含错误信息的图片，而不是抛出异常
                    error_msg = f"模型 '{model}' 不支持图像生成\n\n模型响应:\n{content[:200]}{'...' if len(content) > 200 else ''}"
                    return self._create_error_image(error_msg)
            else:
                if debug:
                    print(f"[DEBUG] ❌ 聊天API返回空响应")
                return self._create_error_image("聊天API返回空响应")
                
        except Exception as e:
            total_duration = time.time() - total_start
            if debug:
                print(f"[DEBUG] ❌ 兼容模式处理失败!")
                print(f"[DEBUG] ⏱️  失败前耗时: {total_duration:.2f} 秒")
                # 安全地打印错误信息，避免base64数据
                error_str = str(e)
                if len(error_str) > 500:
                    error_str = f"{error_str[:250]}... [错误信息太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                print(f"[DEBUG] 🔍 错误: {error_str}")
            raise e
    
    def _extract_image_from_chat_response(self, response, debug=False):
        """从chat响应中提取图像数据（URL或base64）"""
        if not response:
            return None
        
        import re
        import base64
        
        # 首先尝试从响应结构中直接提取图像（新格式）
        # 检查是否有 images 字段（某些API返回的格式）
        if isinstance(response, dict):
            # 尝试从不同的可能位置提取图像
            images_data = None
            
            # 检查是否有 images 字段
            if "images" in response:
                images_data = response["images"]
            # 检查是否有 choices[0].message.images
            elif "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                if "images" in message:
                    images_data = message["images"]
            
            if images_data and isinstance(images_data, list) and len(images_data) > 0:
                first_image = images_data[0]
                if isinstance(first_image, dict):
                    # 检查 image_url.url 字段
                    if "image_url" in first_image and "url" in first_image["image_url"]:
                        url = first_image["image_url"]["url"]
                        if url.startswith("data:image"):
                            # 提取base64部分
                            if "base64," in url:
                                base64_data = url.split("base64,", 1)[1]
                                if debug:
                                    print(f"[DEBUG] 🎯 从images字段找到base64数据，长度: {len(base64_data)}")
                                return base64_data
                        else:
                            if debug:
                                print(f"[DEBUG] 🔗 从images字段找到URL: {url}")
                            return url
        
        # 如果没有找到，继续从content文本中搜索
        content = ""
        if isinstance(response, dict):
            # 尝试从不同位置获取content
            if "content" in response:
                # 处理过的响应格式 (来自chat_complete函数的返回值)
                content = response["content"]
            elif "choices" in response and response["choices"]:
                # 原始OpenAI响应格式
                message = response["choices"][0].get("message", {})
                content = message.get("content", "")
        else:
            content = str(response)
            
        if not content:
            if debug:
                print(f"[DEBUG] ❌ 响应中没有content字段或images字段")
            return None
        
        # 尝试提取base64格式的图像
        base64_patterns = [
            r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)',  # 完整data URL
            r'base64,([A-Za-z0-9+/=]+)',  # 简化格式
            r'```base64\s*\n([A-Za-z0-9+/=\s]+)\n```',  # markdown代码块
            r'([A-Za-z0-9+/=]{100,})',  # 长base64字符串
        ]
        
        for pattern in base64_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 清理可能的空白字符
                clean_match = re.sub(r'\s+', '', match)
                if len(clean_match) > 100:  # 基本长度检查
                    try:
                        # 验证base64格式
                        base64.b64decode(clean_match)
                        if debug:
                            print(f"[DEBUG] 🎯 从content找到base64数据，长度: {len(clean_match)}")
                        return clean_match
                    except Exception:
                        continue
        
        # 尝试提取HTTP(S) URL，支持多种格式
        url_patterns = [
            r'!\[.*?\]\((https?://[^\s)]+)\)',  # Markdown格式: ![alt](url)
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML格式: <img src="url">
            r'https?://[^\s<>"()]+\.(?:jpg|jpeg|png|gif|webp|bmp)',  # 标准图像URL
            r'https?://[^\s<>"()]+',  # 通用HTTP(S) URL
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                url = match if isinstance(match, str) else match[0] if isinstance(match, tuple) else str(match)
                if debug:
                    print(f"[DEBUG] 🔗 从content找到URL: {url}")
                return url
        
        if debug:
            print(f"[DEBUG] ❌ 未在响应中找到图像数据")
        
        return None

    def _create_error_image(self, error_msg):
        """Create error message image with English text only"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create a simple error message image
            img = Image.new('RGB', (512, 256), color='#ff4444')
            draw = ImageDraw.Draw(img)
            
            # Try to use system font, fallback to default font
            try:
                # Try multiple font paths for better cross-platform compatibility
                font_paths = [
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "/Windows/Fonts/arial.ttf",  # Windows
                    "/Windows/Fonts/calibri.ttf"  # Windows
                ]
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, 16)
                        break
                    except:
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Convert error message to English if it contains Chinese characters
            english_error = self._translate_error_to_english(error_msg)
            
            # Truncate long text
            text = english_error[:120] + "..." if len(english_error) > 120 else english_error
            
            # Simple text wrapping
            words = text.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) <= 40:  # About 40 characters per line
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Add header
            draw.text((10, 5), "IMAGE GENERATION ERROR", fill='white', font=font)
            draw.text((10, 25), "=" * 35, fill='white', font=font)
            
            # Draw multi-line text
            y = 50
            for line in lines[:7]:  # Max 7 lines to leave space for header
                draw.text((10, y), line, fill='white', font=font)
                y += 25
            
            # Add footer with timestamp
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            draw.text((10, 230), f"Time: {timestamp}", fill='#ffcccc', font=font)
            
            tensor_img = pil_to_tensor(img)
            return (tensor_img,)
            
        except Exception:
            # If error image creation fails, return a simple red image
            img = Image.new('RGB', (256, 256), color='red')
            tensor_img = pil_to_tensor(img)
            return (tensor_img,)
    
    def _translate_error_to_english(self, error_msg):
        """Translate common Chinese error messages to English"""
        # Common Chinese to English translations
        translations = {
            "未找到 provider": "Provider not found",
            "API 密钥未配置或仍为占位符": "API key not configured or still placeholder",
            "图片生成失败": "Image generation failed",
            "图片编辑失败": "Image editing failed",
            "下载图片失败": "Image download failed",
            "无法连接到图片URL": "Unable to connect to image URL",
            "下载图片超时": "Image download timeout",
            "base64图像数据解码失败": "Base64 image data decode failed",
            "API 返回空响应或无数据": "API returned empty response or no data",
            "没有有效的图片可以处理": "No valid images to process",
            "无效的base64数据格式": "Invalid base64 data format",
            "未返回有效的base64数据或URL": "No valid base64 data or URL returned"
        }
        
        english_msg = error_msg
        for chinese, english in translations.items():
            english_msg = english_msg.replace(chinese, english)
        
        # Remove any remaining Chinese characters and replace with placeholder
        import re
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        if chinese_pattern.search(english_msg):
            english_msg = re.sub(chinese_pattern, '[Chinese text]', english_msg)
        
        return english_msg

__all__ = ["OpenAIImageNode"]
