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
debug_log = _openai_standard_module.debug_log

plugin_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(plugin_root, '.env'), override=True)


def download_image_from_url(url: str, timeout: int = 30, debug: bool = False) -> Image.Image:
    """从URL下载图片并返回PIL Image对象"""
    if debug:
        debug_log(f"Starting image download from URL: {url[:100]}...")
    
    # 记录下载开始时间
    import time
    start_time = time.time()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        request_duration = time.time() - start_time
        
        response.raise_for_status()  # 如果HTTP状态不是200会抛出异常
        
        download_duration = time.time() - start_time
        
        # 直接从字节数据创建PIL Image
        parse_start = time.time()
        img = Image.open(BytesIO(response.content))
        parse_duration = time.time() - parse_start
        
        if debug:
            total_duration = time.time() - start_time
            debug_log(f"Image download completed: {len(response.content)} bytes in {total_duration:.2f}s")
        
        return img
        
    except requests.exceptions.Timeout:
        download_duration = time.time() - start_time
        error_msg = f"下载图片超时 (>{timeout}秒): {url}"
        if debug:
            debug_log(f"Download timeout after {download_duration:.2f}s: {url[:100]}...")
        raise ValueError(error_msg)
        
    except requests.exceptions.ConnectionError:
        download_duration = time.time() - start_time
        error_msg = f"无法连接到图片URL: {url}"
        if debug:
            debug_log(f"Connection error after {download_duration:.2f}s: {url[:100]}...")
        raise ValueError(error_msg)
        
    except requests.exceptions.HTTPError as e:
        download_duration = time.time() - start_time
        status_code = e.response.status_code if e.response else 'unknown'
        error_msg = f"HTTP错误 {status_code}: {url}"
        if debug:
            debug_log(f"HTTP error {status_code} after {download_duration:.2f}s")
        raise ValueError(error_msg)
        
    except Exception as e:
        download_duration = time.time() - start_time
        error_msg = f"下载图片失败: {e}"
        if debug:
            debug_log(f"Download failed after {download_duration:.2f}s: {type(e).__name__}")
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
                "streaming": ("BOOLEAN", {"default": False, "tooltip": "启用流式输出（仅兼容模式有效）：实时接收响应数据，适用于支持streaming的chat端点"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "flexai"

    def generate_image(self, provider, model, prompt, image_1=None, image_2=None, 
                      image_3=None, image_4=None, size="1024x1024", compatibility_mode=False, streaming=False, debug=False):
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
                error_tensor = self._create_error_image(f"未找到 provider: {provider}")
                return (error_tensor,)
            api_key = prov.api_key
            base_url = prov.base_url
            if not api_key or api_key.startswith("your_key"):
                error_tensor = self._create_error_image("API 密钥未配置或仍为占位符")
                return (error_tensor,)

            client = ensure_client(api_key, base_url)
            
            # 收集所有提供的图片
            input_images = [image_1, image_2, image_3, image_4]
            active_images = [img for img in input_images if img is not None]
            
            if compatibility_mode:
                # 兼容模式：统一使用chat端点
                if debug:
                    if active_images:
                        debug_log(f"Compatibility mode - editing {len(active_images)} images")
                    else:
                        debug_log("Compatibility mode - generating image")
                    if streaming:
                        debug_log("Streaming mode enabled")
                result = self._chat_mode_image(client, model, prompt, active_images, size, streaming, debug)
            else:
                # 原生模式：使用OpenAI专用端点
                if streaming and debug:
                    debug_log("Warning: streaming not supported in native mode")
                if active_images:
                    # 编辑模式：使用提供的图片进行编辑
                    if debug:
                        debug_log(f"Native mode - editing {len(active_images)} images")
                    result = self._edit_images(client, model, prompt, active_images, size, debug)
                else:
                    # 生成模式：纯文本生成
                    if debug:
                        debug_log("Native mode - generating image")
                    result = self._generate_image(client, model, prompt, size, debug)
            
            return (result,)
            
        except Exception as e:
            if debug:
                debug_log(f"Error occurred: {str(e)[:200]}...")
            
            # 返回错误信息图片而不是抛出异常
            error_tensor = self._create_error_image(str(e))
            return (error_tensor,)
    
    def _generate_image(self, client, model, prompt, size, debug):
        """生成新图片"""
        if debug:
            debug_log(f"Starting image generation: model={model}, size={size}")
        
        # 记录总体开始时间
        import time
        total_start = time.time()
        
        try:
            api_start = time.time()
            b64 = generate_image_b64(client, model=model, prompt=prompt, size=size, seed=None, debug=debug)
            api_duration = time.time() - api_start
            
            if debug:
                debug_log(f"generate_image_b64 call completed, duration: {api_duration:.2f} seconds")
                
        except Exception as e:
            api_duration = time.time() - api_start
            if debug:
                debug_log(f"generate_image_b64 failed after {api_duration:.2f}s: {str(e)[:200]}...")
            return self._create_error_image(f"图片生成API调用失败: {str(e)}")
        
        # 检查base64数据是否有效
        if not b64:
            if debug:
                debug_log("Empty base64 data received")
            return self._create_error_image("响应中没有图片数据，请重试")
            
        if not isinstance(b64, str):
            if debug:
                debug_log(f"Invalid base64 data type: {type(b64)}")
            return self._create_error_image("响应数据格式异常，请重试")
        
        if debug:
            debug_log(f"Received base64 data: length={len(b64)}")
        
        try:
            decode_start = time.time()
            img = Image.open(BytesIO(base64.b64decode(b64)))
            tensor = pil_to_tensor(img)
            decode_duration = time.time() - decode_start
            
            if debug:
                debug_log("base64 decode successful")
                debug_log(f"Decode duration: {decode_duration:.2f} seconds")
                debug_log(f"Final image: {img.size} {img.mode}")
                
                total_duration = time.time() - total_start
                debug_log(f"Image generation completed: total={total_duration:.2f}s, api={api_duration:.2f}s, decode={decode_duration:.2f}s")
                
            return tensor
            
        except Exception as e:
            decode_duration = time.time() - decode_start
            if debug:
                debug_log(f"Base64 decode failed after {decode_duration:.2f}s: {str(e)[:200]}...")
            return self._create_error_image(f"base64图像数据解码失败: {str(e)}")
    
    def _edit_images(self, client, model, prompt, input_images, size, debug):
        """编辑多张图片（1-4张）"""
        if debug:
            debug_log(f"Starting image editing: {len(input_images)} images, size={size}")
            debug_log(f"Model: {model}")
            debug_log(f"Input images count: {len(input_images)}")
        
        # 记录总体开始时间
        import time
        total_start = time.time()
        
        try:
            # 处理输入图片，转换为文件对象列表
            if debug:
                debug_log("Starting to process input images...")
                
            process_start = time.time()
            image_files = []
            
            for i, img_tensor in enumerate(input_images, 1):
                try:
                    if img_tensor.ndim == 4 and img_tensor.shape[0] >= 1:
                        img_tensor = img_tensor[0]
                    
                    pil_img = tensor_to_pil(img_tensor)
                    
                    # 转换为RGBA格式（某些 edit API 可能需要）
                    if pil_img.mode not in ['RGB', 'RGBA']:
                        if debug:
                            debug_log(f"Converting color mode: {pil_img.mode} -> RGB")
                        pil_img = pil_img.convert('RGB')
                    
                    # 转换为字节流
                    img_bytes = BytesIO()
                    pil_img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    image_files.append(img_bytes)
                        
                except Exception as e:
                    if debug:
                        # 安全地打印错误信息，避免base64数据
                        error_str = str(e)
                        if len(error_str) > 200:
                            error_str = f"{error_str[:100]}... [error too long, truncated, total length: {len(error_str)} chars] ...{error_str[-100:]}"
                        debug_log(f"Skipping unprocessable image {i}: {error_str}")
                    continue
            
            process_duration = time.time() - process_start
            
            if debug:
                debug_log(f"Image preprocessing completed, duration: {process_duration:.2f} seconds")
                debug_log(f"Successfully processed {len(image_files)} images")
            
            if not image_files:
                if debug:
                    debug_log("No valid images to process")
                return self._create_error_image("没有有效的图片可以处理")
            
            # 调用 OpenAI images.edit API
            if debug:
                debug_log("Starting image editing request")
                debug_log(f"Request time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                debug_log("Parameters submitted to OpenAI Images Edit API:")
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
                    debug_log(f"API parameters: {json.dumps(truncated_params, ensure_ascii=False, indent=2)}")
                except Exception as e:
                    debug_log(f"Parameter serialization failed: {e}")
                
                debug_log("Sending API request (image editing may take 15-90 seconds)...")
            
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
                    debug_log("API request completed successfully!")
                    debug_log(f"Total duration: {request_duration:.2f} seconds")
                    # 安全地检查response.data是否存在
                    if response.data is not None:
                        debug_log(f"Returned {len(response.data)} images")
                    else:
                        debug_log("No data field in response or data is empty")
                    
            except Exception as api_error:
                request_duration = time.time() - start_time
                if debug:
                    debug_log("API request failed!")
                    debug_log(f"Duration before failure: {request_duration:.2f} seconds")
                    debug_log(f"Error type: {type(api_error).__name__}")
                    debug_log(f"Error details: {str(api_error)}")
                return self._create_error_image(f"图片编辑API调用失败: {str(api_error)}")
            
            if debug:
                debug_log(f"Analyzing API response: {type(response)}")
                # 打印API返回的JSON数据
                import json
                try:
                    resp_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                    if isinstance(resp_dict, dict):
                        # 打印完整的响应对象JSON体
                        debug_log("Complete response object JSON body:")
                        complete_resp = resp_dict.copy()
                        # 为了可读性，如果base64数据太长，截取前100和后100字符
                        if 'data' in complete_resp and isinstance(complete_resp['data'], list):
                            for i, item in enumerate(complete_resp['data']):
                                if isinstance(item, dict):
                                    complete_item = item.copy()
                                    for field in ['b64_json', 'b64', 'base64']:
                                        if field in complete_item and isinstance(complete_item[field], str) and len(complete_item[field]) > 200:
                                            b64_data = complete_item[field]
                                            complete_item[field] = f"{b64_data[:100]}...{b64_data[-100:]} [full length: {len(b64_data)} chars]"
                                    complete_resp['data'][i] = complete_item
                        debug_log(f"Complete response: {json.dumps(complete_resp, ensure_ascii=False, indent=2)}")
                    else:
                        truncated_dict = _truncate_base64_in_dict(resp_dict)
                        debug_log(f"Truncated response: {truncated_dict}")
                except Exception as e:
                    debug_log(f"JSON serialization failed: {e}")
                    # 安全地打印响应信息，避免base64数据
                    response_str = str(response)
                    if len(response_str) > 500:
                        response_str = f"{response_str[:200]}... [response too long, truncated, total length: {len(response_str)} chars] ...{response_str[-200:]}"
                    debug_log(f"Raw response: {response_str}")
                    # 如果JSON序列化失败，尝试打印响应对象的属性
                    if hasattr(response, '__dict__'):
                        response_dict = response.__dict__
                        truncated_dict = _truncate_base64_in_dict(response_dict)
                        debug_log(f"Response object attributes: {truncated_dict}")
                    else:
                        available_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                        debug_log(f"Available response attributes: {available_attrs}")
                
                # 解析响应
                if response.data is not None and len(response.data) > 0:
                    if debug:
                        debug_log("Starting to parse response data...")
                        
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
                                debug_log(f"Found base64 data in field '{attr_name}'")
                                debug_log(f"Data length: {len(b64_data)} characters")
                                debug_log(f"Data type: {type(b64_data)}")
                            break
                
                if not b64_data:
                    # 检查是否有URL字段，支持URL响应
                    if hasattr(first_item, 'url') and first_item.url:
                        if debug:
                            debug_log("No base64 data found, starting URL download")
                            debug_log(f"Image URL: {first_item.url}")
                        
                        try:
                            # 从URL下载图片
                            download_start = time.time()
                            img = download_image_from_url(first_item.url, debug=debug)
                            download_duration = time.time() - download_start
                            
                            if debug:
                                debug_log(f"URL download completed, duration: {download_duration:.2f} seconds")
                                debug_log(f"Image info: {img.size} {img.mode}")
                            
                            tensor = pil_to_tensor(img)
                            return tensor
                        except Exception as e:
                            if debug:
                                # 安全地打印错误信息，避免base64数据
                                error_str = str(e)
                                if len(error_str) > 500:
                                    error_str = f"{error_str[:250]}... [error too long, truncated, total length: {len(error_str)} chars] ...{error_str[-250:]}"
                                debug_log(f"URL download failed: {error_str}")
                            return self._create_error_image(f"从URL下载图片失败: {str(e)}")
                    else:
                        available_attrs = [attr for attr in dir(first_item) if not attr.startswith('_')]
                        if debug:
                            debug_log("No valid base64 or URL data found")
                            debug_log(f"Available attributes: {available_attrs}")
                        return self._create_error_image("响应中没有图片数据，请重试")
                
                if debug:
                    debug_log("Starting base64 data validation and decoding...")
                    
                if not isinstance(b64_data, str):
                    if debug:
                        debug_log(f"Base64 data type error: expected str, got {type(b64_data)}")
                    return self._create_error_image(f"base64数据格式错误，期望字符串，实际: {type(b64_data)}")
                
                try:
                    decode_start = time.time()
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                    tensor = pil_to_tensor(img)
                    decode_duration = time.time() - decode_start
                    
                    if debug:
                        debug_log("Base64 decode successful!")
                        debug_log(f"Decode duration: {decode_duration:.2f} seconds")
                        debug_log(f"Generated image info: {img.size} {img.mode}")
                        
                        total_duration = time.time() - start_time
                        debug_log("Image editing completed!")
                        debug_log(f"Total processing time: {total_duration:.2f} seconds")
                        debug_log(f"Data source: {found_field}")
                    
                    return tensor
                except Exception as e:
                    if debug:
                        # 安全地打印错误信息，避免base64数据
                        error_str = str(e)
                        if len(error_str) > 500:
                            error_str = f"{error_str[:250]}... [error too long, truncated, total length: {len(error_str)} chars] ...{error_str[-250:]}"
                        debug_log(f"Base64 decode failed: {error_str}")
                        debug_log(f"First 100 chars of base64 data: {b64_data[:100] if len(b64_data) > 100 else b64_data}")
                    raise ValueError(f"base64图像数据解码失败: {e}")
            else:
                if debug:
                    debug_log("API returned empty response or no data")
                    debug_log(f"Response object type: {type(response)}")
                    # 安全地打印response.data信息
                    if hasattr(response, 'data') and response.data:
                        if isinstance(response.data, list):
                            debug_log(f"response.data length: {len(response.data)} elements")
                        else:
                            data_str = str(response.data)
                            if len(data_str) > 200:
                                data_str = f"{data_str[:100]}... [data too long, truncated, total length: {len(data_str)} chars] ...{data_str[-100:]}"
                            debug_log(f"response.data value: {data_str}")
                    else:
                        debug_log("response.data: no data or empty")
                    debug_log(f"response.data type: {type(response.data) if hasattr(response, 'data') else 'No data attribute'}")
                    
                    # 尝试获取响应的所有属性
                    if hasattr(response, '__dict__'):
                        debug_log(f"Response object attributes: {list(response.__dict__.keys())}")
                    else:
                        available_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                        debug_log(f"Available response attributes: {available_attrs}")
                
                raise RuntimeError("API 返回空响应或无数据")
            
        except Exception as e:
            error_msg = f"图片编辑失败: {str(e)}"
            if "safety system" in str(e).lower():
                error_msg += "\n提示: 请尝试调整提示词或使用其他图片，避免可能违反安全策略的内容"
            if debug:
                debug_log(error_msg)
            raise RuntimeError(error_msg)
        finally:
            # 清理文件对象
            for img_file in image_files if 'image_files' in locals() else []:
                try:
                    img_file.close()
                except:
                    pass
    
    def _chat_mode_image(self, client, model, prompt, input_images, size, streaming, debug):
        """兼容模式：通过chat端点实现图像生成/编辑"""
        if debug:
            debug_log(f"Starting compatibility mode image processing: model={model}, streaming={streaming}")
            debug_log(f"Prompt: {prompt[:100]}...")
            debug_log(f"Size: {size}")
            if input_images:
                debug_log(f"Input images: {len(input_images)} (edit mode)")
            else:
                debug_log("Text-only generation mode")
        
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
                    debug_log("Processing input images to base64...")
                    
                for i, img_tensor in enumerate(input_images, 1):
                    try:
                        if debug:
                            debug_log(f"Processing image {i}/{len(input_images)}")
                            
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
                                debug_log(f"Image resized to: {new_size}")
                        
                        # 转换为base64 (pil_to_base64返回完整的data URI)
                        img_base64 = pil_to_base64(pil_img)
                        
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64  # 不需要再添加前缀，pil_to_base64已经包含了
                            }
                        })
                        
                        if debug:
                            debug_log(f"Image {i} converted to base64 successfully")
                            
                    except Exception as e:
                        if debug:
                            debug_log(f"Failed to process image {i}: {str(e)[:100]}...")
                        continue
            
            # 构建聊天消息
            messages = [{
                "role": "user",
                "content": message_content
            }]
            
            if debug:
                debug_log("Sending chat request...")
                debug_log(f"Request messages structure: {len(messages)} messages")
            
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
                stream=streaming,
                include_usage=True,
                debug=debug
            )
            api_duration = time.time() - api_start
            
            if debug:
                mode_str = "streaming" if streaming else "non-streaming"
                debug_log(f"Chat API call completed ({mode_str}), duration: {api_duration:.2f}s")
            
            # 解析响应
            if response:
                if debug:
                    content = response.get("content", "")
                    debug_log(f"Received response content length: {len(content) if content else 0} characters")
                
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
                        debug_log("Successfully extracted image data from response")
                    
                    # 转换为tensor
                    decode_start = time.time()
                    try:
                        if image_data.startswith('http'):
                            # URL格式
                            pil_img = download_image_from_url(image_data, debug=debug)
                        else:
                            # Base64格式
                            img = Image.open(BytesIO(base64.b64decode(image_data)))
                            pil_img = img
                    except Exception as e:
                        if debug:
                            error_str = str(e)
                            if len(error_str) > 500:
                                error_str = f"{error_str[:250]}... [错误太长，已截断，总长度: {len(error_str)} 字符] ...{error_str[-250:]}"
                            debug_log(f"Image data processing failed: {error_str}")
                        return self._create_error_image(f"图像数据处理失败: {str(e)}")
                    
                    tensor = pil_to_tensor(pil_img)
                    decode_duration = time.time() - decode_start
                    
                    total_duration = time.time() - total_start
                    if debug:
                        debug_log("Compatible mode image processing completed!")
                        debug_log(f"Total duration: {total_duration:.2f} seconds")
                        debug_log(f"API call: {api_duration:.2f} seconds")
                        debug_log(f"Image parsing: {decode_duration:.2f} seconds")
                        debug_log(f"Final image: {pil_img.size} {pil_img.mode}")
                    
                    return tensor
                else:
                    if debug:
                        debug_log("Failed to extract image data from response")
                        debug_log(f"Raw response content: {content[:500]}...")
                    
                    # 创建一个包含错误信息的图片，而不是抛出异常
                    return self._create_error_image("响应中没有图片数据，请重试")
            else:
                if debug:
                    debug_log("Chat API returned empty response")
                return self._create_error_image("响应中没有图片数据，请重试")
                
        except Exception as e:
            total_duration = time.time() - total_start
            if debug:
                debug_log("Compatible mode processing failed!")
                debug_log(f"Duration before failure: {total_duration:.2f} seconds")
                # 安全地打印错误信息，避免base64数据
                error_str = str(e)
                if len(error_str) > 500:
                    error_str = f"{error_str[:250]}... [error message too long, truncated, total length: {len(error_str)} chars] ...{error_str[-250:]}"
                debug_log(f"Error: {error_str}")
            return self._create_error_image(f"兼容模式处理失败: {str(e)}")
    
    def _extract_image_from_chat_response(self, response, debug=False):
        """从chat响应中提取图像数据（URL或base64）"""
        if not response:
            return None
        
        import re
        import base64
        
        if debug:
            debug_log(f"Starting image data extraction from response...")
            debug_log(f"Response type: {type(response)}")
            if isinstance(response, dict):
                debug_log(f"Response fields: {list(response.keys())}")
        
        # 首先尝试从响应结构中直接提取图像（新格式）
        # 检查是否有 images 字段（某些API返回的格式）
        if isinstance(response, dict):
            # 尝试从不同的可能位置提取图像
            images_data = None
            
            # 优先检查经过chat_complete处理后的images字段（流式聚合后的结果）
            if "images" in response and response["images"]:
                images_data = response["images"]
                if debug:
                    debug_log(f"Found images field at response top level: {type(images_data)} (length: {len(images_data) if isinstance(images_data, list) else 'N/A'})")
            
            # 检查是否有 choices[0].message.images (原始响应格式)
            elif "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                if "images" in message:
                    images_data = message["images"]
                    if debug:
                        debug_log(f"Found images field in choices[0].message: {type(images_data)}")
                        
            # 如果还是没找到，打印调试信息看看response结构
            else:
                if debug:
                    debug_log("No images field found in expected locations")
                    debug_log(f"Response top-level fields: {list(response.keys()) if isinstance(response, dict) else 'not dict'}")
                    if isinstance(response, dict) and "choices" in response and response["choices"]:
                        message = response["choices"][0].get("message", {})
                        debug_log(f"Message fields: {list(message.keys()) if isinstance(message, dict) else 'not dict'}")
            
            # 🔧 处理找到的images数据
            if images_data:
                if debug:
                    debug_log(f"Starting to process images data: {type(images_data)}")
                    
                # 如果images_data是列表
                if isinstance(images_data, list) and len(images_data) > 0:
                    first_image = images_data[0]
                    if debug:
                        debug_log(f"First image item type: {type(first_image)}")
                        if isinstance(first_image, dict):
                            debug_log(f"First image item fields: {list(first_image.keys())}")
                    
                    if isinstance(first_image, dict):
                        # 检查 image_url.url 字段 (OpenRouter/Gemini格式)
                        if "image_url" in first_image and "url" in first_image["image_url"]:
                            url = first_image["image_url"]["url"]
                            if debug:
                                debug_log(f"Found image_url.url field: {url[:100]}..." if len(url) > 100 else f"Found image_url.url field: {url}")
                            
                            if url.startswith("data:image"):
                                # 提取base64部分
                                if "base64," in url:
                                    base64_data = url.split("base64,", 1)[1]
                                    if debug:
                                        debug_log(f"Found base64 data from images field, length: {len(base64_data)}")
                                    return base64_data
                            else:
                                if debug:
                                    debug_log(f"Found URL from images field: {url[:100]}...")
                                return url
                        
                        # 检查直接的url字段
                        elif "url" in first_image:
                            url = first_image["url"]
                            if debug:
                                debug_log(f"Found direct url field: {url[:100]}...")
                            
                            if url.startswith("data:image") and "base64," in url:
                                base64_data = url.split("base64,", 1)[1]
                                if debug:
                                    debug_log(f"Found base64 data from url field, length: {len(base64_data)}")
                                return base64_data
                            else:
                                if debug:
                                    debug_log(f"Found URL from url field: {url[:100]}...")
                                return url
                        
                        # 检查其他可能的字段名
                        for field in ["data", "content", "base64"]:
                            if field in first_image and first_image[field]:
                                data = first_image[field]
                                if debug:
                                    debug_log(f"Found {field} field: {type(data)}")
                                
                                if isinstance(data, str) and len(data) > 100:
                                    try:
                                        # 尝试验证为base64
                                        base64.b64decode(data)
                                        if debug:
                                            debug_log(f"Found valid base64 data from {field} field, length: {len(data)}")
                                        return data
                                    except:
                                        pass
                    
                    elif isinstance(first_image, str):
                        # 如果直接是字符串（可能是base64或URL）
                        if debug:
                            debug_log(f"Image item is string, length: {len(first_image)}")
                        
                        if first_image.startswith("data:image") and "base64," in first_image:
                            base64_data = first_image.split("base64,", 1)[1]
                            if debug:
                                debug_log(f"Found base64 data from string, length: {len(base64_data)}")
                            return base64_data
                        elif first_image.startswith("http"):
                            if debug:
                                debug_log(f"Found URL from string: {first_image[:100]}...")
                            return first_image
                        elif len(first_image) > 100:
                            try:
                                # 尝试验证为base64
                                base64.b64decode(first_image)
                                if debug:
                                    debug_log(f"String is valid base64, length: {len(first_image)}")
                                return first_image
                            except:
                                pass
                
                elif isinstance(images_data, str):
                    # 如果images_data直接是字符串
                    if debug:
                        debug_log(f"images_data is string, length: {len(images_data)}")
                    
                    if images_data.startswith("data:image") and "base64," in images_data:
                        base64_data = images_data.split("base64,", 1)[1]
                        if debug:
                            debug_log(f"Found base64 data from images_data string, length: {len(base64_data)}")
                        return base64_data
                    elif images_data.startswith("http"):
                        if debug:
                            debug_log(f"Found URL from images_data string: {images_data}")
                        return images_data
        
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
                debug_log("No content or images field in response")
            return None
        
        if debug:
            debug_log(f"Searching for image data in content, content length: {len(content)}")
        
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
                            debug_log(f"Found base64 data from content, length: {len(clean_match)}")
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
                    debug_log(f"Found URL from content: {url}")
                return url
        
        if debug:
            debug_log("No image data found in response")
        
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
            return tensor_img
            
        except Exception:
            # If error image creation fails, return a simple red image
            img = Image.new('RGB', (256, 256), color='red')
            tensor_img = pil_to_tensor(img)
            return tensor_img
    
    def _translate_error_to_english(self, error_msg):
        """Translate common Chinese error messages to English"""
        # Common Chinese to English translations
        translations = {
            # Provider 相关
            "未找到 provider": "Provider not found",
            "API 密钥未配置或仍为占位符": "API key not configured or still placeholder",
            
            # 简化的用户友好消息
            "响应中没有图片数据，请重试": "No image data in response, please retry",
            "响应数据格式异常，请重试": "Response format error, please retry",
            
            # 图片处理相关
            "图片生成失败": "Image generation failed",
            "图片编辑失败": "Image editing failed",
            "图片生成API调用失败": "Image generation API call failed",
            "图片编辑API调用失败": "Image editing API call failed",
            "图像数据处理失败": "Image data processing failed",
            "兼容模式处理失败": "Compatibility mode processing failed",
            
            # 网络请求相关
            "下载图片失败": "Image download failed",
            "从URL下载图片失败": "Failed to download image from URL",
            "无法连接到图片URL": "Unable to connect to image URL",
            "下载图片超时": "Image download timeout",
            
            # 数据格式相关
            "base64图像数据解码失败": "Base64 image data decode failed",
            "没有有效的图片可以处理": "No valid images to process",
            "无效的base64数据格式": "Invalid base64 data format",
            "未返回有效的base64数据或URL": "No valid base64 data or URL returned",
            "base64数据格式错误": "Base64 data format error",
            
            # 通用错误
            "期望字符串，收到": "Expected string, received",
            "模型响应": "Model response",
            "错误信息太长，已截断，总长度": "Error message too long, truncated, total length",
            "字符": "characters",
            "请重试": "please retry"
        }
        
        english_msg = error_msg
        for chinese, english in translations.items():
            english_msg = english_msg.replace(chinese, english)
        
        return english_msg

__all__ = ["OpenAIImageNode"]
