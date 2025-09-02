"""Standardized OpenAI helpers for FlexAI (openai_* nodes).

Goals:
 - 只使用现代 openai>=1.x 客户端
 - 提供统一: client 创建 / chat 完成 / image 生成 / 重试逻辑
 - 精简节点实现, 降低重复代码
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Dict, Any, Callable, Type
import time
import os
from functools import wraps
from openai import OpenAI
from openai import RateLimitError, APIError

__all__ = [
    "ensure_client",
    "build_multimodal_messages", 
    "chat_complete",
    "generate_image_b64",
    "_truncate_base64_in_dict",
    "debug_log",
]


# Debug日志文件路径
DEBUG_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'debug.log')


def debug_log(message: str, debug: bool = True) -> None:
    """统一的debug日志函数，输出到debug.log文件
    
    Args:
        message: 日志消息
        debug: 是否启用debug模式
    """
    if not debug:
        return
        
    try:
        # 确保debug.log文件存在，清空旧内容（只在第一次调用时）
        if not hasattr(debug_log, '_initialized'):
            with open(DEBUG_LOG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Debug session started\n")
            debug_log._initialized = True
            
        # 写入新日志
        with open(DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        # 如果写入失败，静默忽略以避免影响主流程
        pass


def ensure_client(api_key: str, base_url: str) -> OpenAI:
    """Return a configured OpenAI client (requires modern SDK)."""
    return OpenAI(api_key=api_key, base_url=base_url)


def _truncate_base64_in_dict(obj: Any, max_length: int = 100) -> Any:
    """递归截断字典中的base64数据以便于调试显示，保留前后50字符"""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(value, str) and len(value) > max_length:
                # 检查是否是base64数据的几种情况
                is_base64 = False
                if 'base64' in key.lower():
                    is_base64 = True
                elif value.startswith('data:image/') and 'base64,' in value:
                    is_base64 = True
                elif len(value) > 200 and value.replace('/', '').replace('+', '').replace('=', '').replace('-', '').replace('_', '').isalnum():
                    is_base64 = True
                
                if is_base64:
                    # 截断base64数据，保留前后50字符
                    if len(value) > 100:
                        result[key] = f"{value[:50]}...[truncated]...{value[-50:]}"
                    else:
                        result[key] = value
                else:
                    # 对于非base64的长字符串，也进行截断
                    if len(value) > max_length:
                        result[key] = f"{value[:50]}...[truncated]...{value[-50:]}"
                    else:
                        result[key] = value
            else:
                result[key] = _truncate_base64_in_dict(value, max_length)
        return result
    elif isinstance(obj, list):
        return [_truncate_base64_in_dict(item, max_length) for item in obj]
    elif isinstance(obj, str) and len(obj) > max_length:
        # 处理字符串情况：检查是否是base64或长字符串
        is_base64 = False
        if obj.startswith('data:image/') and 'base64,' in obj:
            is_base64 = True
        elif len(obj) > 200 and obj.replace('/', '').replace('+', '').replace('=', '').replace('-', '').replace('_', '').isalnum():
            is_base64 = True
        
        if is_base64 or len(obj) > max_length:
            return f"{obj[:50]}...[truncated]...{obj[-50:]}"
        return obj
    else:
        return obj


# ---------------- Retry Decorator -----------------
def _retryable(ex: BaseException) -> bool:
    return isinstance(ex, (RateLimitError, APIError))


def with_retry(retries: int = 3, base_delay: float = 1.0, factor: float = 2.0):
    """Exponential backoff retry decorator for transient OpenAI errors."""
    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa
                    attempt += 1
                    if attempt > retries or not _retryable(e):
                        raise
                    time.sleep(delay)
                    delay *= factor
        return wrapper
    return deco


def build_multimodal_messages(system_prompt: Optional[str], user_text: str, image_data_urls: Sequence[str]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    messages.append({"role": "user", "content": content})
    return messages


@with_retry()
def chat_complete(client: OpenAI, *, model: str, messages: List[Dict[str, Any]], temperature: float, top_p: float,
                  max_tokens: int, seed: Optional[int], stream: bool, include_usage: bool, debug: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(model=model, messages=messages, temperature=temperature, top_p=top_p)
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if seed and seed > 0:
        kwargs["seed"] = seed
    if stream and include_usage:
        kwargs["stream_options"] = {"include_usage": True}
    
    # Debug: 记录提交到API的关键信息
    if debug:
        debug_log(f"Chat API request - model: {model}, stream: {stream}")
    
    if stream:
        # Aggregate streaming manually for simplicity; capture finish_reason & usage
        full = []
        finish_reason = None
        usage = None
        stream_chunks = []  # 用于调试存储所有chunk
        all_delta_data = {}  # 存储所有delta中的字段
        
        debug_log("Starting streaming response processing", debug)
        
        for chunk in client.chat.completions.create(stream=True, **kwargs):  # type: ignore[arg-type]
            if debug:
                # 收集chunk用于调试打印
                try:
                    chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else str(chunk)
                    stream_chunks.append(chunk_dict)
                except Exception:
                    stream_chunks.append(str(chunk))
            
            choices = getattr(chunk, "choices", [])
            if not choices:
                if hasattr(chunk, "usage"):
                    usage = getattr(chunk, "usage")
                continue
                
            choice = choices[0]
            
            # 处理content
            if hasattr(choice, "delta") and getattr(choice.delta, "content", None):
                full.append(choice.delta.content)
            
            # 收集delta中的所有其他字段（通用递归遍历）
            if hasattr(choice, "delta"):
                delta = choice.delta
                
                # 简化的字段提取：直接尝试已知字段 + __dict__ 遍历
                extracted_fields = {}
                
                # 方法1: 直接尝试已知的关键字段
                known_fields = ['images', 'tool_calls', 'function_call', 'role', 'refusal', 'attachments']
                for field_name in known_fields:
                    try:
                        if hasattr(delta, field_name):
                            field_value = getattr(delta, field_name)
                            if field_value is not None:
                                extracted_fields[field_name] = field_value
                    except Exception as e:
                        if debug:
                            debug_log(f"Failed to extract field '{field_name}': {e}")
                
                # 方法2: __dict__ 遍历
                try:
                    if hasattr(delta, '__dict__'):
                        for key, value in delta.__dict__.items():
                            if key.startswith('_') or key == 'content' or value is None:
                                continue
                            if key not in extracted_fields:  # 避免重复
                                extracted_fields[key] = value
                except Exception as e:
                    pass
                
                # 方法3: dir() 遍历公共属性
                try:
                    for attr_name in dir(delta):
                        if attr_name.startswith('_') or attr_name == 'content':
                            continue
                        if attr_name in extracted_fields:  # 避免重复
                            continue
                        if attr_name in known_fields:  # 已经处理过了
                            continue
                            
                        try:
                            attr_value = getattr(delta, attr_name)
                            # 跳过方法和特殊类型
                            if callable(attr_value) or isinstance(attr_value, (type, type(None))):
                                continue
                            if attr_value is not None:
                                extracted_fields[attr_name] = attr_value
                        except Exception:
                            continue
                except Exception as e:
                    pass
                
                # 方法4: 特殊情况处理 - 对于可能使用动态属性的OpenAI对象
                special_fields = ['images', 'attachments', 'tool_calls', 'function_call']
                for field_name in special_fields:
                    if field_name not in extracted_fields:
                        try:
                            # 尝试直接访问，即使hasattr返回False
                            field_value = getattr(delta, field_name, None)
                            if field_value is not None:
                                extracted_fields[field_name] = field_value
                        except Exception:
                            continue
                
                # 方法3: dir() 遍历公共属性
                try:
                    for attr_name in dir(delta):
                        if attr_name.startswith('_') or attr_name == 'content':
                            continue
                        if attr_name in extracted_fields:  # 避免重复
                            continue
                        if attr_name in known_fields:  # 已经处理过了
                            continue
                            
                        try:
                            attr_value = getattr(delta, attr_name)
                            # 跳过方法和特殊类型
                            if callable(attr_value) or isinstance(attr_value, (type, type(None))):
                                continue
                            if attr_value is not None:
                                extracted_fields[attr_name] = attr_value
                        except Exception:
                            continue
                except Exception as e:
                    if debug:
                        debug_log(f"dir() traversal failed: {e}")
                
                # 方法4: 特殊情况处理 - 对于可能使用动态属性的OpenAI对象
                # 有些字段可能通过__getattr__或描述符实现，不在__dict__中
                special_fields = ['images', 'attachments', 'tool_calls', 'function_call']
                for field_name in special_fields:
                    if field_name not in extracted_fields:
                        try:
                            # 尝试直接访问，即使hasattr返回False
                            field_value = getattr(delta, field_name, None)
                            if field_value is not None:
                                extracted_fields[field_name] = field_value
                        except (AttributeError, Exception):
                            # 继续尝试下一个字段
                            continue
                
                # 将提取到的字段添加到all_delta_data中
                for key, value in extracted_fields.items():
                    if key not in all_delta_data:
                        all_delta_data[key] = []
                    
                    # 根据数据类型进行不同处理
                    if isinstance(value, list):
                        all_delta_data[key].extend(value)
                    else:
                        all_delta_data[key].append(value)
            
            # 检查完成状态
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
                debug_log(f"Streaming completed with reason: {finish_reason}", debug)
        
        # 构建增强的结果，包含所有收集到的字段
        result = {
            "content": "".join(full), 
            "finish_reason": finish_reason, 
            "usage": usage
        }
        
        # 将收集到的额外字段添加到结果中
        if all_delta_data:
            debug_log(f"Collected fields from streaming: {list(all_delta_data.keys())}", debug)
            for key, values in all_delta_data.items():
                # 去重并合并值
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], dict):
                        # 如果是字典列表，直接使用（通常是images数组）
                        result[key] = values
                    elif isinstance(values[0], list):
                        # 如果是嵌套列表，展平
                        flat_values = []
                        for v in values:
                            flat_values.extend(v)
                        result[key] = flat_values
                    else:
                        # 其他情况，使用最后一个非空值或所有值
                        result[key] = values[-1] if len(values) == 1 else values
        
        debug_log(f"Streaming completed: content_length={len(result.get('content', ''))}, fields={list(all_delta_data.keys())}", debug)
        
        return result
    else:
        resp = client.chat.completions.create(stream=False, **kwargs)  # type: ignore[arg-type]
        
        # Debug: 打印从API返回的原生JSON数据 (非流式)
        if debug:
            import json
            debug_log("Raw JSON data from OpenAI Chat API (non-streaming):")
            try:
                resp_dict = resp.model_dump() if hasattr(resp, 'model_dump') else str(resp)
                if isinstance(resp_dict, dict):
                    # 截断base64数据后再打印
                    truncated_resp = _truncate_base64_in_dict(resp_dict)
                    debug_log(f"Response data: {json.dumps(truncated_resp, ensure_ascii=False, indent=2)}")
                else:
                    # 如果不是dict，安全地打印字符串表示
                    resp_str = str(resp_dict)
                    if len(resp_str) > 1000:
                        resp_str = f"{resp_str[:500]}... [response too long, truncated, total length: {len(resp_str)} chars] ...{resp_str[-500:]}"
                    debug_log(f"Response string: {resp_str}")
            except Exception as e:
                debug_log(f"JSON serialization failed: {e}")
                # 安全地打印响应对象，避免base64数据
                resp_str = str(resp)
                if len(resp_str) > 1000:
                    resp_str = f"{resp_str[:500]}... [response too long, truncated, total length: {len(resp_str)} chars] ...{resp_str[-500:]}"
                debug_log(f"Raw response: {resp_str}")
        
        content = resp.choices[0].message.content if resp.choices else ""
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        usage = getattr(resp, "usage", None)
        
        # 提取可能的images字段
        images = None
        if resp.choices and hasattr(resp.choices[0].message, 'images'):
            images = resp.choices[0].message.images
        
        result = {
            "content": content, 
            "finish_reason": finish_reason, 
            "usage": usage,
            "images": images,
            "_raw_response": resp  # 包含原始响应以便提取其他字段
        }
        
        # Debug: 记录处理后的结果
        if debug:
            debug_log(f"Non-streaming response processed: content_length={len(result.get('content', ''))}")
        
        return result


@with_retry()
def generate_image_b64(client: OpenAI, *, model: str, prompt: str, size: str, seed: Optional[int], debug: bool) -> str:
    """Generate a single image returning base64 (PNG/JPEG) via images.generate.

    Returns raw base64 (no data URL prefix). If API returns URL, downloads and converts to base64."""
    kwargs: Dict[str, Any] = {"model": model, "prompt": prompt, "size": size, "response_format": "b64_json"}
    # Note: OpenAI images.generate does not support seed parameter
    # Some providers may support it, but official OpenAI API does not
    if debug and seed:
        debug_log(f"Seed {seed} specified but images.generate does not support seed parameter")
    
    # Debug: 记录请求开始
    if debug:
        debug_log(f"Image generation request: model={model}, size={size}")
    
    # 记录请求开始时间
    start_time = time.time()
    
    try:
        resp = client.images.generate(**kwargs)
        request_duration = time.time() - start_time
        
        if debug:
            debug_log(f"Image generation completed in {request_duration:.2f}s")
    except Exception as e:
        request_duration = time.time() - start_time
        if debug:
            debug_log(f"Image generation failed after {request_duration:.2f}s: {type(e).__name__}: {str(e)}")
        raise
    
    # Debug: 打印从API返回的原生JSON数据
    if debug:
        import json
        debug_log("Received API response data analysis:")
        debug_log(f"Response object type: {type(resp)}")
        try:
            resp_dict = resp.model_dump() if hasattr(resp, 'model_dump') else str(resp)
            if isinstance(resp_dict, dict):
                # 创建用于调试显示的副本（不包含长base64数据）
                debug_resp = resp_dict.copy()
                if 'data' in debug_resp and isinstance(debug_resp['data'], list):
                    for i, item in enumerate(debug_resp['data']):
                        if isinstance(item, dict):
                            debug_item = item.copy()
                            for field in ['b64_json', 'b64', 'base64', 'data']:
                                if field in debug_item and isinstance(debug_item[field], str):
                                    data_length = len(debug_item[field])
                                    debug_item[field] = f'<base64_data_length: {data_length}>'
                            debug_resp['data'][i] = debug_item
                debug_log(f"Response summary: {json.dumps(debug_resp, ensure_ascii=False, indent=2)}")
            else:
                debug_log(f"Response type: {type(resp)} (detailed content omitted for brevity)")
        except Exception as e:
            debug_log(f"Response analysis failed: {e}")
        
        # 安全地打印data属性信息
        if hasattr(resp, 'data'):
            data_attr = resp.data
            if isinstance(data_attr, list):
                debug_log(f"Response data: List with {len(data_attr)} items")
            elif data_attr is None:
                debug_log("Response data: None")
            else:
                data_str = str(data_attr)
                if len(data_str) > 200:
                    data_str = f"{data_str[:100]}... [data too long, truncated, total length: {len(data_str)} chars] ...{data_str[-100:]}"
                debug_log(f"Response data: {data_str}")
        else:
            debug_log("Response data attribute: NO DATA ATTR")
    
    data_list = getattr(resp, 'data', [])
    if not data_list:
        if debug:
            debug_log("API returned empty data list")
        raise ValueError("images.generate 返回空 data 列表")
    
    if debug:
        debug_log(f"Received {len(data_list)} image data items")
    
    first = data_list[0]
    if debug:
        debug_log(f"Analyzing first data item: {type(first)}")
    
    # Try different possible base64 field names
    b64_field = None
    found_attr = None
    for attr_name in ['b64_json', 'b64', 'base64', 'data']:
        if hasattr(first, attr_name):
            field_value = getattr(first, attr_name)
            if field_value:  # 确保不是空字符串或None
                b64_field = field_value
                found_attr = attr_name
                if debug:
                    debug_log(f"Found base64 data in '{attr_name}': length={len(field_value) if isinstance(field_value, str) else 'N/A'}")
                break
    
    if not b64_field:
        # If no b64 field, check if we have url field and download
        if hasattr(first, 'url') and first.url:
            if debug:
                debug_log(f"Downloading image from URL: {first.url}")
                debug_log("Starting download...")
            
            download_start = time.time()
            try:
                # 动态导入以避免循环导入
                import requests
                from PIL import Image
                from io import BytesIO
                import base64
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                if debug:
                    debug_log("Sending HTTP GET request...")
                    
                response = requests.get(first.url, headers=headers, timeout=30)
                response.raise_for_status()
                
                download_duration = time.time() - download_start
                
                if debug:
                    debug_log(f"Download completed: {len(response.content)} bytes in {download_duration:.2f}s")
                
                # 转换为base64
                convert_start = time.time()
                img = Image.open(BytesIO(response.content))
                
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                b64_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                convert_duration = time.time() - convert_start
                
                if debug:
                    debug_log(f"Conversion completed: base64 length={len(b64_data)} in {convert_duration:.2f}s")
                
                return b64_data
                
            except Exception as e:
                download_duration = time.time() - download_start
                if debug:
                    debug_log(f"URL download failed after {download_duration:.2f}s: {type(e).__name__}: {str(e)}")
                raise ValueError(f"无法从URL下载图片: {first.url}, 错误: {e}")
        else:
            if debug:
                available_attrs = [attr for attr in dir(first) if not attr.startswith('_')]
                debug_log(f"No base64 data or URL found. Available attributes: {available_attrs}")
            raise ValueError(f"images.generate 未返回 b64 字段或有效URL，可用属性: {available_attrs}")
    
    # 验证base64数据类型和内容
    if not isinstance(b64_field, str):
        if debug:
            debug_log(f"Base64 data type error: expected str, got {type(b64_field)}")
        raise ValueError(f"base64数据类型错误，期望字符串，实际: {type(b64_field)}")
    
    if len(b64_field) == 0:
        if debug:
            debug_log("Base64 data is empty string")
        raise ValueError("base64数据为空字符串")
    
    # 简单验证base64格式（基本检查）
    try:
        # 尝试解码前几个字符来验证格式
        import base64
        sample_data = base64.b64decode(b64_field[:100] if len(b64_field) > 100 else b64_field, validate=True)
        
        if debug:
            debug_log("Base64 format validation passed")
            # 尝试检测图片格式
            if sample_data.startswith(b'\x89PNG'):
                debug_log("Detected PNG format")
            elif sample_data.startswith(b'\xff\xd8\xff'):
                debug_log("Detected JPEG format")
            else:
                debug_log(f"Unknown format, first 16 bytes: {sample_data[:16]}")
            
    except Exception as e:
        if debug:
            debug_log(f"Base64 validation failed: {str(e)[:200]}...")
        raise ValueError(f"无效的base64数据格式: {e}")
    
    if debug:
        total_duration = time.time() - start_time
        debug_log(f"Image generation completed: duration={total_duration:.2f}s, data_length={len(b64_field)}, source={found_attr}")
    
    return b64_field

