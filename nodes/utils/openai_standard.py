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
    "log_api_interaction",
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
    """递归截断字典或对象中的base64数据以便于调试显示"""
    # 优先处理可转换为字典的对象
    if hasattr(obj, 'model_dump'):
        obj = obj.model_dump()
    elif hasattr(obj, 'dict'):
        obj = obj.dict()
    elif not isinstance(obj, (list, str, int, float, bool, type(None))) and hasattr(obj, '__dict__'):
        # 避免转换已知不会包含敏感数据的内置类型
        try:
            obj = vars(obj)
        except TypeError:
            pass # vars() fails on some types, just proceed

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
        # 尝试将长字符串解析为JSON并递归处理
        import json
        stripped_obj = obj.strip()
        if (stripped_obj.startswith('{') and stripped_obj.endswith('}')) or \
           (stripped_obj.startswith('[') and stripped_obj.endswith(']')):
            try:
                parsed_json = json.loads(obj)
                # 如果解析成功，则递归地对解析后的对象进行截断
                return _truncate_base64_in_dict(parsed_json, max_length)
            except (json.JSONDecodeError, TypeError):
                # 解析失败，则按普通字符串处理
                pass

        # 处理普通长字符串或无法解析为JSON的字符串
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


def log_api_interaction(title: str, payload: Any, debug: bool = True):
    """
    统一记录API交互日志，包括请求和响应。
    会自动截断base64数据并格式化为JSON。
    """
    if not debug:
        return

    import json
    
    debug_log(f"--- {title} ---")
    try:
        # 截断base64数据
        truncated_payload = _truncate_base64_in_dict(payload)
        # 格式化为JSON, 使用 default=str 来处理无法序列化的对象
        log_content = json.dumps(truncated_payload, ensure_ascii=False, indent=2, default=str)
        debug_log(log_content)
    except Exception as e:
        # 如果序列化失败（即使有default=str），记录错误并尝试打印原始对象
        debug_log(f"Could not serialize payload for debug log even with default=str: {e}")
        try:
            debug_log(str(payload))
        except Exception as e2:
            debug_log(f"Could not even convert payload to string: {e2}")
    debug_log("-" * (len(title) + 8))


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
        mode = "Streaming" if stream else "Non-Streaming"
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed
        }
        log_api_interaction(f"Chat API Request ({mode})", params, debug)
    
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
            
            # 收集delta中的所有其他字段
            if hasattr(choice, "delta"):
                delta = choice.delta
                
                # 使用 model_dump() 来安全地获取所有存在的字段
                try:
                    delta_dict = delta.model_dump(exclude_unset=True)
                    for key, value in delta_dict.items():
                        if key == 'content' or value is None:
                            continue
                        
                        if key not in all_delta_data:
                            all_delta_data[key] = []
                        
                        # 直接追加，后续再处理合并
                        all_delta_data[key].append(value)
                except Exception as e:
                    if debug:
                        debug_log(f"Could not process delta chunk via model_dump: {e}")
            
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
        
        # 将收集到的额外字段合并到结果中
        if all_delta_data:
            debug_log(f"Collected fields from streaming: {list(all_delta_data.keys())}", debug)
            for key, values in all_delta_data.items():
                # 展平列表的列表
                if all(isinstance(v, list) for v in values):
                    combined_list = []
                    for sublist in values:
                        combined_list.extend(sublist)
                    result[key] = combined_list
                else:
                    # 对于其他类型，简单地将所有收集到的值放在一个列表中
                    result[key] = values

        debug_log(f"Streaming completed: content_length={len(result.get('content', ''))}, fields={list(result.keys())}", debug)
        
        if debug:
            # 记录原始的chunks和聚合后的结果
            log_api_interaction("Raw Stream Chunks", stream_chunks, debug)
            log_api_interaction("Aggregated Chat API Response", result, debug)

        return result
    else:
        resp = client.chat.completions.create(stream=False, **kwargs)  # type: ignore[arg-type]
        
        # Debug: 打印从API返回的原生JSON数据 (非流式)
        if debug:
            log_api_interaction("Chat API Response (Non-Streaming)", resp, debug)
        
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
        log_api_interaction("Image Generation Request", kwargs, debug)
    
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
        log_api_interaction("Image Generation Response", resp, debug)
    
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
