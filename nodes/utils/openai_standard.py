"""Standardized OpenAI helpers for FlexAI (openai_* nodes).

Goals:
 - 只使用现代 openai>=1.x 客户端
 - 提供统一: client 创建 / chat 完成 / image 生成 / 重试逻辑
 - 精简节点实现, 降低重复代码
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Dict, Any, Callable, Type
import time
from functools import wraps
from openai import OpenAI
from openai import RateLimitError, APIError

__all__ = [
    "ensure_client",
    "build_multimodal_messages",
    "chat_complete",
    "generate_image_b64",
]


def ensure_client(api_key: str, base_url: str) -> OpenAI:
    """Return a configured OpenAI client (requires modern SDK)."""
    return OpenAI(api_key=api_key, base_url=base_url)


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
    
    # Debug: 打印提交到API的原生JSON数据
    if debug:
        import json
        print("=" * 50)
        print("[DEBUG] 提交到OpenAI Chat API的原生JSON数据:")
        # 创建一个可序列化的副本用于打印
        debug_kwargs = kwargs.copy()
        try:
            print(json.dumps(debug_kwargs, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"JSON序列化失败: {e}")
            print(f"原始kwargs: {debug_kwargs}")
        print("=" * 50)
    
    if stream:
        # Aggregate streaming manually for simplicity; capture finish_reason & usage
        full = []
        finish_reason = None
        usage = None
        stream_chunks = []  # 用于调试存储所有chunk
        
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
            if hasattr(choice, "delta") and getattr(choice.delta, "content", None):
                full.append(choice.delta.content)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        
        result = {"content": "".join(full), "finish_reason": finish_reason, "usage": usage}
        
        # Debug: 打印从API返回的原生JSON数据 (流式)
        if debug:
            import json
            print("=" * 50)
            print("[DEBUG] 从OpenAI Chat API返回的原生JSON数据 (流式):")
            print("流式chunk数量:", len(stream_chunks))
            if len(stream_chunks) <= 5:  # 如果chunk不多，全部打印
                for i, chunk in enumerate(stream_chunks):
                    print(f"Chunk {i+1}:")
                    try:
                        if isinstance(chunk, dict):
                            print(json.dumps(chunk, ensure_ascii=False, indent=2))
                        else:
                            print(chunk)
                    except Exception as e:
                        print(f"打印chunk失败: {e}")
                    print("-" * 30)
            else:  # 如果chunk太多，只打印前几个和最后几个
                print("前3个chunk:")
                for i in range(3):
                    print(f"Chunk {i+1}:")
                    try:
                        if isinstance(stream_chunks[i], dict):
                            print(json.dumps(stream_chunks[i], ensure_ascii=False, indent=2))
                        else:
                            print(stream_chunks[i])
                    except Exception as e:
                        print(f"打印chunk失败: {e}")
                    print("-" * 20)
                print("... (省略中间chunk) ...")
                print("最后2个chunk:")
                for i in range(-2, 0):
                    print(f"Chunk {len(stream_chunks)+i+1}:")
                    try:
                        if isinstance(stream_chunks[i], dict):
                            print(json.dumps(stream_chunks[i], ensure_ascii=False, indent=2))
                        else:
                            print(stream_chunks[i])
                    except Exception as e:
                        print(f"打印chunk失败: {e}")
                    print("-" * 20)
            
            print("聚合后的最终结果:")
            try:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"JSON序列化失败: {e}")
                print(f"原始结果: {result}")
            print("=" * 50)
        
        return result
    else:
        resp = client.chat.completions.create(stream=False, **kwargs)  # type: ignore[arg-type]
        
        # Debug: 打印从API返回的原生JSON数据 (非流式)
        if debug:
            import json
            print("=" * 50)
            print("[DEBUG] 从OpenAI Chat API返回的原生JSON数据 (非流式):")
            try:
                resp_dict = resp.model_dump() if hasattr(resp, 'model_dump') else str(resp)
                if isinstance(resp_dict, dict):
                    print(json.dumps(resp_dict, ensure_ascii=False, indent=2))
                else:
                    print(resp_dict)
            except Exception as e:
                print(f"JSON序列化失败: {e}")
                print(f"原始响应: {resp}")
            print("=" * 50)
        
        content = resp.choices[0].message.content if resp.choices else ""
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        usage = getattr(resp, "usage", None)
        result = {"content": content, "finish_reason": finish_reason, "usage": usage}
        
        # Debug: 打印处理后的结果
        if debug:
            import json
            print("=" * 50)
            print("[DEBUG] 处理后的最终结果:")
            try:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"JSON序列化失败: {e}")
                print(f"原始结果: {result}")
            print("=" * 50)
        
        return result


@with_retry()
def generate_image_b64(client: OpenAI, *, model: str, prompt: str, size: str, seed: Optional[int], debug: bool) -> str:
    """Generate a single image returning base64 (PNG/JPEG) via images.generate.

    Returns raw base64 (no data URL prefix). If API returns URL, downloads and converts to base64."""
    kwargs: Dict[str, Any] = {"model": model, "prompt": prompt, "size": size, "response_format": "b64_json"}
    # Note: OpenAI images.generate does not support seed parameter
    # Some providers may support it, but official OpenAI API does not
    if debug and seed:
        print(f"[DEBUG] Seed {seed} specified but images.generate does not support seed parameter")
    
    # Debug: 打印提交到API的原生JSON数据
    if debug:
        import json
        print("=" * 60)
        print("[DEBUG] 🚀 开始图片生成请求")
        print(f"[DEBUG] ⏰ 请求时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("[DEBUG] 📝 提交到OpenAI Images API的原生JSON数据:")
        try:
            print(json.dumps(kwargs, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"JSON序列化失败: {e}")
            print(f"原始kwargs: {kwargs}")
        print("=" * 60)
        print("[DEBUG] 📡 正在发送API请求...")
        print("[DEBUG] ⚠️  注意: OpenAI图片生成是同步API，需要等待完整生成后返回")
        print("[DEBUG] 💡 生成时间通常在10-60秒之间，请耐心等待...")
    
    # 记录请求开始时间
    start_time = time.time()
    
    try:
        resp = client.images.generate(**kwargs)
        request_duration = time.time() - start_time
        
        if debug:
            print(f"[DEBUG] ✅ API请求成功完成!")
            print(f"[DEBUG] ⏱️  总耗时: {request_duration:.2f} 秒")
    except Exception as e:
        request_duration = time.time() - start_time
        if debug:
            print(f"[DEBUG] ❌ API请求失败!")
            print(f"[DEBUG] ⏱️  失败前耗时: {request_duration:.2f} 秒")
            print(f"[DEBUG] 📋 错误类型: {type(e).__name__}")
            print(f"[DEBUG] 🔍 错误详情: {str(e)}")
        raise
    
    # Debug: 打印从API返回的原生JSON数据
    if debug:
        import json
        print("=" * 60)
        print("[DEBUG] 📨 收到API响应数据分析:")
        print(f"[DEBUG] 📊 响应对象类型: {type(resp)}")
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
                print(json.dumps(debug_resp, ensure_ascii=False, indent=2))
            else:
                print(resp_dict)
        except Exception as e:
            print(f"JSON序列化失败: {e}")
            print(f"原始响应: {resp}")
        print(f"[DEBUG] 📊 Response data attribute: {getattr(resp, 'data', 'NO DATA ATTR')}")
        print("=" * 60)
    
    data_list = getattr(resp, 'data', [])
    if not data_list:
        if debug:
            print("[DEBUG] ❌ API返回空的data列表")
        raise ValueError("images.generate 返回空 data 列表")
    
    if debug:
        print(f"[DEBUG] 📊 收到 {len(data_list)} 个图片数据项")
    
    first = data_list[0]
    if debug:
        print(f"[DEBUG] 🔍 分析第一个数据项:")
        print(f"[DEBUG] 📊 数据项类型: {type(first)}")
        available_attrs = [attr for attr in dir(first) if not attr.startswith('_')]
        print(f"[DEBUG] 📋 可用属性: {available_attrs}")
    
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
                    print(f"[DEBUG] ✅ 在属性 '{attr_name}' 中找到base64数据")
                    if isinstance(field_value, str):
                        print(f"[DEBUG] 📏 Base64数据长度: {len(field_value)} 字符")
                        print(f"[DEBUG] 🔤 数据类型: {type(field_value)}")
                        # 检查base64数据的开头，判断图片格式
                        try:
                            import base64
                            sample_bytes = base64.b64decode(field_value[:100])
                            if sample_bytes.startswith(b'\x89PNG'):
                                print(f"[DEBUG] 🖼️  检测到PNG格式图片")
                            elif sample_bytes.startswith(b'\xff\xd8\xff'):
                                print(f"[DEBUG] 🖼️  检测到JPEG格式图片")
                            else:
                                print(f"[DEBUG] 🖼️  未知图片格式，前16字节: {sample_bytes[:16]}")
                        except:
                            pass
                    else:
                        print(f"[DEBUG] ⚠️  Base64数据不是字符串类型: {type(field_value)}")
                break
    
    if not b64_field:
        # If no b64 field, check if we have url field and download
        if hasattr(first, 'url') and first.url:
            if debug:
                print(f"[DEBUG] 🌐 未找到base64数据，开始从URL下载图片")
                print(f"[DEBUG] 🔗 图片URL: {first.url}")
                print(f"[DEBUG] ⏬ 开始下载...")
            
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
                    print(f"[DEBUG] 📡 发送HTTP GET请求...")
                    
                response = requests.get(first.url, headers=headers, timeout=30)
                response.raise_for_status()
                
                download_duration = time.time() - download_start
                
                if debug:
                    print(f"[DEBUG] ✅ 下载成功!")
                    print(f"[DEBUG] ⏱️  下载耗时: {download_duration:.2f} 秒")
                    print(f"[DEBUG] 📏 下载数据大小: {len(response.content)} 字节")
                    print(f"[DEBUG] 📄 Content-Type: {response.headers.get('content-type', 'unknown')}")
                    print(f"[DEBUG] 🔄 开始转换为base64...")
                
                # 转换为base64
                convert_start = time.time()
                img = Image.open(BytesIO(response.content))
                
                if debug:
                    print(f"[DEBUG] 🖼️  图片信息: {img.size} 像素, {img.mode} 模式")
                
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                b64_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                convert_duration = time.time() - convert_start
                
                if debug:
                    print(f"[DEBUG] ✅ 转换完成!")
                    print(f"[DEBUG] ⏱️  转换耗时: {convert_duration:.2f} 秒")
                    print(f"[DEBUG] 📏 Base64长度: {len(b64_data)} 字符")
                    print(f"[DEBUG] 🎯 返回base64数据")
                
                return b64_data
                
            except Exception as e:
                download_duration = time.time() - download_start
                if debug:
                    print(f"[DEBUG] ❌ URL下载失败!")
                    print(f"[DEBUG] ⏱️  失败前耗时: {download_duration:.2f} 秒")
                    print(f"[DEBUG] 📋 错误类型: {type(e).__name__}")
                    print(f"[DEBUG] 🔍 错误详情: {str(e)}")
                raise ValueError(f"无法从URL下载图片: {first.url}, 错误: {e}")
        else:
            if debug:
                print(f"[DEBUG] ❌ 未找到base64数据或URL字段")
                available_attrs = [attr for attr in dir(first) if not attr.startswith('_')]
                print(f"[DEBUG] 📋 可用属性列表: {available_attrs}")
            raise ValueError(f"images.generate 未返回 b64 字段或有效URL，可用属性: {available_attrs}")
    
    if debug:
        print(f"[DEBUG] 🔍 开始验证base64数据...")
    
    # 验证base64数据类型和内容
    if not isinstance(b64_field, str):
        if debug:
            print(f"[DEBUG] ❌ base64数据类型错误: 期望str，实际{type(b64_field)}")
        raise ValueError(f"base64数据类型错误，期望字符串，实际: {type(b64_field)}")
    
    if len(b64_field) == 0:
        if debug:
            print(f"[DEBUG] ❌ base64数据为空字符串")
        raise ValueError("base64数据为空字符串")
    
    # 简单验证base64格式（基本检查）
    try:
        # 尝试解码前几个字符来验证格式
        import base64
        sample_data = base64.b64decode(b64_field[:100] if len(b64_field) > 100 else b64_field, validate=True)
        
        if debug:
            print(f"[DEBUG] ✅ base64格式验证通过")
            # 尝试检测图片格式
            if sample_data.startswith(b'\x89PNG'):
                print(f"[DEBUG] 🖼️  检测到PNG格式")
            elif sample_data.startswith(b'\xff\xd8\xff'):
                print(f"[DEBUG] 🖼️  检测到JPEG格式")
            else:
                print(f"[DEBUG] 🖼️  未知格式，前16字节: {sample_data[:16]}")
            
    except Exception as e:
        if debug:
            print(f"[DEBUG] ❌ base64格式验证失败: {e}")
            print(f"[DEBUG] 🔍 数据前100字符: {b64_field[:100]}")
        raise ValueError(f"无效的base64数据格式: {e}")
    
    if debug:
        total_duration = time.time() - start_time
        print("=" * 60)
        print(f"[DEBUG] 🎉 图片生成完成!")
        print(f"[DEBUG] ⏱️  总处理时间: {total_duration:.2f} 秒")
        print(f"[DEBUG] 📏 最终base64数据长度: {len(b64_field)} 字符")
        print(f"[DEBUG] 📦 数据来源: {found_attr}")
        print("=" * 60)
    
    return b64_field

