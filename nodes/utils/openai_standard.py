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
    if stream:
        # Aggregate streaming manually for simplicity; capture finish_reason & usage
        full = []
        finish_reason = None
        usage = None
        for chunk in client.chat.completions.create(stream=True, **kwargs):  # type: ignore[arg-type]
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
        return {"content": "".join(full), "finish_reason": finish_reason, "usage": usage}
    else:
        resp = client.chat.completions.create(stream=False, **kwargs)  # type: ignore[arg-type]
        content = resp.choices[0].message.content if resp.choices else ""
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        usage = getattr(resp, "usage", None)
        return {"content": content, "finish_reason": finish_reason, "usage": usage}


@with_retry()
def generate_image_b64(client: OpenAI, *, model: str, prompt: str, size: str, seed: Optional[int], debug: bool) -> str:
    """Generate a single image returning base64 (PNG/JPEG) via images.generate.

    Returns raw base64 (no data URL prefix)."""
    kwargs: Dict[str, Any] = {"model": model, "prompt": prompt, "size": size, "response_format": "b64_json"}
    # Note: OpenAI images.generate does not support seed parameter
    # Some providers may support it, but official OpenAI API does not
    if debug and seed:
        print(f"[DEBUG] Seed {seed} specified but images.generate does not support seed parameter")
    if debug:
        print(f"[DEBUG] Calling images.generate with: {kwargs}")
    resp = client.images.generate(**kwargs)
    if debug:
        print(f"[DEBUG] Response object: {resp}")
        print(f"[DEBUG] Response data attribute: {getattr(resp, 'data', 'NO DATA ATTR')}")
    
    data_list = getattr(resp, 'data', [])
    if not data_list:
        raise ValueError("images.generate 返回空 data 列表")
    
    first = data_list[0]
    if debug:
        print(f"[DEBUG] First data item: {first}")
        print(f"[DEBUG] First data item attributes: {dir(first)}")
    
    # Try different possible base64 field names
    b64_field = None
    for attr_name in ['b64_json', 'b64', 'base64', 'data']:
        if hasattr(first, attr_name):
            b64_field = getattr(first, attr_name)
            if b64_field:
                if debug:
                    print(f"[DEBUG] Found base64 data in attribute: {attr_name}")
                break
    
    if not b64_field:
        # If no b64 field, check if we have url field and can download
        if hasattr(first, 'url') and first.url:
            if debug:
                print(f"[DEBUG] No b64 field found, got URL instead: {first.url}")
            raise ValueError("images.generate 返回 URL 而非 base64，请检查 API 配置或使用支持 base64 的提供商")
        else:
            available_attrs = [attr for attr in dir(first) if not attr.startswith('_')]
            raise ValueError(f"images.generate 未返回 b64 字段，可用属性: {available_attrs}")
    
    return b64_field

