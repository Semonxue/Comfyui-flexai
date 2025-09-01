"""Provider configuration utilities (suffix naming, simplified).

规则:
1. 多 provider: 使用 OPENAI_API_KEY_<token> 与可选 OPENAI_API_BASE_<token>
2. 可声明 OPENAI_PROVIDERS=token1,token2 来限定顺序; 若缺省则自动扫描 OPENAI_API_KEY_* 后缀。
3. 回退兼容单一: OPENAI_API_KEY / OPENAI_API_BASE。

示例:
OPENAI_API_KEY_default=sk_xxx
OPENAI_API_KEY_alt=sk_yyy
 -> tokens = default, alt

或显式:
OPENAI_PROVIDERS=default,alt
OPENAI_API_KEY_default=sk_xxx
OPENAI_API_KEY_alt=sk_yyy
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Provider:
    token: str          # token 名(用于下拉显示)
    api_key: str
    base_url: str


_CACHE: List[Provider] | None = None


def _legacy_single_provider() -> List[Provider]:
    """Single provider fallback (OPENAI_* only)."""
    api_key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
    if not api_key:
        return []
    return [Provider(token="default", api_key=api_key, base_url=base)]


def _parse_multi() -> List[Provider]:
    providers_var = os.getenv("OPENAI_PROVIDERS")
    tokens: List[str] = []
    if providers_var:
        tokens = [t.strip() for t in providers_var.split(',') if t.strip()]
    else:
        prefix = "OPENAI_API_KEY_"
        for k in os.environ.keys():
            if k.startswith(prefix):
                suffix = k[len(prefix):].strip()
                if suffix:
                    tokens.append(suffix)
        tokens = sorted(set(tokens))
    if not tokens:
        return _legacy_single_provider()
    providers: List[Provider] = []
    for token in tokens:
        key = os.getenv(f"OPENAI_API_KEY_{token}")
        if not key:
            continue
        base = os.getenv(f"OPENAI_API_BASE_{token}") or "https://api.openai.com/v1"
        providers.append(Provider(token=token, api_key=key, base_url=base))
    if not providers:
        return _legacy_single_provider()
    return providers


def load_providers(force_reload: bool = False) -> List[Provider]:
    global _CACHE
    if _CACHE is None or force_reload:
        _CACHE = _parse_multi()
    return _CACHE


def get_provider_display_names() -> List[str]:
    return [p.token for p in load_providers()]


def get_provider_by_name(name: str) -> Provider | None:
    for p in load_providers():
        if p.token == name:
            return p
    return None


def ensure_any_provider():
    """若没有任何 provider, 抛出友好错误."""
    if not load_providers():
        raise ValueError(
            "未检测到 API 配置: 请在 .env 中设置 OPENAI_API_KEY= 或 使用 OPENAI_PROVIDERS=... 多组配置"
        )


__all__ = [
    "Provider",
    "load_providers",
    "get_provider_display_names",
    "get_provider_by_name",
    "ensure_any_provider",
]
