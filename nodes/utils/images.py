"""Image helper utilities for FlexAI nodes.

Centralizes tensor<->PIL conversions and base64 encoding to reduce duplication.
"""
from __future__ import annotations
import base64
from io import BytesIO
from typing import Tuple
from PIL import Image
import numpy as np
import torch

__all__ = [
    "tensor_to_pil",
    "pil_to_tensor",
    "pil_to_base64",
]

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor (H,W,C) or (B,H,W,C) float32 0-1 to PIL Image.
    If batch provided, take first frame.
    """
    if tensor is None:
        raise ValueError("tensor is None")
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expect 3D tensor (H,W,C), got shape {tuple(tensor.shape)}")
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor (B,H,W,C) float32 0-1."""
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    t = torch.from_numpy(arr)
    return t.unsqueeze(0)

def pil_to_base64(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    """Encode PIL image to data URL base64."""
    buf = BytesIO()
    save_kwargs = {}
    target_fmt = fmt.upper()
    if target_fmt == "JPEG":
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        save_kwargs["quality"] = quality
    img.save(buf, format=target_fmt, **save_kwargs)
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if target_fmt == "JPEG" else f"image/{target_fmt.lower()}"
    return f"data:{mime};base64,{b}"
