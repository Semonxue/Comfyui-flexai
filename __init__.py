"""Plugin entry for Comfyui-flexai.

Provides:
  - V3 image node  (flexai:openai_image, registered via comfy_entrypoint)
  - V3 video node  (flexai:openai_video, registered via comfy_entrypoint; only
                    when the running ComfyUI exposes the VIDEO type)
  - V1 text node   (flexai:openai_text, injected on_load from the extension)

IMPORTANT: ComfyUI's boot code short-circuits — once __init__.py has
NODE_CLASS_MAPPINGS the comfy_entrypoint branch is *never* reached.
Therefore the V3 nodes go through comfy_entrypoint and the text node is
injected manually in on_load().
"""
__version__ = "1.3.1"

import os
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def load_node_module(module_rel_path: str):
    module_fs_path = os.path.join(current_dir, 'nodes', f'{module_rel_path}.py')
    module_name = module_rel_path.replace('/', '.')
    spec = importlib.util.spec_from_file_location(module_name, module_fs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load FlexAI node modules.
image_module = load_node_module('flexai/openai_image')
text_module = load_node_module('flexai/openai_text')
video_module = load_node_module('flexai/openai_video')
OpenAITextNode = text_module.OpenAITextNode


def _v3_node_classes():
    nodes = [image_module.OpenAIImageNode]
    # The video node needs `comfy_api.latest.io.Video` (VIDEO output type).
    video_cls = getattr(video_module, "OpenAIVideoNode", None)
    if video_cls is not None:
        nodes.append(video_cls)
    return nodes


# ---- V3 registration via comfy_entrypoint ----
try:
    from comfy_api.latest import ComfyExtension
    _V3_OK = True
except ImportError:
    ComfyExtension = None
    _V3_OK = False


if _V3_OK:

    class FlexAIExtension(ComfyExtension):
        async def on_load(self):
            """Inject the V1 text node so existing workflows keep working."""
            try:
                import nodes
                nodes.NODE_CLASS_MAPPINGS["flexai:openai_text"] = OpenAITextNode
                nodes.NODE_DISPLAY_NAME_MAPPINGS["flexai:openai_text"] = "OpenAI Text (FlexAI)"
            except Exception:
                pass  # best-effort; the text node is non-critical

        async def get_node_list(self):
            return _v3_node_classes()


async def comfy_entrypoint():
    if not _V3_OK:
        return None
    return FlexAIExtension()


# ---- Fallback for older ComfyUI without comfy_api ----
# The image node module already defines a V1 fallback class, so adding
# NODE_CLASS_MAPPINGS here works for both. The video node is intentionally
# skipped: the VIDEO output type does not exist on those ComfyUI versions.
if not _V3_OK:
    NODE_CLASS_MAPPINGS = {
        "flexai:openai_image": image_module.OpenAIImageNode,
        "flexai:openai_text": OpenAITextNode,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "flexai:openai_image": "OpenAI Image (FlexAI)",
        "flexai:openai_text": "OpenAI Text (FlexAI)",
    }


__all__ = ['__version__', 'comfy_entrypoint']
