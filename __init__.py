"""Plugin entry for Comfyui-flexai.

Provides:
  - V3 image node  (FlexAI_OpenAIImageNode, registered via comfy_entrypoint)
  - V1 text node   (flexai:openai_text, injected on_load from the extension)

IMPORTANT: ComfyUI's boot code short-circuits — once __init__.py has
NODE_CLASS_MAPPINGS the comfy_entrypoint branch is *never* reached.
Therefore the image node goes through comfy_entrypoint and the text
node is injected manually in on_load().
"""
__version__ = "1.0.8"

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
OpenAITextNode = text_module.OpenAITextNode


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
            return [image_module.OpenAIImageNode]


async def comfy_entrypoint():
    if not _V3_OK:
        return None
    return FlexAIExtension()


# ---- Fallback for older ComfyUI without comfy_api ----
# The image node module already defines a V1 fallback class, so adding
# NODE_CLASS_MAPPINGS here works for both.
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