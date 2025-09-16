"""Plugin entry for Comfyui_flexai.

Provides the FlexAI series of nodes.
Node Naming Convention:
  - Key: flexai:openai_image / flexai:openai_text
  - Display Name: OpenAI Image (FlexAI) / OpenAI Text (FlexAI)
Category: flexai
"""

__version__ = "1.0.6"

import os
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def load_node_module(module_rel_path: str):
    """Load a node module dynamically from nodes/ subpath.

    module_rel_path: e.g. "flexai/genimage" (without .py)
    """
    module_fs_path = os.path.join(current_dir, 'nodes', f'{module_rel_path}.py')
    module_name = module_rel_path.replace('/', '.')
    spec = importlib.util.spec_from_file_location(module_name, module_fs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load FlexAI nodes
openai_image_module = load_node_module('flexai/openai_image')
openai_text_module = load_node_module('flexai/openai_text')

OpenAIImageNode = openai_image_module.OpenAIImageNode
OpenAITextNode = openai_text_module.OpenAITextNode

NODE_CLASS_MAPPINGS = {
    "flexai:openai_image": OpenAIImageNode,
    "flexai:openai_text": OpenAITextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "flexai:openai_image": "OpenAI Image (FlexAI)",
    "flexai:openai_text": "OpenAI Text (FlexAI)",
}

__all__ = [
    '__version__',
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'OpenAIImageNode',
    'OpenAITextNode'
]
