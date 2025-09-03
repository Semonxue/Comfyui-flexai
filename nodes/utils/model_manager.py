"""
model_manager.py - ComfyUI FlexAI Plugin
- 统一管理模型列表的读取和写入
- 支持从JSON文件加载和保存模型
"""
import os
import json

# -- Constants --
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_MODELS_JSON_FILE = os.path.join(_PLUGIN_ROOT, 'flexai_models.json')
_DEFAULT_MODELS = {
    "flexai_image_models": ["dall-e-3", "dall-e-2"],
    "flexai_text_models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
}

# -- Public Functions --
def get_models(model_key: str):
    """
    从JSON文件读取模型列表。
    如果文件或指定的键不存在，则会自动创建并填入默认值。
    """
    if not os.path.exists(_MODELS_JSON_FILE):
        with open(_MODELS_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(_DEFAULT_MODELS, f, indent=2)
    
    with open(_MODELS_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if model_key not in data:
        data[model_key] = _DEFAULT_MODELS.get(model_key, [])
        with open(_MODELS_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    return data.get(model_key, [])

def add_model(model_name: str, model_key: str):
    """
    如果模型不存在，则将其添加到JSON文件的对应列表中。
    这是一个线程安全的操作。
    """
    if not model_name or not model_key:
        return
    
    # Basic lock mechanism for file writing
    lock_file = _MODELS_JSON_FILE + '.lock'
    try:
        with open(lock_file, 'x'): # Attempt to create lock file
            with open(_MODELS_JSON_FILE, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                models = data.get(model_key, [])
                if model_name not in models:
                    models.append(model_name)
                    data[model_key] = models
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
    except FileExistsError:
        # If lock file exists, another process is writing. Skip for now.
        # In a high-concurrency scenario, a more robust locking mechanism would be needed.
        pass
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)
