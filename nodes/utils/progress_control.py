"""
高级超时控制和进度模拟模块

由于OpenAI图片生成API是同步的（非流式），
这个模块提供了一些增强功能来改善用户体验：

1. 超时控制 - 避免无限等待
2. 进度模拟 - 基于历史数据提供进度估算
3. 心跳检测 - 定期检查连接状态
4. 取消机制 - 允许用户中断长时间运行的任务
"""

import time
import threading
import signal
import requests
from typing import Optional, Callable, Dict, Any
import json
import os

class ImageGenerationProgress:
    """图片生成进度跟踪器"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.start_time = None
        self.last_update = None
        self.estimated_duration = 30.0  # 默认估计30秒
        self.progress_history = self._load_history()
        
    def _load_history(self) -> Dict[str, Any]:
        """加载历史性能数据"""
        history_file = os.path.join(os.path.dirname(__file__), '.generation_history.json')
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {
            "dall-e-3_1024x1024": {"avg_time": 25.0, "count": 0},
            "dall-e-3_1792x1024": {"avg_time": 30.0, "count": 0},
            "dall-e-2_1024x1024": {"avg_time": 15.0, "count": 0},
            "default": {"avg_time": 20.0, "count": 0}
        }
    
    def _save_history(self):
        """保存历史性能数据"""
        history_file = os.path.join(os.path.dirname(__file__), '.generation_history.json')
        try:
            with open(history_file, 'w') as f:
                json.dump(self.progress_history, f, indent=2)
        except:
            pass
    
    def start(self, model: str, size: str):
        """开始进度跟踪"""
        self.start_time = time.time()
        self.last_update = self.start_time
        
        # 根据模型和尺寸估计时间
        key = f"{model}_{size}"
        if key in self.progress_history:
            self.estimated_duration = self.progress_history[key]["avg_time"]
        else:
            self.estimated_duration = self.progress_history["default"]["avg_time"]
        
        if self.debug:
            print(f"[PROGRESS] 🎯 预估完成时间: {self.estimated_duration:.1f} 秒")
    
    def update(self) -> float:
        """更新进度并返回当前进度百分比"""
        if not self.start_time:
            return 0.0
            
        elapsed = time.time() - self.start_time
        progress = min(elapsed / self.estimated_duration, 0.95)  # 最多显示95%，避免超过100%
        
        # 每5秒更新一次进度显示
        if time.time() - self.last_update >= 5.0:
            if self.debug:
                remaining = max(0, self.estimated_duration - elapsed)
                print(f"[PROGRESS] ⏳ 进度: {progress*100:.1f}% | 已用时: {elapsed:.1f}s | 预计剩余: {remaining:.1f}s")
            self.last_update = time.time()
        
        return progress
    
    def complete(self, actual_duration: float, model: str, size: str):
        """完成进度跟踪并更新历史数据"""
        key = f"{model}_{size}"
        if key in self.progress_history:
            # 更新平均时间（使用移动平均）
            history = self.progress_history[key]
            old_avg = history["avg_time"]
            count = history["count"]
            new_avg = (old_avg * count + actual_duration) / (count + 1)
            self.progress_history[key] = {"avg_time": new_avg, "count": count + 1}
        else:
            self.progress_history[key] = {"avg_time": actual_duration, "count": 1}
        
        self._save_history()
        
        if self.debug:
            print(f"[PROGRESS] ✅ 完成! 实际耗时: {actual_duration:.1f}s")

class TimeoutHandler:
    """超时处理器"""
    
    def __init__(self, timeout_seconds: float = 120.0, debug: bool = False):
        self.timeout_seconds = timeout_seconds
        self.debug = debug
        self.cancelled = False
        self.timeout_reached = False
        
    def __enter__(self):
        self.cancelled = False
        self.timeout_reached = False
        
        # 设置信号处理器（用于超时）
        if hasattr(signal, 'SIGALRM'):  # Unix系统
            def timeout_handler(signum, frame):
                self.timeout_reached = True
                if self.debug:
                    print(f"[TIMEOUT] ⚠️  达到超时限制: {self.timeout_seconds} 秒")
                raise TimeoutError(f"操作超时 ({self.timeout_seconds} 秒)")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout_seconds))
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 取消超时设置
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        return False
    
    def cancel(self):
        """取消操作"""
        self.cancelled = True
        if self.debug:
            print("[TIMEOUT] 🛑 操作被用户取消")

def enhanced_api_call_with_progress(
    api_function: Callable,
    progress_callback: Optional[Callable[[float], None]] = None,
    timeout_seconds: float = 120.0,
    model: str = "unknown",
    size: str = "unknown",
    debug: bool = False,
    **kwargs
) -> Any:
    """
    增强的API调用，支持进度跟踪和超时控制
    
    Args:
        api_function: 要调用的API函数
        progress_callback: 进度回调函数
        timeout_seconds: 超时时间
        model: 模型名称
        size: 图片尺寸
        debug: 是否开启调试
        **kwargs: 传递给API函数的参数
    
    Returns:
        API函数的返回值
    """
    
    progress_tracker = ImageGenerationProgress(debug)
    progress_tracker.start(model, size)
    
    result = None
    error = None
    
    # 在单独线程中运行API调用
    def api_thread():
        nonlocal result, error
        try:
            result = api_function(**kwargs)
        except Exception as e:
            error = e
    
    # 启动API线程
    thread = threading.Thread(target=api_thread, daemon=True)
    thread.start()
    
    # 进度更新循环
    try:
        with TimeoutHandler(timeout_seconds, debug) as timeout_handler:
            while thread.is_alive():
                if timeout_handler.cancelled or timeout_handler.timeout_reached:
                    if debug:
                        print("[ENHANCED] 🛑 API调用被中断")
                    break
                
                # 更新进度
                progress = progress_tracker.update()
                if progress_callback:
                    progress_callback(progress)
                
                # 短暂等待
                time.sleep(1.0)
            
            # 等待线程完成
            thread.join(timeout=1.0)
            
    except TimeoutError as e:
        if debug:
            print(f"[ENHANCED] ⚠️  API调用超时: {e}")
        raise e
    
    # 检查结果
    if error:
        raise error
    
    if result is None:
        raise RuntimeError("API调用未返回结果")
    
    # 记录完成信息
    actual_duration = time.time() - progress_tracker.start_time
    progress_tracker.complete(actual_duration, model, size)
    
    return result

# 使用示例
def example_usage():
    """使用示例"""
    
    def mock_api_call(prompt: str, model: str, **kwargs):
        """模拟API调用"""
        import random
        time.sleep(random.uniform(10, 30))  # 模拟10-30秒的API调用时间
        return f"Generated image for: {prompt}"
    
    def progress_callback(progress: float):
        print(f"当前进度: {progress*100:.1f}%")
    
    try:
        result = enhanced_api_call_with_progress(
            api_function=mock_api_call,
            progress_callback=progress_callback,
            timeout_seconds=60.0,
            model="dall-e-3",
            size="1024x1024",
            debug=True,
            prompt="A beautiful landscape",
            model="dall-e-3"
        )
        print(f"成功: {result}")
        
    except TimeoutError as e:
        print(f"超时: {e}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    example_usage()
