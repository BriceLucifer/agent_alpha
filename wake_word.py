import pyaudio
import webrtcvad
import threading
import time
import logging
from typing import Callable, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class WakeWordDetector:
    """
    唤醒词检测器 - 重构版
    使用线程池和队列来避免异步冲突
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 320,  # 20ms at 16kHz
                 volume_threshold: float = 0.01,
                 vad_mode: int = 3):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.volume_threshold = volume_threshold
        
        # 初始化VAD
        self.vad = webrtcvad.Vad(vad_mode)
        
        # 音频相关
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 控制变量
        self.is_listening = False
        self.stop_event = threading.Event()
        self.wake_callback = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def set_wake_callback(self, callback: Callable):
        """设置唤醒回调函数"""
        self.wake_callback = callback
    
    def _calculate_volume(self, audio_data: bytes) -> float:
        """计算音频音量 - 修复版"""
        try:
            import numpy as np
            if len(audio_data) == 0:
                return 0.0
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            
            # 计算RMS音量，添加安全检查
            mean_square = np.mean(audio_array.astype(np.float64)**2)
            if mean_square < 0 or np.isnan(mean_square) or np.isinf(mean_square):
                return 0.0
            
            volume = float(np.sqrt(mean_square)) / 32768.0
            return max(0.0, min(1.0, volume))  # 限制在0-1范围内
            
        except Exception as e:
            self.logger.error(f"音量计算失败: {e}")
            return 0.0
    
    def _is_speech(self, audio_data: bytes) -> bool:
        """检测是否为语音"""
        try:
            if len(audio_data) == 320:  # 20ms at 16kHz
                return self.vad.is_speech(audio_data, self.sample_rate)
            return False
        except Exception as e:
            self.logger.error(f"语音检测失败: {e}")
            return False
    
    def _listen_for_wake_word(self):
        """监听唤醒词的主循环 - 在独立线程中运行"""
        try:
            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info("开始监听唤醒词...")
            
            while not self.stop_event.is_set():
                try:
                    # 读取音频数据
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # 计算音量
                    volume = self._calculate_volume(audio_data)
                    
                    # 检查音量阈值
                    if volume > self.volume_threshold:
                        # 检测语音活动
                        if len(audio_data) >= 320:
                            vad_data = audio_data[:320]
                            if self._is_speech(vad_data):
                                self.logger.info(f"检测到语音活动，音量: {volume:.3f}")
                                
                                # 触发唤醒回调
                                if self.wake_callback:
                                    # 使用线程池执行回调，避免阻塞
                                    self.executor.submit(self._trigger_wake_sync)
                                
                                # 短暂休息避免重复触发
                                time.sleep(2)
                    
                    # 短暂休息
                    time.sleep(0.01)
                    
                except Exception as e:
                    self.logger.error(f"音频处理错误: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"监听失败: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def _trigger_wake_sync(self):
        """同步触发唤醒回调"""
        try:
            if self.wake_callback:
                if asyncio.iscoroutinefunction(self.wake_callback):
                    # 创建新的事件循环来运行异步回调
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.wake_callback())
                    finally:
                        loop.close()
                else:
                    self.wake_callback()
        except Exception as e:
            self.logger.error(f"唤醒回调执行失败: {e}")
    
    def start_listening(self):
        """开始监听唤醒词"""
        if self.is_listening:
            self.logger.warning("已经在监听中")
            return
        
        self.is_listening = True
        self.stop_event.clear()
        
        # 在新线程中启动监听
        self.listen_thread = threading.Thread(target=self._listen_for_wake_word)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        self.logger.info("唤醒词检测已启动")
    
    def stop_listening(self):
        """停止监听唤醒词"""
        if not self.is_listening:
            return
        
        self.stop_event.set()
        self.is_listening = False
        
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=5)
        
        self.logger.info("唤醒词检测已停止")
    
    def __del__(self):
        """析构函数，清理资源"""
        self.stop_listening()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'audio'):
            self.audio.terminate()