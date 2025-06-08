import pyaudio
import webrtcvad
import threading
import time
import logging
from typing import Callable, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import wave

class AdvancedWakeWordDetector:
    """
    高级唤醒词检测器
    支持真正的唤醒词识别
    """
    
    def __init__(self, 
                 wake_words: List[str] = ["小助手", "你好助手", "嘿助手"],
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 volume_threshold: float = 0.01,  # 从0.02降低到0.01，使更敏感
                 vad_mode: int = 3,
                 confidence_threshold: float = 0.7):
        
        self.wake_words = [word.lower() for word in wake_words]
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.volume_threshold = volume_threshold
        self.confidence_threshold = confidence_threshold
        
        # 初始化VAD
        self.vad = webrtcvad.Vad(vad_mode)
        
        # 初始化Whisper用于唤醒词识别
        self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        
        # 音频相关
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 控制变量
        self.is_listening = False
        self.stop_event = threading.Event()
        self.wake_callback = None
        
        # 音频缓冲区（用于存储最近的音频）
        self.audio_buffer = []
        self.buffer_duration = 3.0  # 保持3秒音频
        self.max_buffer_size = int(self.sample_rate * self.buffer_duration / self.chunk_size)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"唤醒词设置: {', '.join(self.wake_words)}")
        
    def set_wake_callback(self, callback: Callable):
        """设置唤醒回调函数"""
        self.wake_callback = callback
    
    def _calculate_volume(self, audio_data: bytes) -> float:
        """计算音频音量 - 安全版本"""
        try:
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
            if len(audio_data) >= 320:  # VAD需要至少320字节
                vad_data = audio_data[:320]
                return self.vad.is_speech(vad_data, self.sample_rate)
            return False
        except Exception as e:
            self.logger.error(f"语音检测失败: {e}")
            return False
    
    def _check_wake_word(self, audio_buffer: List[bytes]) -> bool:
        """检查音频缓冲区中是否包含唤醒词"""
        try:
            if not audio_buffer:
                return False
            
            # 合并音频数据
            combined_audio = b''.join(audio_buffer)
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)
            
            if len(audio_array) < self.sample_rate * 0.5:  # 至少0.5秒
                return False
            
            # 归一化音频
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # 使用Whisper进行语音识别
            segments, info = self.whisper_model.transcribe(
                audio_float, 
                language="zh",
                condition_on_previous_text=False
            )
            
            # 检查识别结果中是否包含唤醒词
            for segment in segments:
                text = segment.text.strip().lower()
                self.logger.debug(f"识别文本: {text}")
                
                for wake_word in self.wake_words:
                    if wake_word in text:
                        self.logger.info(f"检测到唤醒词: {wake_word} (在文本: {text})")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"唤醒词检测失败: {e}")
            return False
    
    def _listen_for_wake_word(self):
        try:
            # 初始化变量
            speech_detected = False
            speech_start_time = 0
            
            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.debug(f"开始监听唤醒词: {', '.join(self.wake_words)}")
            
            while not self.stop_event.is_set():
                try:
                    # 读取音频数据
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # 计算音量
                    volume = self._calculate_volume(audio_data)
                    self.logger.debug(f"当前音量: {volume:.4f} (阈值: {self.volume_threshold})")
                    
                    # 检查是否为语音
                    is_speech = self._is_speech(audio_data)
                    self.logger.debug(f"语音检测: {'是' if is_speech else '否'}")
                    
                    if volume > self.volume_threshold and is_speech:
                        if not speech_detected:
                            speech_detected = True
                            speech_start_time = time.time()
                            self.audio_buffer = []
                            self.logger.debug("开始检测语音...")
                        
                        # 添加到音频缓冲区
                        self.audio_buffer.append(audio_data)
                        
                        # 限制缓冲区大小
                        if len(self.audio_buffer) > self.max_buffer_size:
                            self.audio_buffer.pop(0)
                    
                    elif speech_detected:
                        # 语音结束，检查是否包含唤醒词
                        speech_duration = time.time() - speech_start_time
                        
                        if speech_duration > 0.5:  # 至少0.5秒的语音
                            self.logger.debug(f"语音结束，时长: {speech_duration:.2f}秒，开始检查唤醒词...")
                            
                            # 在线程池中检查唤醒词
                            future = self.executor.submit(self._check_wake_word, self.audio_buffer.copy())
                            
                            try:
                                if future.result(timeout=2.0):  # 2秒超时
                                    # 检测到唤醒词，触发回调
                                    if self.wake_callback:
                                        self.executor.submit(self._trigger_wake_sync)
                                    
                                    # 短暂休息避免重复触发
                                    time.sleep(3)
                            except Exception as e:
                                self.logger.error(f"唤醒词检查超时或失败: {e}")
                        
                        speech_detected = False
                        self.audio_buffer = []
                    
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
        
        self.logger.info("高级唤醒词检测已启动")
    
    def stop_listening(self):
        """停止监听唤醒词"""
        if not self.is_listening:
            return
        
        self.stop_event.set()
        self.is_listening = False
        
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=5)
        
        self.logger.info("高级唤醒词检测已停止")
    
    def add_wake_word(self, word: str):
        """添加新的唤醒词"""
        word = word.lower()
        if word not in self.wake_words:
            self.wake_words.append(word)
            self.logger.info(f"添加唤醒词: {word}")
    
    def remove_wake_word(self, word: str):
        """移除唤醒词"""
        word = word.lower()
        if word in self.wake_words:
            self.wake_words.remove(word)
            self.logger.info(f"移除唤醒词: {word}")
    
    def __del__(self):
        """析构函数，清理资源"""
        self.stop_listening()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'audio'):
            self.audio.terminate()