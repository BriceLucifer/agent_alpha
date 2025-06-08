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
import difflib

class AdvancedWakeWordDetector:
    """
    高级唤醒词检测器
    支持真正的唤醒词识别，使用 faster-whisper 进行语音识别
    """
    
    def __init__(self, 
                 wake_words: List[str] = None,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 vad_mode: int = 3,
                 volume_threshold: float = 0.003,
                 confidence_threshold: float = 0.7,
                 debug_mode: bool = True):
        
        # 首先设置日志 - 这必须是第一步
        self.logger = logging.getLogger(__name__)
        if debug_mode:
            self.logger.setLevel(logging.INFO)
        
        # 设置基本属性
        self.debug_mode = debug_mode
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.volume_threshold = volume_threshold
        self.confidence_threshold = confidence_threshold
        
        # 控制变量 - 早期初始化
        self.is_listening = False
        self.stop_event = threading.Event()
        self.wake_callback = None
        
        # 默认唤醒词列表
        if wake_words is None:
            wake_words = [
                # 主要唤醒词
                "你好助手", "您好助手", "小助手", "嘿助手", "智能助手",
                "你好帮手", "您好帮手", "小帮手", "嘿帮手", "智能帮手",
                
                # 简化版本
                "助手", "帮手", "小助理", "小ai", "小AI",
                
                # 可能的识别变体
                "消助手", "笑助手", "小主手", "小竹手",
                "你好主手", "您好主手", "你好竹手", "您好竹手",
                
                # 英文变体
                "hey assistant", "hi assistant", "hello assistant",
                "assistant", "ai assistant"
            ]
        
        # 转换为小写并去重
        self.wake_words = list(set([word.lower().strip() for word in wake_words]))
        
        # 音频检测参数
        self.min_speech_duration = 0.8  # 最小语音时长
        self.max_speech_duration = 5.0  # 最大语音时长
        self.pause_threshold = 0.5      # 静音检测阈值
        
        # 初始化VAD
        try:
            self.vad = webrtcvad.Vad(vad_mode)
            self.logger.info("VAD 初始化成功")
        except Exception as e:
            self.logger.error(f"VAD 初始化失败: {e}")
            raise
        
        # 初始化Whisper模型
        try:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            self.logger.info("使用 Whisper base 模型")
        except Exception as e:
            self.logger.warning(f"base 模型加载失败，使用 tiny 模型: {e}")
            try:
                self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
                self.logger.info("使用 Whisper tiny 模型")
            except Exception as e2:
                self.logger.error(f"Whisper 模型加载完全失败: {e2}")
                raise
        
        # 音频相关
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = None
            self.logger.info("PyAudio 初始化成功")
        except Exception as e:
            self.logger.error(f"PyAudio 初始化失败: {e}")
            raise
        
        # 音频缓冲区
        self.audio_buffer = []
        self.buffer_duration = 4.0  # 保持4秒音频
        self.max_buffer_size = int(self.sample_rate * self.buffer_duration / self.chunk_size)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 统计信息
        self.detection_count = 0
        self.false_positive_count = 0
        
        self.logger.info(f"🎯 唤醒词检测器初始化完成")
        self.logger.info(f"📝 唤醒词列表: {', '.join(self.wake_words[:5])}{'...' if len(self.wake_words) > 5 else ''}")
        self.logger.info(f"🔊 音量阈值: {self.volume_threshold}, 最小语音时长: {self.min_speech_duration}s")
        
    def set_wake_callback(self, callback: Callable):
        """设置唤醒回调函数"""
        self.wake_callback = callback
        self.logger.info("✅ 唤醒回调函数已设置")
    
    def _calculate_volume(self, audio_data: bytes) -> float:
        """计算音频音量"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            
            # 计算RMS音量
            mean_square = np.mean(audio_array.astype(np.float64)**2)
            if mean_square < 0 or np.isnan(mean_square) or np.isinf(mean_square):
                return 0.0
            
            volume = float(np.sqrt(mean_square)) / 32768.0
            return max(0.0, min(1.0, volume))
            
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"音量计算失败: {e}")
            return 0.0
    
    def _is_speech(self, audio_data: bytes) -> bool:
        """检测是否为语音"""
        try:
            if len(audio_data) >= 320:  # VAD需要至少320字节
                return self.vad.is_speech(audio_data[:320], self.sample_rate)
            return False
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"语音检测失败: {e}")
            return False
    
    def _fuzzy_match_wake_word(self, text: str, threshold: float = 0.75) -> tuple[bool, str, float]:
        """模糊匹配唤醒词"""
        text = text.strip().lower()
        best_match = ""
        best_similarity = 0.0
        
        for wake_word in self.wake_words:
            # 精确匹配
            if wake_word in text or text in wake_word:
                return True, wake_word, 1.0
            
            # 模糊匹配
            similarity = difflib.SequenceMatcher(None, text, wake_word).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = wake_word
        
        if best_similarity >= threshold:
            return True, best_match, best_similarity
        
        return False, best_match, best_similarity
    
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
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # 检查识别结果
            detected_texts = []
            for segment in segments:
                text = segment.text.strip()
                detected_texts.append(text)
                
                # 模糊匹配检查
                is_match, matched_word, similarity = self._fuzzy_match_wake_word(text)
                
                if self.debug_mode:
                    self.logger.info(f"🎤 识别文本: '{text}'")
                    if is_match:
                        self.logger.info(f"✅ 匹配唤醒词: '{matched_word}' (相似度: {similarity:.2f})")
                
                if is_match:
                    self.detection_count += 1
                    return True
            
            if self.debug_mode and detected_texts:
                self.logger.info(f"❌ 未匹配唤醒词，识别文本: {detected_texts}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"唤醒词检测失败: {e}")
            return False
    
    def _listen_for_wake_word(self):
        """监听唤醒词的主循环"""
        try:
            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.logger.info(f"🎧 开始监听唤醒词...")
            
            # 状态变量
            speech_detected = False
            speech_start_time = 0
            silence_start_time = 0
            debug_counter = 0
            
            while not self.stop_event.is_set():
                try:
                    # 读取音频数据
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # 计算音量和语音检测
                    volume = self._calculate_volume(audio_data)
                    is_speech = self._is_speech(audio_data)
                    
                    # 调试信息（每50次循环打印一次）
                    debug_counter += 1
                    if self.debug_mode and debug_counter % 50 == 0:
                        self.logger.info(f"📊 音量: {volume:.4f}, 语音: {'是' if is_speech else '否'}, 阈值: {self.volume_threshold}")
                    
                    # 检测语音开始
                    if volume > self.volume_threshold and is_speech:
                        if not speech_detected:
                            speech_detected = True
                            speech_start_time = time.time()
                            self.audio_buffer = []
                            if self.debug_mode:
                                self.logger.info("🎤 检测到语音开始...")
                        
                        # 重置静音计时
                        silence_start_time = 0
                        
                        # 添加到音频缓冲区
                        self.audio_buffer.append(audio_data)
                        
                        # 限制缓冲区大小
                        if len(self.audio_buffer) > self.max_buffer_size:
                            self.audio_buffer.pop(0)
                    
                    # 检测语音结束
                    elif speech_detected:
                        # 开始计时静音
                        if silence_start_time == 0:
                            silence_start_time = time.time()
                        
                        # 检查静音时长
                        silence_duration = time.time() - silence_start_time
                        speech_duration = time.time() - speech_start_time
                        
                        # 语音结束条件：静音超过阈值 或 语音时长超过最大值
                        if (silence_duration > self.pause_threshold or 
                            speech_duration > self.max_speech_duration):
                            
                            if speech_duration >= self.min_speech_duration:
                                if self.debug_mode:
                                    self.logger.info(f"🔍 语音结束，时长: {speech_duration:.2f}s，开始检查唤醒词...")
                                
                                # 在线程池中检查唤醒词
                                future = self.executor.submit(self._check_wake_word, self.audio_buffer.copy())
                                
                                try:
                                    if future.result(timeout=3.0):  # 3秒超时
                                        # 检测到唤醒词
                                        if self.wake_callback:
                                            self.logger.info("🎯 唤醒词检测成功，触发回调...")
                                            self.executor.submit(self._trigger_wake_sync)
                                        
                                        # 短暂休息避免重复触发
                                        time.sleep(2)
                                except Exception as e:
                                    self.logger.error(f"唤醒词检查失败: {e}")
                            else:
                                if self.debug_mode:
                                    self.logger.info(f"⏱️ 语音时长过短: {speech_duration:.2f}s < {self.min_speech_duration}s")
                            
                            # 重置状态
                            speech_detected = False
                            silence_start_time = 0
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
                self.logger.info("🔇 音频流已关闭")
    
    def _trigger_wake_sync(self):
        """同步触发唤醒回调"""
        try:
            self.logger.info("🚀 开始执行唤醒回调...")
            if self.wake_callback:
                if asyncio.iscoroutinefunction(self.wake_callback):
                    # 异步回调
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.wake_callback())
                        self.logger.info("✅ 异步回调执行完成")
                    finally:
                        loop.close()
                else:
                    # 同步回调
                    self.wake_callback()
                    self.logger.info("✅ 同步回调执行完成")
            else:
                self.logger.warning("⚠️ 未设置唤醒回调函数")
        except Exception as e:
            self.logger.error(f"❌ 唤醒回调执行失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
    
    def start_listening(self):
        """开始监听唤醒词"""
        if self.is_listening:
            self.logger.warning("⚠️ 已经在监听中")
            return
        
        self.is_listening = True
        self.stop_event.clear()
        
        # 在新线程中启动监听
        self.listen_thread = threading.Thread(target=self._listen_for_wake_word, daemon=True)
        self.listen_thread.start()
        
        self.logger.info("🎧 高级唤醒词检测已启动")
    
    def stop_listening(self):
        """停止监听唤醒词"""
        if not self.is_listening:
            return
        
        self.stop_event.set()
        self.is_listening = False
        
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=5)
        
        self.logger.info("🔇 高级唤醒词检测已停止")
    
    def add_wake_word(self, word: str):
        """添加新的唤醒词"""
        word = word.lower().strip()
        if word not in self.wake_words:
            self.wake_words.append(word)
            self.logger.info(f"➕ 添加唤醒词: {word}")
    
    def remove_wake_word(self, word: str):
        """移除唤醒词"""
        word = word.lower().strip()
        if word in self.wake_words:
            self.wake_words.remove(word)
            self.logger.info(f"➖ 移除唤醒词: {word}")
    
    def get_statistics(self) -> dict:
        """获取检测统计信息"""
        return {
            "detection_count": self.detection_count,
            "false_positive_count": self.false_positive_count,
            "wake_words_count": len(self.wake_words),
            "is_listening": self.is_listening
        }
    
    def update_settings(self, **kwargs):
        """更新检测设置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"🔧 更新设置: {key} = {value}")
    
    def __del__(self):
        """析构函数，清理资源"""
        self.stop_listening()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'audio'):
            self.audio.terminate()