import json
import pyaudio
# import vosk  # 移除这行
import asyncio
# import whisper  # 移除这行，使用 faster-whisper
from faster_whisper import WhisperModel
import numpy as np
import sounddevice as sd
import tempfile
import asyncio
from typing import Optional, Callable
import logging
import os
import tempfile
import wave
from pathlib import Path

class SpeechToText:
    """
    基于 Vosk 的离线语音转文字服务
    完全兼容原有接口，适合 Jetson NX 等资源受限设备
    """
    
    def __init__(self, model_path: str = "vosk-model-cn-0.22", language: str = "zh-CN"):
        """
        初始化 Vosk STT 服务
        
        Args:
            model_path: Vosk 模型路径
            language: 语言代码（保持兼容性）
        """
        self.language = language
        self.sample_rate = 16000
        self.logger = logging.getLogger(__name__)
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            self.logger.error(f"Vosk 模型未找到: {model_path}")
            self.logger.info("请下载模型: wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip && unzip vosk-model-cn-0.22.zip")
            raise FileNotFoundError(f"Vosk 模型未找到: {model_path}")
        
        # 初始化 Vosk 模型和识别器
        try:
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.logger.info(f"Vosk STT 初始化成功，模型: {model_path}")
        except Exception as e:
            self.logger.error(f"Vosk 初始化失败: {e}")
            raise
        
        # 音频配置
        self.chunk_size = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # 识别器参数（保持兼容性）
        self.energy_threshold = 300
        self.dynamic_energy_threshold = True
        self.pause_threshold = 0.8
        self.phrase_threshold = 0.3
    
    async def listen_and_recognize(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        监听麦克风并识别语音（兼容原有接口）
        
        Args:
            timeout: 等待语音开始的超时时间
            phrase_time_limit: 单次录音的最大时长
        
        Returns:
            str: 识别的文本，失败返回None
        """
        try:
            self.logger.info("开始监听语音...")
            
            def _listen_and_recognize():
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                # 重置识别器
                self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
                
                frames_count = int(self.sample_rate / self.chunk_size * phrase_time_limit)
                speech_detected = False
                
                try:
                    for i in range(frames_count):
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        
                        if self.recognizer.AcceptWaveform(data):
                            result = json.loads(self.recognizer.Result())
                            text = result.get('text', '').strip()
                            if text:
                                speech_detected = True
                                return text
                    
                    # 获取最终结果
                    if not speech_detected:
                        final_result = json.loads(self.recognizer.FinalResult())
                        text = final_result.get('text', '').strip()
                        return text if text else None
                    
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    
                return None
            
            result = await asyncio.to_thread(_listen_and_recognize)
            
            if result:
                self.logger.info(f"识别结果: {result}")
                return result
            else:
                self.logger.warning("未检测到语音")
                return None
                
        except Exception as e:
            self.logger.error(f"语音监听失败: {e}")
            return None
    
    async def recognize_file(self, file_path: str) -> Optional[str]:
        """
        识别音频文件中的语音（兼容原有接口）
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            str: 识别的文本，失败返回None
        """
        try:
            def _recognize_file():
                wf = wave.open(file_path, 'rb')
                
                # 检查音频格式
                if (wf.getnchannels() != 1 or 
                    wf.getsampwidth() != 2 or 
                    wf.getframerate() != self.sample_rate):
                    self.logger.error(
                        f"音频格式不支持: {wf.getnchannels()}声道, "
                        f"{wf.getsampwidth()*8}位, {wf.getframerate()}Hz. "
                        f"需要: 1声道, 16位, {self.sample_rate}Hz"
                    )
                    wf.close()
                    return None
                
                # 重置识别器
                recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
                results = []
                
                try:
                    while True:
                        data = wf.readframes(self.chunk_size)
                        if len(data) == 0:
                            break
                        
                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            text = result.get('text', '').strip()
                            if text:
                                results.append(text)
                    
                    # 获取最终结果
                    final_result = json.loads(recognizer.FinalResult())
                    final_text = final_result.get('text', '').strip()
                    if final_text:
                        results.append(final_text)
                    
                finally:
                    wf.close()
                
                full_text = ' '.join(results).strip()
                return full_text if full_text else None
            
            result = await asyncio.to_thread(_recognize_file)
            
            if result:
                self.logger.info(f"文件识别结果: {result}")
            else:
                self.logger.warning("文件识别失败")
            
            return result
            
        except Exception as e:
            self.logger.error(f"音频文件识别失败: {e}")
            return None
    
    async def record_audio(self, duration: float = 5.0, save_path: Optional[str] = None) -> str:
        """
        录制音频到文件（兼容原有接口）
        
        Args:
            duration: 录制时长（秒）
            save_path: 保存路径，不指定则使用临时文件
        
        Returns:
            str: 音频文件路径
        """
        if save_path is None:
            save_path = tempfile.mktemp(suffix=".wav")
        
        def _record():
            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            rate = self.sample_rate
            
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk
            )
            
            frames = []
            
            for _ in range(0, int(rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 保存音频文件
            wf = wave.open(save_path, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return save_path
        
        file_path = await asyncio.to_thread(_record)
        self.logger.info(f"音频已保存到: {file_path}")
        return file_path
    
    async def start_streaming(self, callback: Callable[[str], None]) -> tuple:
        """
        启动流式语音识别
        
        Args:
            callback: 识别结果回调函数
        
        Returns:
            (stream, pyaudio_instance): 用于后续关闭
        """
        p = pyaudio.PyAudio()
        
        # 重置识别器
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        
        def audio_callback(in_data, frame_count, time_info, status):
            try:
                if self.recognizer.AcceptWaveform(in_data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    if text:
                        # 异步执行回调
                        asyncio.create_task(self._async_callback(callback, text))
            except Exception as e:
                self.logger.error(f"流式识别错误: {e}")
            
            return (None, pyaudio.paContinue)
        
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        
        stream.start_stream()
        self.logger.info("流式识别已启动")
        
        return stream, p
    
    async def _async_callback(self, callback: Callable[[str], None], text: str):
        """异步执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(text)
            else:
                await asyncio.to_thread(callback, text)
        except Exception as e:
            self.logger.error(f"回调执行失败: {e}")
    
    def stop_streaming(self, stream, p):
        """停止流式识别"""
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
            p.terminate()
            self.logger.info("流式识别已停止")
        except Exception as e:
            self.logger.error(f"停止流式识别失败: {e}")

# 使用示例
async def main():
    """使用示例"""
    try:
        # 初始化（确保已下载模型）
        stt = SpeechToText(model_path="vosk-model-cn-0.22")
        
        # 单次识别
        print("请说话（5秒内）...")
        text = await stt.listen_and_recognize(timeout=5.0)
        if text:
            print(f"识别结果: {text}")
        else:
            print("未识别到语音")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先下载 Vosk 中文模型:")
        print("wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip")
        print("unzip vosk-model-cn-0.22.zip")
    except Exception as e:
        print(f"运行错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())


import numpy as np
import sounddevice as sd
import tempfile
import asyncio
import threading
import queue
import time
from typing import Optional, Callable
from faster_whisper import WhisperModel

class SpeechToText:
    """
    基于 faster-whisper 的语音转文字服务
    """
    
    def __init__(self, model_size: str = "base", language: str = "zh", device: str = "auto"):
        """
        初始化 Whisper STT 服务
        
        Args:
            model_size: Whisper 模型大小 (tiny, base, small, medium, large, large-v2, large-v3)
            language: 语言代码 (zh, en, auto)
            device: 设备类型 (cpu, cuda, auto)
        """
        self.language = language
        self.sample_rate = 16000
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.model_size = model_size
        
        # 调试信息：初始化开始
        self.logger.info(f"🚀 开始初始化 STT 服务")
        self.logger.info(f"📊 配置参数: 模型={model_size}, 语言={language}, 设备={device}")
        
        # 自动检测设备
        self.device = self._detect_device(device)
        self.compute_type = self._get_compute_type()
        
        # 初始化 Whisper 模型
        try:
            start_time = time.time()
            self.logger.info(f"🔄 正在加载 Whisper 模型: {model_size}")
            self.logger.info(f"🖥️  使用设备: {self.device}, 计算类型: {self.compute_type}")
            
            self.model = WhisperModel(
                model_size, 
                device=self.device, 
                compute_type=self.compute_type,
                download_root=os.path.expanduser("~/.cache/whisper")
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ Whisper 模型加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            self.logger.error(f"❌ Whisper 初始化失败: {e}")
            self.logger.error(f"💡 建议检查网络连接或手动下载模型")
            raise
        
        # 音频配置
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # 语音检测参数
        self.energy_threshold = 300
        self.pause_threshold = 1.0
        self.phrase_threshold = 0.3
        self.min_audio_length = 0.5  # 最小音频长度（秒）
        
        # 统计信息
        self.stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'total_audio_time': 0.0,
            'total_processing_time': 0.0
        }
        
        self.logger.info(f"🎯 STT 服务初始化完成")
        self._log_system_info()
    
    def _detect_device(self, device: str) -> str:
        """自动检测最佳设备"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info(f"🚀 检测到 CUDA 支持，GPU: {torch.cuda.get_device_name()}")
                    return "cuda"
                else:
                    self.logger.info(f"💻 使用 CPU 设备")
                    return "cpu"
            except ImportError:
                self.logger.info(f"💻 PyTorch 未安装，使用 CPU 设备")
                return "cpu"
        else:
            self.logger.info(f"🎯 手动指定设备: {device}")
            return device
    
    def _get_compute_type(self) -> str:
        """根据设备获取最佳计算类型"""
        if self.device == "cuda":
            return "float16"
        else:
            return "int8"
    
    def _log_system_info(self):
        """记录系统信息"""
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            self.logger.info(f"💾 系统信息: CPU核心={cpu_count}, 内存={memory.total//1024//1024//1024}GB")
        except ImportError:
            self.logger.debug("psutil 未安装，跳过系统信息记录")
    
    def _calculate_audio_volume(self, audio_data: bytes) -> float:
        """计算音频音量"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return 0.0
            return float(np.sqrt(np.mean(audio_array.astype(np.float32)**2)))
        except Exception as e:
            self.logger.warning(f"⚠️ 音量计算失败: {e}")
            return 0.0
    
    def _save_debug_audio(self, audio_data: bytes, prefix: str = "debug") -> str:
        """保存调试音频文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.wav"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            self.logger.debug(f"🎵 调试音频已保存: {filepath}")
            return filepath
        except Exception as e:
            self.logger.warning(f"⚠️ 保存调试音频失败: {e}")
            return ""
    
    async def listen_and_recognize(self, timeout: float = 5.0, phrase_time_limit: float = 10.0, 
                                 save_debug_audio: bool = False) -> Optional[str]:
        """
        监听麦克风并识别语音
        
        Args:
            timeout: 等待语音开始的超时时间
            phrase_time_limit: 单次录音的最大时长
            save_debug_audio: 是否保存调试音频
        
        Returns:
            str: 识别的文本，失败返回None
        """
        recognition_id = self.stats['total_recognitions'] + 1
        self.logger.info(f"🎤 开始语音识别 #{recognition_id}")
        self.logger.debug(f"📋 参数: timeout={timeout}s, phrase_limit={phrase_time_limit}s")
        
        start_time = time.time()
        
        try:
            def _record_audio():
                """录制音频的同步函数"""
                self.logger.debug(f"🔧 初始化音频设备...")
                p = pyaudio.PyAudio()
                
                # 列出可用的音频设备
                self._log_audio_devices(p)
                
                stream = p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                frames = []
                recording = False
                silence_count = 0
                max_silence = int(self.sample_rate / self.chunk_size * self.pause_threshold)
                max_frames = int(self.sample_rate / self.chunk_size * phrase_time_limit)
                
                self.logger.debug(f"🎯 开始监听，最大帧数: {max_frames}, 最大静音: {max_silence}")
                
                try:
                    for i in range(max_frames):
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        
                        # 计算音量
                        volume = self._calculate_audio_volume(data)
                        
                        if i % 50 == 0:  # 每50帧记录一次音量
                            self.logger.debug(f"🔊 音量: {volume:.2f}, 阈值: {self.energy_threshold}")
                        
                        if volume > self.energy_threshold:
                            if not recording:
                                self.logger.info(f"🎙️ 检测到语音开始，音量: {volume:.2f}")
                                recording = True
                            frames.append(data)
                            silence_count = 0
                        elif recording:
                            frames.append(data)
                            silence_count += 1
                            
                            # 检查是否应该停止录音
                            if silence_count > max_silence:
                                silence_duration = silence_count * self.chunk_size / self.sample_rate
                                self.logger.info(f"🔇 检测到静音 {silence_duration:.2f}s，停止录音")
                                break
                    
                    if frames:
                        # 合并音频数据
                        audio_data = b''.join(frames)
                        audio_duration = len(frames) * self.chunk_size / self.sample_rate
                        self.logger.info(f"📊 录音完成，时长: {audio_duration:.2f}s, 帧数: {len(frames)}")
                        
                        # 检查最小音频长度
                        if audio_duration < self.min_audio_length:
                            self.logger.warning(f"⚠️ 音频时长过短: {audio_duration:.2f}s < {self.min_audio_length}s")
                            return None
                        
                        return audio_data
                    else:
                        self.logger.warning(f"⚠️ 未录制到任何音频")
                        return None
                    
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    self.logger.debug(f"🔧 音频设备已释放")
            
            # 在线程中录制音频
            self.logger.debug(f"🔄 开始录音线程...")
            audio_data = await asyncio.to_thread(_record_audio)
            
            if audio_data:
                # 保存调试音频
                if save_debug_audio:
                    self._save_debug_audio(audio_data, f"recognition_{recognition_id}")
                
                # 转换为numpy数组
                self.logger.debug(f"🔄 转换音频格式...")
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # 使用Whisper识别
                self.logger.info(f"🧠 开始 Whisper 识别...")
                transcribe_start = time.time()
                
                segments, info = self.model.transcribe(
                    audio_float,
                    language=self.language if self.language != "auto" else None,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                transcribe_time = time.time() - transcribe_start
                self.logger.info(f"⚡ Whisper 识别完成，耗时: {transcribe_time:.2f}s")
                self.logger.debug(f"📊 识别信息: 语言={info.language}, 概率={info.language_probability:.3f}")
                
                # 提取文本
                text_parts = []
                for i, segment in enumerate(segments):
                    text = segment.text.strip()
                    if text:
                        self.logger.debug(f"📝 片段 {i+1}: [{segment.start:.2f}s-{segment.end:.2f}s] {text}")
                        text_parts.append(text)
                
                result = ' '.join(text_parts).strip()
                
                # 更新统计信息
                total_time = time.time() - start_time
                audio_duration = len(audio_data) / (self.sample_rate * 2)  # 16-bit = 2 bytes
                
                self.stats['total_recognitions'] += 1
                self.stats['total_audio_time'] += audio_duration
                self.stats['total_processing_time'] += total_time
                
                if result:
                    self.stats['successful_recognitions'] += 1
                    self.logger.info(f"✅ 识别成功 #{recognition_id}: '{result}'")
                    self.logger.info(f"📊 总耗时: {total_time:.2f}s, 音频: {audio_duration:.2f}s, 处理: {transcribe_time:.2f}s")
                    return result
                else:
                    self.stats['failed_recognitions'] += 1
                    self.logger.warning(f"❌ 识别失败 #{recognition_id}: 未识别到有效文本")
                    return None
            else:
                self.stats['total_recognitions'] += 1
                self.stats['failed_recognitions'] += 1
                self.logger.warning(f"❌ 识别失败 #{recognition_id}: 未录制到音频")
                return None
                
        except Exception as e:
            self.stats['total_recognitions'] += 1
            self.stats['failed_recognitions'] += 1
            total_time = time.time() - start_time
            self.logger.error(f"❌ 语音识别异常 #{recognition_id}: {e}")
            self.logger.error(f"🔍 异常详情: {type(e).__name__}, 耗时: {total_time:.2f}s")
            return None
    
    def _log_audio_devices(self, p: pyaudio.PyAudio):
        """记录可用的音频设备"""
        try:
            device_count = p.get_device_count()
            self.logger.debug(f"🎧 检测到 {device_count} 个音频设备:")
            
            for i in range(device_count):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    self.logger.debug(f"  📱 设备 {i}: {info['name']} (输入通道: {info['maxInputChannels']})")
        except Exception as e:
            self.logger.debug(f"⚠️ 获取音频设备信息失败: {e}")
    
    async def recognize_file(self, file_path: str) -> Optional[str]:
        """
        识别音频文件中的语音
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            str: 识别的文本，失败返回None
        """
        recognition_id = self.stats['total_recognitions'] + 1
        self.logger.info(f"📁 开始文件识别 #{recognition_id}: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"❌ 文件不存在: {file_path}")
            return None
        
        file_size = os.path.getsize(file_path)
        self.logger.debug(f"📊 文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"🔄 开始转录文件...")
            segments, info = self.model.transcribe(
                file_path,
                language=self.language if self.language != "auto" else None,
                vad_filter=True
            )
            
            transcribe_time = time.time() - start_time
            self.logger.info(f"⚡ 文件转录完成，耗时: {transcribe_time:.2f}s")
            self.logger.debug(f"📊 识别信息: 语言={info.language}, 概率={info.language_probability:.3f}")
            
            text_parts = []
            total_duration = 0
            
            for i, segment in enumerate(segments):
                text = segment.text.strip()
                if text:
                    self.logger.debug(f"📝 片段 {i+1}: [{segment.start:.2f}s-{segment.end:.2f}s] {text}")
                    text_parts.append(text)
                    total_duration = max(total_duration, segment.end)
            
            result = ' '.join(text_parts).strip()
            
            # 更新统计信息
            self.stats['total_recognitions'] += 1
            self.stats['total_audio_time'] += total_duration
            self.stats['total_processing_time'] += transcribe_time
            
            if result:
                self.stats['successful_recognitions'] += 1
                self.logger.info(f"✅ 文件识别成功 #{recognition_id}: '{result}'")
                self.logger.info(f"📊 音频时长: {total_duration:.2f}s, 处理耗时: {transcribe_time:.2f}s")
                return result
            else:
                self.stats['failed_recognitions'] += 1
                self.logger.warning(f"❌ 文件识别失败 #{recognition_id}: 未识别到有效文本")
                return None
                
        except Exception as e:
            self.stats['total_recognitions'] += 1
            self.stats['failed_recognitions'] += 1
            processing_time = time.time() - start_time
            self.logger.error(f"❌ 文件识别异常 #{recognition_id}: {e}")
            self.logger.error(f"🔍 异常详情: {type(e).__name__}, 耗时: {processing_time:.2f}s")
            return None
    
    async def record_audio(self, duration: float = 5.0, save_path: Optional[str] = None) -> str:
        """
        录制音频到文件
        
        Args:
            duration: 录制时长（秒）
            save_path: 保存路径，不指定则使用临时文件
        
        Returns:
            str: 音频文件路径
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(tempfile.gettempdir(), f"recording_{timestamp}.wav")
        
        self.logger.info(f"🎙️ 开始录音，时长: {duration}s, 保存到: {save_path}")
        
        def _record():
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            frames_to_record = int(self.sample_rate / self.chunk_size * duration)
            
            self.logger.debug(f"🔄 录制 {frames_to_record} 帧...")
            
            for i in range(frames_to_record):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                if i % 50 == 0:  # 每50帧记录一次进度
                    progress = (i + 1) / frames_to_record * 100
                    self.logger.debug(f"📊 录制进度: {progress:.1f}%")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 保存音频文件
            with wave.open(save_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            return save_path
        
        try:
            file_path = await asyncio.to_thread(_record)
            file_size = os.path.getsize(file_path)
            self.logger.info(f"✅ 录音完成: {file_path}, 大小: {file_size / 1024:.1f} KB")
            return file_path
        except Exception as e:
            self.logger.error(f"❌ 录音失败: {e}")
            raise
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_recognitions'] > 0:
            stats['success_rate'] = stats['successful_recognitions'] / stats['total_recognitions'] * 100
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_recognitions']
        else:
            stats['success_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def log_stats(self):
        """记录统计信息"""
        stats = self.get_stats()
        self.logger.info(f"📊 STT 统计信息:")
        self.logger.info(f"   总识别次数: {stats['total_recognitions']}")
        self.logger.info(f"   成功次数: {stats['successful_recognitions']}")
        self.logger.info(f"   失败次数: {stats['failed_recognitions']}")
        self.logger.info(f"   成功率: {stats['success_rate']:.1f}%")
        self.logger.info(f"   总音频时长: {stats['total_audio_time']:.1f}s")
        self.logger.info(f"   总处理时间: {stats['total_processing_time']:.1f}s")
        self.logger.info(f"   平均处理时间: {stats['avg_processing_time']:.2f}s")
    
    def __del__(self):
        """析构函数，记录最终统计信息"""
        try:
            if hasattr(self, 'stats') and self.stats['total_recognitions'] > 0:
                self.logger.info(f"🏁 STT 服务结束")
                self.log_stats()
        except:
            pass

# 兼容性别名
class STT(SpeechToText):
    """向后兼容的别名"""
    pass

# 测试函数
async def test_stt():
    """测试STT功能"""
    print("🧪 开始 STT 测试...")
    
    try:
        # 初始化服务
        # 将模型从 tiny 升级到 base 或 small
        stt_service = SpeechToText(model_size="base", language="zh")
        # 或者使用更大的模型获得更好效果
        # stt_service = SpeechToText(model_size="small", language="zh")
        print("\n🎤 请说话（5秒内）...")
        result = await stt_service.listen_and_recognize(
            timeout=5.0, 
            phrase_time_limit=5.0,
            save_debug_audio=True
        )
        
        if result:
            print(f"✅ 识别结果: {result}")
        else:
            print("❌ 未识别到语音")
        
        # 显示统计信息
        print("\n📊 测试统计:")
        stt_service.log_stats()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stt())