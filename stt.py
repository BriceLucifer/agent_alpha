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
import threading
import queue
import time
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
    
    def __init__(self, model_size: str = "base", language: str = "zh"):
        """
        初始化 Whisper STT 服务
        
        Args:
            model_size: Whisper 模型大小 (tiny, base, small, medium, large)
            language: 语言代码
        """
        self.language = language
        self.sample_rate = 16000
        self.logger = logging.getLogger(__name__)
        
        # 初始化 Whisper 模型
        try:
            self.logger.info(f"正在加载Whisper模型: {model_size}")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.logger.info("Whisper模型加载完成")
        except Exception as e:
            self.logger.error(f"Whisper 初始化失败: {e}")
            raise
        
        # 音频配置
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        # 录音参数
        self.energy_threshold = 300
        self.pause_threshold = 1.0
        self.phrase_threshold = 0.3
    
    async def listen_and_recognize(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
        """
        监听麦克风并识别语音
        
        Args:
            timeout: 等待语音开始的超时时间
            phrase_time_limit: 单次录音的最大时长
        
        Returns:
            str: 识别的文本，失败返回None
        """
        try:
            self.logger.info("开始监听语音...")
            
            def _record_audio():
                """录制音频的同步函数"""
                p = pyaudio.PyAudio()
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
                
                try:
                    for i in range(max_frames):
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        
                        # 计算音量
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        volume = np.sqrt(np.mean(audio_array**2))
                        
                        if volume > self.energy_threshold:
                            if not recording:
                                self.logger.debug("开始录音...")
                                recording = True
                            frames.append(data)
                            silence_count = 0
                        elif recording:
                            frames.append(data)
                            silence_count += 1
                            
                            # 检查是否应该停止录音
                            if silence_count > max_silence:
                                self.logger.debug("检测到静音，停止录音")
                                break
                    
                    if frames:
                        # 合并音频数据
                        audio_data = b''.join(frames)
                        return audio_data
                    
                    return None
                    
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
            
            # 在线程中录制音频
            audio_data = await asyncio.to_thread(_record_audio)
            
            if audio_data:
                # 转换为numpy数组
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # 使用Whisper识别
                segments, info = self.model.transcribe(
                    audio_float,
                    language=self.language,
                    condition_on_previous_text=False
                )
                
                # 提取文本
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())
                
                result = ' '.join(text_parts).strip()
                
                if result:
                    self.logger.info(f"识别结果: {result}")
                    return result
                else:
                    self.logger.warning("未识别到有效文本")
                    return None
            else:
                self.logger.warning("未录制到音频")
                return None
                
        except Exception as e:
            self.logger.error(f"语音识别失败: {e}")
            return None
    
    async def recognize_file(self, file_path: str) -> Optional[str]:
        """
        识别音频文件中的语音
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            str: 识别的文本，失败返回None
        """
        try:
            segments, info = self.model.transcribe(
                file_path,
                language=self.language
            )
            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            result = ' '.join(text_parts).strip()
            
            if result:
                self.logger.info(f"文件识别结果: {result}")
                return result
            else:
                self.logger.warning("文件中未识别到有效文本")
                return None
                
        except Exception as e:
            self.logger.error(f"文件识别失败: {e}")
            return None

# 兼容性别名
class STT(SpeechToText):
    """向后兼容的别名"""
    pass

# 测试函数
async def test_stt():
    """测试STT功能"""
    stt_service = SpeechToText()
    
    print("请说话...")
    result = await stt_service.listen_and_recognize(timeout=5.0, phrase_time_limit=5.0)
    
    if result:
        print(f"识别结果: {result}")
    else:
        print("未识别到语音")

if __name__ == "__main__":
    asyncio.run(test_stt())