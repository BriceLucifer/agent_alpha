import whisper
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import asyncio

class WhisperSTT:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
    
    async def transcribe_realtime(self, duration=3.0):
        """实时录音并转录"""
        def record_audio():
            audio = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            return audio.flatten()
        
        # 异步录音
        audio = await asyncio.to_thread(record_audio)
        
        # 异步转录
        def transcribe():
            result = self.model.transcribe(audio, language="zh")
            return result["text"].strip()
        
        text = await asyncio.to_thread(transcribe)
        return text