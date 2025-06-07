# this file is for edge-tts
import asyncio
import platform
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
import edge_tts

async def tts_play(text: str, save_path: str = None) -> str:
    """
    文本转语音并实时播放（跨平台）
    
    :param text: 需要合成的文本内容
    :param save_path: 可选的文件保存路径（不传则不保存）
    :return: 生成的音频文件路径
    """
    # 生成语音数据流
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural")
    
    # 创建临时文件（如果不需要保存）
    temp_file = None
    file_path = save_path or Path(NamedTemporaryFile(suffix=".mp3", delete=False).name)
    
    try:
        # 保存音频数据
        await communicate.save(str(file_path))
        
        # 跨平台播放逻辑
        def _play():
            system = platform.system()
            try:
                if system == "Darwin":  # macOS
                    subprocess.run(["afplay", str(file_path)], check=True)
                elif system == "Windows":  # Windows
                    subprocess.run(
                        ["cmd", "/c", "start", "/B", str(file_path)], 
                        shell=True, 
                        check=True
                    )
                else:  # Linux及其他系统
                    subprocess.run(["aplay", str(file_path)], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"播放失败: {e}") from e
            finally:
                if not save_path:  # 删除临时文件
                    Path(file_path).unlink(missing_ok=True)
        
        # 在独立线程中执行播放
        await asyncio.to_thread(_play)
        
        return str(file_path)
    
    except Exception as e:
        # 异常时清理临时文件
        if not save_path and Path(file_path).exists():
            Path(file_path).unlink(missing_ok=True)
        raise RuntimeError(f"语音处理失败: {e}") from e

# 使用示例
async def main():
    # 仅播放不保存
    await tts_play("欢迎使用语音合成系统")
    
    # 播放并保存
    await tts_play("已为您保存日志文件", save_path="output.mp3")

if __name__ == "__main__":
    asyncio.run(main())