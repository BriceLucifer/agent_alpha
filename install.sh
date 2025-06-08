#!/bin/bash

# 安装系统依赖
sudo apt update
sudo apt install -y build-essential aplay portaudio19-dev python3-dev

# 安装 Python 依赖
uv sync

# 下载 Vosk 中文模型
echo "正在下载 Vosk 中文模型..."
if [ ! -d "vosk-model-cn-0.22" ]; then
    wget https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
    unzip vosk-model-cn-0.22.zip
    rm vosk-model-cn-0.22.zip
    echo "✅ Vosk 模型下载完成"
else
    echo "✅ Vosk 模型已存在"
fi

echo "🚀 安装完成！"
echo "使用方法:"
echo "  文本模式: uv run python main.py text"
echo "  语音模式: uv run python main.py voice"