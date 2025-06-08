# Agent Alpha

在 Jetson NX 上实现的智能对话系统，支持语音唤醒、语音转文字、文字转语音、多轮对话、知识库问答和 Agent 角色扮演函数调用等功能。

## 功能特性

- 🎤 **语音唤醒**: 支持自定义唤醒词检测
- 🗣️ **语音转文字**: 本地语音识别，支持实时和文件识别
- 🔊 **文字转语音**: 本地 TTS 服务，支持多种语音
- 💬 **多轮对话**: 基于 OpenAI 格式 API 的智能对话
- 📚 **知识库**: 向量数据库支持的知识问答
- 🛠️ **函数调用**: MCP 协议支持的工具调用
- 🎭 **角色扮演**: Agent 角色扮演功能

## 安装依赖

```bash
# 安装系统依赖
sudo apt update
sudo apt install build-essential aplay portaudio19-dev python3-dev

# 安装 Python 依赖
uv sync

# 使用说明
# 激活虚拟环境（如果尚未激活）
# source .venv/Scripts/activate

# 运行 MCP 客户端，连接到天气查询 MCP 服务器（示例）
uv run main.py ./weather.py

# 示例问题：
```