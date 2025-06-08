# Agent Alpha

## 项目介绍

Agent Alpha 是一个融合多种AI能力的智能助手系统，支持多轮对话、语音交互、知识库学习和Agent角色扮演。系统基于大语言模型和MCP（Model-Controller-Processor）架构，能够通过API调用实现天气查询等功能，并支持中文语音识别和语音合成。

## 核心功能

- **多模态交互**：支持文本和语音两种交互模式
- **知识库管理**：基于向量数据库的知识存储与检索系统
- **工具调用**：通过MCP架构实现天气查询等API功能调用
- **角色扮演**：支持多种预设角色，如智能助手、天气专家、知识导师等
- **语音识别**：集成Whisper和Faster-Whisper模型，支持中文语音识别
- **语音合成**：使用Edge TTS实现自然流畅的中文语音合成

## 技术架构

- **核心控制器**：AgentController负责协调各模块工作
- **语音模块**：集成STT（语音识别）和TTS（语音合成）服务
- **知识库**：基于ChromaDB和Sentence Transformers的向量数据库
- **MCP客户端**：处理与MCP服务器的通信，支持工具调用
- **大语言模型**：默认使用DeepSeek API，支持自定义配置

## 安装指南

### 系统要求

- Python 3.13+
- Linux/macOS/Windows系统

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/agent-alpha.git
cd agent-alpha
```

2. 安装依赖

在Linux系统上，可以使用提供的安装脚本：

```bash
chmod +x install.sh
./install.sh
```

或手动安装依赖：

```bash
# 安装系统依赖（Linux）
sudo apt update
sudo apt install -y build-essential aplay portaudio19-dev python3-dev

# 安装Python依赖
uv sync
```

3. 配置环境变量

复制示例配置文件并编辑：

```bash
cp .env.example .env
```

编辑`.env`文件，填入必要的API密钥和配置项：

```
# DeepSeek API 配置
DEEPSEEK_API_KEY="your_deepseek_api_key_here"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"

# 天气API配置
OPENWEATHER_API_KEY="your_openweather_api_key_here"
```

## 使用方法

### 文本模式

```bash
uv run python main.py text
```

### 语音模式

```bash
uv run python main.py voice
```

### 指定MCP服务器

```bash
uv run python main.py text ./weather.py ./other_service.py
```

## 配置选项

在`.env`文件中可以配置以下选项：

### 大语言模型配置

```
DEEPSEEK_API_KEY="your_api_key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"
```

### 语音配置

```
TTS_VOICE="zh-CN-XiaoxiaoNeural"  # 语音合成声音
STT_LANGUAGE="zh-CN"              # 语音识别语言
```

### 知识库配置

```
KNOWLEDGE_BASE_PATH="./knowledge_base"
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIRECTORY="./chroma_db"
```

### 系统配置

```
LOG_LEVEL="INFO"
MAX_CONVERSATION_HISTORY=50
SESSION_TIMEOUT=3600
```

## 功能扩展

### 添加新的MCP服务

1. 创建新的Python脚本，参考`weather.py`的实现
2. 使用`FastMCP`注册新服务和工具函数
3. 启动时指定新服务脚本路径

### 自定义Agent角色

在`agent_controller.py`的`_load_agent_roles`方法中添加新的角色定义：

```python
"new_role": {
    "name": "新角色名称",
    "personality": "角色个性描述",
    "capabilities": ["能力1", "能力2"]
}
```

## 技术细节

- **语音识别**：使用Faster-Whisper模型进行语音识别
- **语音合成**：使用Edge TTS进行跨平台语音合成
- **知识库**：使用ChromaDB存储向量化的知识片段
- **工具调用**：基于MCP架构实现函数调用能力

## 许可证

[添加您的许可证信息]

## 贡献指南

[添加贡献指南]

## 联系方式

[添加联系方式]

        