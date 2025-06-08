import asyncio
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack

# MCP 客户端相关导入
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI

# 环境变量加载相关
from dotenv import load_dotenv
import os

# 本地服务导入
import tts
import stt
from wake_word import WakeWordDetector
from knowledge_base import KnowledgeBase

load_dotenv()

class MCPClient:
    """
    使用 OpenAI 格式 API 的 MCP 客户端类
    处理 MCP 服务器连接和 OpenAI 兼容 API 的交互
    集成本地 TTS 和 STT 功能
    """
    
    def __init__(self):
        # MCP 客户端会话
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # 使用 DeepSeek API 配置（删除重复配置）
        self.llm_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 本地服务初始化
        self.stt_service = stt.SpeechToText()
        self.wake_detector = WakeWordDetector()
        self.knowledge_base = KnowledgeBase()
        
        # 对话历史
        self.conversation_history = []
        
        # 设置唤醒回调
        self.wake_detector.set_wake_callback(self.handle_wake_word)
    
    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # 建立连接
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        # 创建会话
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio_transport[0], stdio_transport[1])
        )
        
        # 初始化会话
        await self.session.initialize()
        
        print(f"✅ 已连接到 MCP 服务: {server_script_path}")
    
    async def handle_wake_word(self):
        """处理唤醒词检测"""
        print("🎤 检测到唤醒词，开始监听...")
        
        # 播放提示音
        await tts.tts_play("我在听，请说话")
        
        # 开始语音识别
        user_input = await self.stt_service.listen_and_recognize(timeout=10.0)
        
        if user_input:
            print(f"👤 用户说: {user_input}")
            
            # 处理用户查询
            response = await self.process_query(user_input)
            
            # 播放回答
            if response:
                await tts.tts_play(response)
        else:
            await tts.tts_play("抱歉，我没有听清楚")
    
    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        try:
            # 添加到对话历史
            self.conversation_history.append({"role": "user", "content": query})
            
            # 检查知识库
            kb_results = await self.knowledge_base.search_documents(query, top_k=3)
            
            # 构建系统提示
            system_prompt = "你是一个智能助手。"
            if kb_results:
                kb_context = "\n".join([doc["content"] for doc in kb_results])
                system_prompt += f"\n\n相关知识库信息:\n{kb_context}"
            
            # 获取可用工具
            tools = []
            if self.session:
                tools_result = await self.session.list_tools()
                tools = tools_result.tools
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history[-10:]  # 保留最近10轮对话
            ]
            
            # 如果有工具，使用工具调用
            if tools:
                # 转换工具格式为OpenAI格式
                openai_tools = self._convert_tools_to_openai_format(tools)
                
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                
                # 处理工具调用
                if message.tool_calls:
                    return await self._handle_tool_calls(message, messages)
                else:
                    final_response = message.content
            else:
                # 直接对话
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                final_response = response.choices[0].message.content
            
            # 添加助手回复到历史
            self.conversation_history.append({"role": "assistant", "content": final_response})
            
            return final_response
            
        except Exception as e:
            print(f"❌ 处理查询失败: {e}")
            return "抱歉，处理您的请求时出现了错误。"
    
    def _convert_tools_to_openai_format(self, mcp_tools):
        """将MCP工具格式转换为OpenAI格式"""
        openai_tools = []
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools
    
    async def _handle_tool_calls(self, message, messages):
        """处理工具调用"""
        # 添加助手消息到历史
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [{
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments
                }
            } for call in message.tool_calls]
        })
        
        # 执行工具调用
        for tool_call in message.tool_calls:
            try:
                # 解析参数
                args = json.loads(tool_call.function.arguments)
                
                # 调用MCP工具
                result = await self.session.call_tool(
                    tool_call.function.name,
                    args
                )
                
                # 添加工具结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.content)
                })
                
            except Exception as e:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"工具调用失败: {str(e)}"
                })
        
        # 获取最终回复
        final_response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    async def start_voice_interaction(self):
        """启动语音交互模式"""
        print("🎙️ 启动语音交互模式...")
        print("💡 说出唤醒词开始对话，按 Ctrl+C 退出")
        
        # 启动唤醒词检测
        self.wake_detector.start_listening()
        
        try:
            # 保持运行
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 退出语音交互模式")
        finally:
            self.wake_detector.stop_listening()
    
    async def text_interaction(self):
        """文本交互模式"""
        print("💬 启动文本交互模式...")
        print("💡 输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    continue
                
                print("🤖 思考中...")
                response = await self.process_query(user_input)
                print(f"🤖 助手: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
        
        print("👋 再见！")
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

# 主函数
async def func_call_main():
    if len(sys.argv) < 2:
        print("用法: python fun_call.py <mcp_server_script> [mode]")
        print("mode: voice (语音模式) 或 text (文本模式，默认)")
        return
    
    server_script = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "text"
    
    client = MCPClient()
    
    try:
        # 连接到MCP服务
        await client.connect_to_server(server_script)
        
        # 根据模式启动交互
        if mode.lower() == "voice":
            await client.start_voice_interaction()
        else:
            await client.text_interaction()
            
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(func_call_main())
