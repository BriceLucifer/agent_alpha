import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import AsyncExitStack

# 导入所有功能模块
from fun_call import MCPClient
from stt import SpeechToText
from tts import tts_play
# 删除唤醒词相关导入
# from wake_word import WakeWordDetector
# from advanced_wake_word import AdvancedWakeWordDetector
from knowledge_base import KnowledgeBase
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class AgentController:
    """
    智能助手主控制器
    融合功能：多轮对话、语音识别、知识库、Agent角色扮演
    """
    
    def __init__(self):
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.mcp_client = MCPClient()
        self.stt_service = SpeechToText()
        # 删除唤醒词检测器
        self.knowledge_base = KnowledgeBase()
        
        # OpenAI客户端
        self.llm_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 对话状态管理
        self.conversation_history = []
        self.current_session_id = None
        self.user_profile = {}
        self.agent_roles = self._load_agent_roles()
        self.current_role = "default"
        
        # 系统状态
        self.is_active = False
        # 删除唤醒词回调设置
        
        self.logger.info("智能助手控制器初始化完成")
    
    def _load_agent_roles(self) -> Dict[str, Dict]:
        """加载Agent角色配置"""
        return {
            "default": {
                "name": "智能助手",
                "personality": "我是一个友善、专业的AI助手，能够帮助您解决各种问题。",
                "capabilities": ["对话", "信息查询", "知识问答", "工具调用"]
            },
            "weather_expert": {
                "name": "天气专家",
                "personality": "我是专业的气象分析师，专门提供准确的天气信息和建议。",
                "capabilities": ["天气查询", "气象分析", "出行建议"]
            },
            "knowledge_tutor": {
                "name": "知识导师",
                "personality": "我是您的学习伙伴，擅长解释复杂概念，帮助您学习和理解知识。",
                "capabilities": ["知识讲解", "学习指导", "答疑解惑"]
            },
            "task_assistant": {
                "name": "任务助手",
                "personality": "我是高效的任务管理专家，帮助您规划和执行各种任务。",
                "capabilities": ["任务规划", "时间管理", "效率优化"]
            }
        }
    
    async def initialize(self, mcp_servers: List[str] = None):
        """初始化系统"""
        try:
            # 连接MCP服务器
            if mcp_servers:
                for server in mcp_servers:
                    await self.mcp_client.connect_to_server(server)
                    self.logger.info(f"已连接MCP服务器: {server}")
            
            # 初始化知识库
            await self.knowledge_base.initialize_if_needed()
            
            # 创建新的对话会话
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.is_active = True
            self.logger.info("系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    
    async def handle_wake_event(self):
        """处理唤醒事件 - 重构版"""
        try:
            self.logger.info("🎙️ 检测到唤醒信号")
            
            # 播放唤醒提示
            await tts_play("我在听，请说话")
            
            # 等待一小段时间让TTS播放完成
            await asyncio.sleep(0.5)
            
            # 开始语音识别
            self.logger.info("开始语音识别...")
            user_input = await self.stt_service.listen_and_recognize(timeout=10.0)
            
            if user_input and user_input.strip():
                self.logger.info(f"👤 用户输入: {user_input}")
                
                # 处理用户输入
                response = await self.process_user_input(user_input, mode="voice")
                
                # 语音回复
                if response:
                    self.logger.info(f"🤖 回复: {response}")
                    await tts_play(response)
                else:
                    await tts_play("抱歉，我无法处理您的请求")
            else:
                self.logger.warning("未识别到有效输入")
                # 不要立即回复，给用户一些时间
                await asyncio.sleep(1.0)
                # 只在确实没有检测到任何音频时才提示
                if user_input is None:
                    await tts_play("请稍等，我再试试听您说话")
                else:
                    await tts_play("抱歉，我没有听清楚，请再试一次")
                    
        except Exception as e:
            self.logger.error(f"语音处理失败: {e}")
            await tts_play("抱歉，出现了技术问题，请稍后再试")
    
    async def process_user_input(self, user_input: str, mode: str = "text") -> str:
        """处理用户输入的核心方法"""
        try:
            # 检查是否是角色切换命令
            if user_input.startswith("切换角色") or user_input.startswith("角色"):
                return await self._handle_role_switch(user_input)
            
            # 检查是否是知识库学习命令
            if user_input.startswith("学习") or user_input.startswith("记住"):
                return await self._handle_knowledge_learning(user_input)
            
            # 添加到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "mode": mode
            })
            
            # 搜索相关知识
            knowledge_context = await self._get_knowledge_context(user_input)
            
            # 构建系统提示
            system_prompt = self._build_system_prompt(knowledge_context)
            
            # 获取可用工具
            available_tools = await self._get_available_tools()
            
            # 构建消息
            messages = self._build_messages(system_prompt, user_input)
            
            # 调用LLM
            if available_tools:
                response = await self._call_llm_with_tools(messages, available_tools)
            else:
                response = await self._call_llm_simple(messages)
            
            # 添加助手回复到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "role_used": self.current_role
            })
            
            # 保持对话历史在合理长度
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
            
        except Exception as e:
            self.logger.error(f"处理用户输入失败: {e}")
            return "抱歉，处理您的请求时出现了错误，请稍后再试。"
    
    async def _handle_role_switch(self, user_input: str) -> str:
        """处理角色切换"""
        # 简单的角色识别逻辑
        if "天气" in user_input:
            self.current_role = "weather_expert"
        elif "学习" in user_input or "知识" in user_input:
            self.current_role = "knowledge_tutor"
        elif "任务" in user_input or "计划" in user_input:
            self.current_role = "task_assistant"
        else:
            self.current_role = "default"
        
        role_info = self.agent_roles[self.current_role]
        return f"好的，我现在是{role_info['name']}。{role_info['personality']}"
    
    async def _handle_knowledge_learning(self, user_input: str) -> str:
        """处理知识库学习"""
        try:
            # 提取要学习的内容
            content = user_input.replace("学习", "").replace("记住", "").strip()
            
            if not content:
                return "请告诉我您想让我学习什么内容。"
            
            # 添加到知识库
            doc_id = await self.knowledge_base.add_document(
                content=content,
                metadata={
                    "source": "user_input",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.current_session_id
                }
            )
            
            return f"好的，我已经学习并记住了这个内容。文档ID: {doc_id}"
            
        except Exception as e:
            self.logger.error(f"知识学习失败: {e}")
            return "抱歉，学习过程中出现了错误。"
    
    async def _get_knowledge_context(self, query: str) -> str:
        """获取知识库相关上下文"""
        try:
            # 修复参数名：top_k -> n_results
            results = await self.knowledge_base.search_documents(query, n_results=3)
            return results
        except Exception as e:
            self.logger.error(f"知识库搜索失败: {e}")
            return []
    
    def _build_system_prompt(self, knowledge_context: List[Dict]) -> str:
        """构建系统提示"""
        role_info = self.agent_roles[self.current_role]
        
        prompt = f"""你是{role_info['name']}。{role_info['personality']}

你的主要能力包括：{', '.join(role_info['capabilities'])}

当前对话会话ID: {self.current_session_id}
当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if knowledge_context:
            prompt += "\n\n相关知识库信息：\n"
            for i, doc in enumerate(knowledge_context, 1):
                prompt += f"{i}. {doc['content']}\n"
        
        prompt += "\n\n请根据你的角色特点和相关知识，为用户提供准确、有用的回答。"
        
        return prompt
    
    async def _get_available_tools(self) -> List[Dict]:
        """获取可用工具"""
        tools = []
        try:
            if self.mcp_client.session:
                tools_result = await self.mcp_client.session.list_tools()
                tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in tools_result.tools]
        except Exception as e:
            self.logger.error(f"获取工具列表失败: {e}")
        
        return tools
    
    def _build_messages(self, system_prompt: str, user_input: str) -> List[Dict]:
        """构建消息列表"""
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加最近的对话历史
        recent_history = self.conversation_history[-10:]  # 最近10轮对话
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return messages
    
    async def _call_llm_with_tools(self, messages: List[Dict], tools: List[Dict]) -> str:
        """使用工具调用LLM"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                # 处理工具调用
                return await self._handle_tool_calls(message, messages)
            else:
                return message.content
                
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return "抱歉，AI服务暂时不可用。"
    
    async def _call_llm_simple(self, messages: List[Dict]) -> str:
        """简单LLM调用"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return "抱歉，AI服务暂时不可用。"
    
    async def _handle_tool_calls(self, message, messages: List[Dict]) -> str:
        """处理工具调用"""
        # 添加助手消息
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
                args = json.loads(tool_call.function.arguments)
                result = await self.mcp_client.session.call_tool(
                    tool_call.function.name,
                    args
                )
                
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
    
    async def start_voice_mode(self):
        """启动语音模式 - 直接语音识别的多轮对话"""
        self.logger.info("启动语音交互模式")
        print("🎙️ 语音模式已启动")
        print("💡 直接说话开始对话，按 Ctrl+C 退出")
        print(f"🤖 当前角色: {self.agent_roles[self.current_role]['name']}")
        
        try:
            while self.is_active:
                try:
                    print("\n🎤 请说话...")
                    
                    # 直接进行语音识别，无需唤醒词
                    user_input = await self.stt_service.listen_and_recognize(timeout=10.0)
                    
                    if user_input and user_input.strip():
                        print(f"👤 您说: {user_input}")
                        
                        # 检查退出命令
                        if user_input.lower() in ['退出', '再见', 'quit', 'exit']:
                            await tts_play("再见！")
                            break
                        
                        print("🤖 思考中...")
                        response = await self.process_user_input(user_input, mode="voice")
                        
                        if response:
                            print(f"🤖 {self.agent_roles[self.current_role]['name']}: {response}")
                            await tts_play(response)
                        else:
                            await tts_play("抱歉，我无法处理您的请求")
                    else:
                        print("⚠️ 未识别到有效输入，请重试")
                        
                except asyncio.TimeoutError:
                    print("⏰ 语音识别超时，请重试")
                except Exception as e:
                    print(f"❌ 错误: {e}")
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    print("\n👋 退出语音模式")
        finally:
                print("🔚 语音模式已结束")
    
        self.wake_detector.start_listening()
        
        try:
            while self.is_active:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 退出语音模式")
        finally:
            self.wake_detector.stop_listening()
    
    async def start_text_mode(self):
        """启动文本模式"""
        self.logger.info("启动文本交互模式")
        print("💬 文本模式已启动")
        print("💡 输入消息开始对话，输入 'quit' 退出")
        print(f"🤖 当前角色: {self.agent_roles[self.current_role]['name']}")
        
        while self.is_active:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    continue
                
                print("🤖 思考中...")
                response = await self.process_user_input(user_input, mode="text")
                print(f"🤖 {self.agent_roles[self.current_role]['name']}: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
        
        print("👋 再见！")
    
    async def cleanup(self):
        """清理资源"""
        self.is_active = False
        # 删除唤醒词检测器的停止调用
        await self.mcp_client.cleanup()
        self.logger.info("系统已清理")

# 使用示例
async def main():
    controller = AgentController()
    
    try:
        # 初始化系统
        await controller.initialize(mcp_servers=["./weather.py"])
        
        # 选择模式
        mode = input("选择模式 (voice/text): ").strip().lower()
        
        if mode == "voice":
            await controller.start_voice_mode()
        else:
            await controller.start_text_mode()
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")
    finally:
        await controller.cleanup()

if __name__ == "__main__":
    asyncio.run(main())