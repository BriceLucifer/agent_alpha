# main.py
from fun_call import func_call_main
import asyncio
import sys
from agent_controller import AgentController

async def main():
    """主程序入口"""
    # 解析命令行参数
    mode = "text"  # 默认文本模式
    mcp_servers = []
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in ["voice", "text"]:
                mode = arg
            elif arg.endswith(".py") or arg.endswith(".js"):
                mcp_servers.append(arg)
    
    # 如果没有指定MCP服务器，使用默认的天气服务
    if not mcp_servers:
        mcp_servers = ["./weather.py"]
    
    print("🚀 启动智能助手系统...")
    print(f"📋 模式: {mode}")
    print(f"🔧 MCP服务器: {', '.join(mcp_servers)}")
    
    controller = AgentController()
    
    try:
        # 初始化系统
        await controller.initialize(mcp_servers=mcp_servers)
        
        # 启动对应模式
        if mode == "voice":
            # 使用修改后的语音模式（无唤醒词）
            await controller.start_voice_mode()
        else:
            await controller.start_text_mode()
        
    except Exception as e:
        print(f"❌ 系统错误: {e}")
    finally:
        await controller.cleanup()

if __name__ == "__main__":
    # 只执行一个主函数
    asyncio.run(main())  # 或者 asyncio.run(func_call_main())