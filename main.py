# main.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_deepseek import ChatDeepSeek
from tools import search_tool, wiki_tool, save_file_tool

# 1. 加载 .env 中的 API Key（需要在 .env 里配置 DEEPSEEK_API_KEY=sk-xxxx）
load_dotenv()


# 2. 定义我们期望 AI 最终返回的结构化模型
class ResearchResponse(BaseModel):
    topic: str = Field(description="研究的主题")
    summary: str = Field(description="针对该主题生成的详细总结报告")
    sources: List[str] = Field(description="研究过程中使用的参考来源列表")
    tools_used: List[str] = Field(description="在执行任务时调用的工具名称")


# 3. 汇总工具箱列表
tools = [search_tool, wiki_tool, save_file_tool]

# 4. 系统提示词
SYSTEM_PROMPT = (
    "你是一个极其专业的AI科研助手。你的任务是解答用户的提问，并在必要时使用工具进行调研。"
    "工具调用规则：最多调用5次工具，收集到足够信息后立即整理输出，不要重复搜索相似内容。"
)

# 5. 创建具备工具调用能力、且自动产出结构化输出的 Agent 实例
#    deepseek-v4-flash 默认是思考模式（会输出 reasoning_content），
#    而 ToolStrategy 为了强制产出结构化结果，会要求模型强制调用某个工具（tool_choice 强制模式）。
#    DeepSeek 的思考模式和强制 tool_choice 是互斥的，会报 400 "Thinking mode does not support this tool_choice"。
#    所以这里显式传 extra_body 关闭 thinking，用非思考模式跑（速度更快，且这个任务本身不需要复杂推理）。
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(ResearchResponse),
)

if __name__ == "__main__":
    # 6. 获取用户输入的调研指令
    user_query = input("您想要研究什么主题？(例如：帮我研究 LangChain 的应用，并保存到文件中): ")

    # 7. 流式运行，实时打印进度
    print("\n[系统提示] Agent 开始思考并查阅资料...\n")
    result = None
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_query}]},
            stream_mode="updates",
            config={"recursion_limit": 10},
        ):
            for node, update in chunk.items():
                if node == "tools":
                    for msg in update.get("messages", []):
                        tool_name = getattr(msg, "name", "")
                        print(f"  🔧 调用工具: {tool_name}")
                elif node == "agent":
                    for msg in update.get("messages", []):
                        calls = getattr(msg, "tool_calls", [])
                        for call in calls:
                            print(f"  🤔 决定调用: {call.get('name', '')}（{call.get('args', {})}）")
            result = chunk
    except Exception as e:
        print(f"\n[警告] Agent 提前终止: {e}\n正在尝试输出已收集的结果...\n")

    # 8. 从最后一个 chunk 提取结构化输出
    try:
        # stream 结束后从最终状态取结构化结果
        final = None
        for node, update in (result or {}).items():
            if "structured_response" in update:
                final = update["structured_response"]
                break

        if final is None:
            raise ValueError("未找到 structured_response")

        print("\n" + "=" * 20 + " 格式化研究结果 " + "=" * 20)
        print(f"【研究主题】: {final.topic}")
        print(f"【详细报告】: {final.summary}")
        print(f"【参考来源】: {final.sources}")
        print(f"【调用工具】: {final.tools_used}")

    except Exception as e:
        print(f"\n[错误] 结构化结果解析失败。原因: {e}")
        if result:
            for node, update in result.items():
                msgs = update.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    print(f"原始输出: {getattr(last, 'content', last)}")