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

    # 7. 让 Agent 开始运转
    print("\n[系统提示] Agent 开始思考并查阅资料...\n")
    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

    # 8. 提取结构化输出
    try:
        structured_data: ResearchResponse = result["structured_response"]

        print("\n" + "=" * 20 + " 格式化研究结果 " + "=" * 20)
        print(f"【研究主题】: {structured_data.topic}")
        print(f"【详细报告】: {structured_data.summary}")
        print(f"【参考来源】: {structured_data.sources}")
        print(f"【调用工具】: {structured_data.tools_used}")

    except Exception as e:
        print(f"\n[错误] 结构化结果解析失败。原因: {e}")
        last_message = result["messages"][-1]
        print(f"原始未解析输出: {getattr(last_message, 'content', last_message)}")