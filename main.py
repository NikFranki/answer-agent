# main.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from tools import search_tool, wiki_tool, save_file_tool

load_dotenv()


# 定义结构化输出模型
class ResearchResponse(BaseModel):
    topic: str = Field(description="研究的主题")
    summary: str = Field(description="针对该主题生成的详细总结报告")
    sources: List[str] = Field(description="研究过程中使用的参考来源列表")
    tools_used: List[str] = Field(description="在执行任务时调用的工具名称")


tools = [search_tool, wiki_tool, save_file_tool]

SYSTEM_PROMPT = (
    "你是一个极其专业的AI科研助手。你的任务是解答用户的提问，并在必要时使用工具进行调研。"
    "收集到足够信息后立即停止搜索，不要重复搜索相似内容。"
)

# Step 1 — 负责调工具搜集信息的 Agent（thinking 与工具调用不兼容，必须关闭）
llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

# Step 2 — 负责整理结构化输出的 LLM（with_structured_output 底层用 tool_choice，不兼容 thinking）
llm_structured = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}},
).with_structured_output(ResearchResponse)


if __name__ == "__main__":
    chat_history = []  # 累积对话历史，格式：[{role, content}, ...]
    print("AI 科研助手已启动，输入 'quit' 退出\n")

    while True:
        user_query = input("您想要研究什么主题？: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("quit", "exit", "退出", "q"):
            print("再见！")
            break

        # --- Phase 1: Agent 调工具搜集原始信息 ---
        print("\n[系统提示] Agent 开始思考并查阅资料...\n")
        messages = []
        tools_called = []

        try:
            for chunk in agent.stream(
                {"messages": chat_history + [{"role": "user", "content": user_query}]},
                stream_mode="updates",
                config={"recursion_limit": 10},
            ):
                for node, update in chunk.items():
                    msgs = update.get("messages", [])
                    messages.extend(msgs)
                    if node == "tools":
                        for msg in msgs:
                            tool_name = getattr(msg, "name", "")
                            if tool_name:
                                tools_called.append(tool_name)
                                print(f"  🔧 调用工具: {tool_name}")
                    elif node == "agent":
                        for msg in msgs:
                            thinking = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                            if thinking:
                                print(f"  💭 推理: {thinking}\n")
                            calls = getattr(msg, "tool_calls", [])
                            for call in calls:
                                print(f"  🤔 决定调用: {call.get('name', '')}（{call.get('args', {})}）")
        except Exception as e:
            # GraphRecursionError 是正常兜底（工具调用达到上限），直接进入 Phase 2
            # 其他异常才提示用户
            if "GraphRecursionError" not in type(e).__name__ and "recursion" not in str(e).lower():
                print(f"\n[错误] 调研过程出现异常: {e}")

        if not messages:
            print("[错误] 没有收集到任何信息，请检查网络或 API Key。\n")
            continue

        # --- Phase 1.5: 显示模型推理过程（thinking 模式，无工具） ---
        raw_content = "\n\n".join(
            getattr(m, "content", "") for m in messages if getattr(m, "content", "")
        )
        format_prompt = (
            f"用户的研究问题是：{user_query}\n\n"
            f"以下是调研过程收集到的原始信息：\n{raw_content}\n\n"
            "请根据以上信息，整理出结构化的研究报告。"
        )

        print("\n[系统提示] 模型正在推理...\n")
        thinking_llm = ChatDeepSeek(
            model="deepseek-v4-flash",
            extra_body={"thinking": {"type": "enabled"}},
        )
        thinking_msg = thinking_llm.invoke(format_prompt)
        reasoning = thinking_msg.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            print("💭 推理过程:")
            print(reasoning)
            print()

        # --- Phase 2: 单次 LLM 调用，整理成结构化输出 ---
        print("\n[系统提示] 正在整理结构化结果...\n")

        final: ResearchResponse = llm_structured.invoke(format_prompt)
        final = final.model_copy(update={"tools_used": list(dict.fromkeys(tools_called))})

        print("=" * 20 + " 格式化研究结果 " + "=" * 20)
        print(f"【研究主题】: {final.topic}")
        print(f"【详细报告】: {final.summary}")
        print(f"【参考来源】: {final.sources}")
        print(f"【调用工具】: {final.tools_used}\n")

        # 把本轮对话存入历史，供下一轮使用
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": final.summary})
