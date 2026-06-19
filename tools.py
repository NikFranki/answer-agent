# tools.py
from datetime import datetime

from langchain_core.tools import tool

# 联网请求超时时间（秒），避免代理没开或网络不通时整个 agent 卡死
NETWORK_TIMEOUT = 10

# ddgs 库不会自动读取系统的 http_proxy/https_proxy 环境变量，
# 它只认自己专属的 DDGS_PROXY 环境变量，或者显式传入的 proxy 参数。
# 如果你本地有代理（比如 Clash/V2ray 之类），把端口改成你自己的代理端口。
# 如果没有代理需求，把这一行改成 PROXY = None 即可。
PROXY = None


# 1. 联网搜索工具
@tool
def search(query: str) -> str:
    """当需要查询最新的网络资讯、实时发生的事件或通用的网页搜索时使用。

    Args:
        query: 要搜索的关键词或问题。
    """
    try:
        from ddgs import DDGS

        with DDGS(proxy=PROXY, timeout=NETWORK_TIMEOUT) as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"搜索失败（可能是网络无法访问 DuckDuckGo，建议检查代理设置）: {e}"

    if not results:
        return "没有找到相关搜索结果。"

    formatted = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        formatted.append(f"标题: {title}\n摘要: {body}\n链接: {href}")
    return "\n\n".join(formatted)


# 2. 维基百科工具
@tool
def wikipedia_search(query: str) -> str:
    """当需要查询某个概念、人物、事件的百科类背景知识时使用维基百科搜索。

    Args:
        query: 要查询的词条或主题。
    """
    try:
        import os

        import wikipedia

        # wikipedia 库底层用 requests，会读取 http_proxy/https_proxy 环境变量。
        # 这里显式设置一次，确保不依赖外部终端是否正确 export 过。
        if PROXY:
            os.environ.setdefault("http_proxy", PROXY)
            os.environ.setdefault("https_proxy", PROXY)

        wikipedia.set_lang("zh")  # 优先用中文维基百科，访问速度可能更友好
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"维基百科查询失败（可能是网络无法访问 Wikipedia，建议检查代理设置）: {e}"


# 3. 自定义 Python 函数工具：将研究报告保存到本地文本
@tool
def save_text_to_file(data: str, file_name: str = "research_output.txt") -> str:
    """专门用于将最终生成的深入研究报告或复杂数据保存进本地的文本文件中。

    Args:
        data: 需要保存的研究报告正文内容，必须是字符串。
        file_name: 保存的文件名，默认为 research_output.txt。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"=== AI 研究助理报告 ===\n时间戳: {timestamp}\n\n{data}")
    return f"成功将报告保存至文件: {file_name}"


# 保持与原项目一致的命名，方便 main.py 直接导入
search_tool = search
wiki_tool = wikipedia_search
save_file_tool = save_text_to_file