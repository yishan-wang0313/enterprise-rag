"""
test_env.py
-----------
这个脚本用于检查当前开发环境是否已经准备好。

你可以在项目根目录运行：
python test_env.py

它会做两件事：
1. 检查依赖库是否已经安装
2. 检查 OPENAI_API_KEY 是否已经配置
"""


def check(name: str, import_str: str) -> None:
    """
    尝试导入一个模块，并打印检查结果。

    参数：
    - name: 给用户看的名字，更容易理解
    - import_str: 实际导入的 Python 模块名
    """
    try:
        # exec() 会动态执行一段 Python 代码。
        # 这里用它来测试某个模块能不能成功 import。
        exec(f"import {import_str}")
        print(f"OK {name}")
    except ImportError:
        print(f"X {name} 未安装，运行: pip install {import_str}")


# 逐个检查本项目依赖的关键库。
check("Streamlit", "streamlit")
check("LangChain", "langchain")
check("LangChain OpenAI", "langchain_openai")
check("ChromaDB", "chromadb")
check("python-docx", "docx")
check("openpyxl", "openpyxl")
check("sentence-transformers", "sentence_transformers")
check("python-dotenv", "dotenv")


print("\n检查 API Key...")

# 从 config.py 中读取最终配置结果。
# 这样可以顺便验证 .env 是否被正确加载。
from config import OPENAI_API_KEY

if OPENAI_API_KEY:
    print("OK API Key 已配置")
else:
    print("X API Key 未配置，请检查 .env 文件")
