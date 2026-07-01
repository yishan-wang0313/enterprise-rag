"""
config.py
---------
这个文件集中管理整个项目的配置。

把配置单独放一个文件有几个好处：
1. 修改参数时更集中，不用到处找。
2. 后续模块都可以统一 import 这些配置。
3. 对初学者来说，更容易理解“哪些值是经常会调整的”。
"""

import os

from dotenv import load_dotenv


# load_dotenv() 会自动读取当前项目中的 .env 文件，
# 并把里面的键值对加载到环境变量中。
# 例如：
# OPENAI_API_KEY=sk-xxxx
load_dotenv()


# =========================
# OpenAI 相关配置
# =========================

# 从环境变量中读取 API Key。
# 这样做比把 key 直接写死在代码里更安全。
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# embedding 模型负责把文本转换成向量。
# 向量会用于后续“相似度检索”。
EMBEDDING_MODEL = "text-embedding-3-small"

# 聊天模型负责基于“检索结果”生成最终答案。
CHAT_MODEL = "gpt-4o-mini"


# =========================
# 向量数据库配置
# =========================

# ChromaDB 的本地存储路径。
# 运行后，Chroma 会在这个目录里自动生成数据库文件。
DB_PATH = "./database/chroma_db"

# collection 可以理解为“一个向量表”或“一个知识库集合”。
# 后续所有文档片段都会被存到这个集合中。
COLLECTION_NAME = "enterprise_docs"


# =========================
# 文本切块配置
# =========================

# 每个文本块的目标大小。
# 块太小，信息容易不完整；
# 块太大，检索时又不够精准。
CHUNK_SIZE = 500

# 相邻文本块之间保留一部分重叠内容，
# 这样可以减少重要信息刚好被切断的情况。
CHUNK_OVERLAP = 50


# =========================
# 检索配置
# =========================

# 向量检索阶段，先取回前 TOP_K 个候选结果。
TOP_K = 20

# 如果后面增加“重排序”模型，
# 就从 TOP_K 个候选里再挑出最相关的前 RERANK_TOP_K 个。
RERANK_TOP_K = 4
