"""
exceptions.py (第 6 段优化 2)
-----------------------------
统一异常层次 + 友好错误消息。

设计思路:
- RagError 是所有业务异常的基类,自带 user_message(给最终用户看的话)
- 各子类对应 pipeline 里不同的失败场景
- app.py 只 catch RagError,直接展示 user_message,不暴露 traceback
- 底层第三方错误(OpenAIError / ChromaError)在业务边界被包装成 RagError

对比"就地 raise RuntimeError":
- 这里把"错在哪一步"、"该给用户说什么"、"原始异常"三件事分离
- Streamlit 界面友好 + 日志能追溯

> Java 类比:相当于 Spring 里的自定义 `BusinessException` + `@ControllerAdvice`。
"""

from __future__ import annotations


class RagError(Exception):
    """业务异常基类。所有子类必须覆盖 user_message。"""

    user_message: str = "Something went wrong. Please try again."


class ConfigError(RagError):
    """配置缺失或错误(比如没设 OPENAI_API_KEY)。"""

    user_message = "Configuration error: please check that OPENAI_API_KEY is set in your .env file."


class DocumentError(RagError):
    """文档解析失败(格式不支持、文件损坏)。"""

    user_message = "Failed to process this document. Check that the file is a valid docx / xlsx / txt / md and not corrupted."


class EmptyKnowledgeBaseError(RagError):
    """知识库为空却尝试检索。"""

    user_message = "The knowledge base is empty. Please upload and import a document from the sidebar first."


class RetrievalError(RagError):
    """检索阶段失败(向量库读写异常、CrossEncoder 加载失败等)。"""

    user_message = "Retrieval failed. The model may still be loading, or the database is busy — please try again."


class GenerationError(RagError):
    """LLM 生成阶段失败(API 报错、超时、额度不足)。"""

    user_message = "The AI failed to generate an answer. Possible causes: API quota exceeded, network issue, or timeout."


def friendly_message(exc: Exception) -> str:
    """
    把任意异常翻译成"给用户看的话"。

    - 是 RagError → 用它自带的 user_message,附带原始细节
    - 其他 → 通用兜底
    """

    if isinstance(exc, RagError):
        detail = str(exc)
        if detail and detail != exc.user_message:
            return f"{exc.user_message}\n\n_Details: {detail}_"
        return exc.user_message

    # 未预期的错误,尽量给点线索
    return f"Unexpected error: {type(exc).__name__}: {exc}"
