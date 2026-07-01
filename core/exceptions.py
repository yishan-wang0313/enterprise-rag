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

    user_message: str = "系统出错了,请稍后重试。"


class ConfigError(RagError):
    """配置缺失或错误(比如没设 OPENAI_API_KEY)。"""

    user_message = "配置错误:请检查 .env 文件里的 OPENAI_API_KEY 是否已经设置。"


class DocumentError(RagError):
    """文档解析失败(格式不支持、文件损坏)。"""

    user_message = "文档处理失败。请检查文件格式是否为 docx / xlsx / txt / md,且没有损坏。"


class EmptyKnowledgeBaseError(RagError):
    """知识库为空却尝试检索。"""

    user_message = "知识库还是空的。请先在左侧上传并导入文档,再来提问。"


class RetrievalError(RagError):
    """检索阶段失败(向量库读写异常、CrossEncoder 加载失败等)。"""

    user_message = "检索失败,请稍后重试。可能是模型加载中或数据库繁忙。"


class GenerationError(RagError):
    """LLM 生成阶段失败(API 报错、超时、额度不足)。"""

    user_message = "AI 生成回答失败。可能是 API 额度不足、网络问题或超时。"


def friendly_message(exc: Exception) -> str:
    """
    把任意异常翻译成"给用户看的话"。

    - 是 RagError → 用它自带的 user_message,附带原始细节
    - 其他 → 通用兜底
    """

    if isinstance(exc, RagError):
        detail = str(exc)
        if detail and detail != exc.user_message:
            return f"{exc.user_message}\n\n_详细信息:{detail}_"
        return exc.user_message

    # 未预期的错误,尽量给点线索
    return f"发生了未预期的错误:{type(exc).__name__}: {exc}"
