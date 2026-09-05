"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )
    # 最终答案的最大中文字符数，通过 Prompt 控制生成长度（非强制截断）
    max_answer_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_ANSWER_CHARS", "800"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "15"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.08"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "20"))
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "15"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "15"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.65"))
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.40"))
    )
    item_name_score_gap: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SCORE_GAP", "0.08"))
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "3"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )
    # 字符串相似度阈值（0~1），用于辅助判断商品名是否匹配
    item_name_string_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_STRING_SIMILARITY_THRESHOLD", "0.80"))
    )
    # 融合分数阈值：向量分数与字符串相似度融合后的判定阈值
    item_name_combined_threshold: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_COMBINED_THRESHOLD", "0.70"))
    )
    # 字符串相似度在融合中的权重
    item_name_string_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_STRING_WEIGHT", "0.40"))
    )
    # 无商品名时是否启用宽范围内部检索兜底
    enable_broad_search_fallback: bool = field(
        default_factory=lambda: os.getenv("ENABLE_BROAD_SEARCH_FALLBACK", "true").lower() in ("true", "1", "yes")
    )
    # 宽范围内部检索未命中时，是否启用 Web 搜索兜底（企业场景默认关闭）
    enable_web_fallback_when_no_item: bool = field(
        default_factory=lambda: os.getenv("ENABLE_WEB_FALLBACK_WHEN_NO_ITEM", "false").lower() in ("true", "1", "yes")
    )
    # 宽范围检索返回的最少文档数阈值，低于此值认为没有有效结果
    broad_search_min_docs: int = field(
        default_factory=lambda: int(os.getenv("BROAD_SEARCH_MIN_DOCS", "1"))
    )
    # 开放性问题（general_question）在未命中商品名时，是否直接让大模型自主回答（而非走检索兜底）
    enable_direct_llm_for_general_question: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DIRECT_LLM_FOR_GENERAL_QUESTION", "false").lower() in ("true", "1", "yes")
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPEN_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config