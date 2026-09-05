import json
import re
from logging import Logger
from typing import Dict, Any, List, Tuple, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import (
    DIRECT_ANSWER_SYSTEM_TEMPLATE,
    DIRECT_ANSWER_USER_TEMPLATE,
    INTENT_SYSTEM_TEMPLATE,
    INTENT_USER_TEMPLATE,
    ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE,
    ITEM_NAME_USER_EXTRACT_TEMPLATE,
)
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class _QueryIntentClassifier:
    """用户意图分类器：规则快速判定 + LLM 兜底。"""

    GREETING = "greeting"
    GOODBYE = "goodbye"
    CHITCHAT = "chitchat"
    ASSISTANT_CAPABILITY = "assistant_capability"
    UNRELATED_QUESTION = "unrelated_question"
    PRODUCT_QUERY = "product_query"
    GENERAL_QUESTION = "general_question"

    _GREETING_KEYWORDS = ["你好", "您好", "hello", "hi", "在吗", "在么", "早上好", "下午好", "晚上好"]
    _GOODBYE_KEYWORDS = ["再见", "拜拜", "bye", "goodbye", "回见", "告辞"]
    _CHITCHAT_KEYWORDS = ["谢谢", "感谢", "你真棒", "辛苦了", "不错", "好的", "OK", "ok"]
    _CAPABILITY_KEYWORDS = ["你能做什么", "你能做啥", "你有什么用", "你有什么功能", "你是谁", "你是做什么的", "介绍一下你自己", "你会什么"]
    _UNRELATED_KEYWORDS = ["天气", "几点", "时间", "日期", "新闻", "笑话"]

    def __init__(self, logger: Logger, config: QueryConfig):
        self._logger = logger
        self._config = config

    def classify(self, original_query: str, formatted_history_str: str) -> str:
        # 1. 先用规则快速判定，降低 LLM 调用成本
        rule_intent = self._rule_classify(original_query)
        if rule_intent:
            self._logger.info(f"规则意图识别结果:{rule_intent}")
            return rule_intent
        # 2. 规则无法判定时，使用 LLM 兜底
        return self._llm_classify(original_query, formatted_history_str)

    def _rule_classify(self, query: str) -> Optional[str]:
        q = query.strip().lower()
        if not q:
            return None
        for keyword in self._GREETING_KEYWORDS:
            if q == keyword.lower() or q.startswith(keyword.lower()):
                return self.GREETING
        for keyword in self._GOODBYE_KEYWORDS:
            if q == keyword.lower() or q.startswith(keyword.lower()):
                return self.GOODBYE
        for keyword in self._CHITCHAT_KEYWORDS:
            if q == keyword.lower() or q.startswith(keyword.lower()):
                return self.CHITCHAT
        for keyword in self._CAPABILITY_KEYWORDS:
            if q == keyword.lower() or q.startswith(keyword.lower()):
                return self.ASSISTANT_CAPABILITY
        for keyword in self._UNRELATED_KEYWORDS:
            if keyword.lower() in q:
                return self.UNRELATED_QUESTION
        return None

    def _llm_classify(self, query: str, formatted_history_str: str) -> str:
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            self._logger.error(f"创建大模型对象失败,{e}")
            return self.PRODUCT_QUERY

        system_message = SystemMessage(content=INTENT_SYSTEM_TEMPLATE)
        user_prompt = INTENT_USER_TEMPLATE.format(
            history_text=formatted_history_str,
            query=query
        )
        human_message = HumanMessage(content=user_prompt)

        try:
            llm_result = llm_client.invoke([system_message, human_message])
            content = (llm_result.content or "").strip().lower()
            for intent in [self.GREETING, self.GOODBYE, self.CHITCHAT, self.ASSISTANT_CAPABILITY, self.UNRELATED_QUESTION, self.GENERAL_QUESTION, self.PRODUCT_QUERY]:
                if intent in content:
                    return intent
            return self.PRODUCT_QUERY
        except Exception as e:
            self._logger.error(f"LLM 意图分类失败,{e}")
            return self.PRODUCT_QUERY


class _ItemNameAligner:
    def __init__(self, logger: Logger, config: QueryConfig):
        self._logger = logger
        self._config = config

    def search_and_align(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 1. 混合检索向量数据库
        search_result = self._search_vector(item_names)
        # 2. 判断检索到的结果，如果为空，则表示 confirmed 和 options 都没有
        if not search_result:
            return [], []
        # 3. 根据混合向量检索到结果做对齐【confirmed/options】
        confirmed, options = self._align(search_result)
        # 4. 分数差异化过滤（多商品命中时，去掉与最高分差距过大的结果）
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
        # 5. 返回确定的 confirmed 容器和 options 容器
        return confirmed, options

    def _search_vector(self, item_names: List[str]):
        """
        [{
            "extracted_name":"大模型提取的item_name",
            "matches":[{"item_name":"从向量数据库匹配到的item_name","score":分数}]
        }]
        """
        final_search_result = []
        if not item_names:
            return final_search_result

        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self._logger.error(f"获取嵌入模型失败,{e}")
            return final_search_result

        try:
            embedding_result = generate_bge_m3_hybrid_vectors(embedding_client, item_names)
            dense_vector_list = embedding_result.get("dense")
            sparse_vector_list = embedding_result.get("sparse")
        except Exception as e:
            self._logger.error(f"向量嵌入失败,{e}")
            return final_search_result

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self._logger.error(f"获取milvus客户端失败,{e}")
            return final_search_result

        for index, item_name in enumerate(item_names):
            search_requests = create_hybrid_search_requests(
                dense_vector=dense_vector_list[index],
                sparse_vector=sparse_vector_list[index],
                limit=self._config.embedding_search_limit
            )

            milvus_search_results = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self._config.item_name_collection,
                search_requests=search_requests,
                limit=self._config.embedding_search_limit,
                output_fields=["item_name"]
            )

            matches = []
            for m in milvus_search_results[0]:
                matches.append({
                    "item_name": m["entity"]["item_name"],
                    "score": m["distance"]
                })

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]):
        """
        融合向量相似度 + 字符串相似度，对商品名进行对齐。
        """
        confirmed = []
        options = []

        for result in search_result:
            extracted_name = result.get("extracted_name", "")
            matches = result.get("matches", [])
            matches = sorted(matches, key=lambda x: x["score"], reverse=True)

            # 给每个候选计算融合分数
            scored_matches = []
            for m in matches:
                item_name = m["item_name"]
                vector_score = m["score"]
                string_sim = self._compute_string_similarity(extracted_name, item_name)
                combined = self._combine_score(vector_score, string_sim)
                scored_matches.append({
                    "item_name": item_name,
                    "vector_score": vector_score,
                    "string_sim": string_sim,
                    "combined": combined
                })

            # 高可置信：融合分数达标，或字符串相似度极高（如 RS12万用表 vs RS-12 数字万用表）
            high_confirmed = [
                sm for sm in scored_matches
                if sm["combined"] >= self._config.item_name_combined_threshold
                or sm["string_sim"] >= self._config.item_name_string_similarity_threshold
            ]

            if high_confirmed:
                best = high_confirmed[0]
                if best["item_name"] not in confirmed:
                    confirmed.append(best["item_name"])
                continue

            # 未命中融合阈值，回退到纯向量高可置信逻辑
            vector_high = [sm for sm in scored_matches if sm["vector_score"] >= self._config.item_name_high_confidence]
            if vector_high:
                if len(vector_high) == 1:
                    if vector_high[0]["item_name"] not in confirmed:
                        confirmed.append(vector_high[0]["item_name"])
                else:
                    max_score = vector_high[0]["vector_score"]
                    second_score = vector_high[1]["vector_score"]
                    if max_score - second_score > self._config.item_name_score_gap:
                        if vector_high[0]["item_name"] not in confirmed:
                            confirmed.append(vector_high[0]["item_name"])
                    else:
                        for sm in vector_high[:self._config.item_name_max_options]:
                            if sm["item_name"] not in confirmed and sm["item_name"] not in options:
                                options.append(sm["item_name"])
                continue

            # 中等可置信候选，放入 options 供用户确认
            medium = [sm for sm in scored_matches if sm["combined"] >= self._config.item_name_mid_confidence]
            if medium:
                for sm in medium[:self._config.item_name_max_options]:
                    if sm["item_name"] not in confirmed and sm["item_name"] not in options:
                        options.append(sm["item_name"])

        return confirmed, options[:self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):
        """
        多商品命中时，丢弃与最高分差距过大的结果。
        """
        item_name_score = {}
        for result in search_result:
            matches = result.get("matches", [])
            for m in matches:
                item_name = m.get("item_name")
                score = m.get("score", 0)
                item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        max_score = 0
        for item_name in confirmed:
            max_score = max(max_score, item_name_score.get(item_name, 0))

        final_confirmed = []
        for item_name in confirmed:
            if max_score - item_name_score.get(item_name, 0) <= self._config.item_name_score_gap:
                final_confirmed.append(item_name)

        return final_confirmed

    def _compute_string_similarity(self, name1: str, name2: str) -> float:
        """
        计算两个名称的字符串相似度，优先使用 rapidfuzz，未安装则使用 difflib。
        同时会生成常见别名（去掉空格、连字符等）取最大相似度。
        额外处理：前缀/子串关系（如 "蔡涛涛" 与 "蔡涛涛-AI应用(全栈开发)"）给予高相似度。
        """
        if not name1 or not name2:
            return 0.0

        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        # 子串/前缀匹配：较短名称是较长名称的子串，且长度不少于 3 个字符（避免 "RS" 等过短误匹配）
        shorter, longer = (norm1, norm2) if len(norm1) <= len(norm2) else (norm2, norm1)
        if len(shorter) >= 3 and shorter in longer:
            # 如果是前缀，相似度更高；如果是中间子串，略低
            return 0.92 if longer.startswith(shorter) else 0.80

        aliases1 = self._generate_aliases(name1)
        aliases2 = self._generate_aliases(name2)

        try:
            from rapidfuzz import fuzz
            max_sim = 0.0
            for a1 in aliases1:
                for a2 in aliases2:
                    max_sim = max(max_sim, fuzz.ratio(a1, a2) / 100.0)
            return max_sim
        except ImportError:
            import difflib
            max_sim = 0.0
            for a1 in aliases1:
                for a2 in aliases2:
                    max_sim = max(max_sim, difflib.SequenceMatcher(None, a1, a2).ratio())
            return max_sim

    def _generate_aliases(self, name: str) -> List[str]:
        """生成商品名的常见别名，用于字符串匹配。"""
        aliases = {name, self._normalize_name(name)}
        norm = self._normalize_name(name)
        # 去掉常见修饰词，进一步兼容口语简称
        for word in ["数字", "台式", "手持", "便携", "智能", "高精度", "多功能"]:
            if word in norm:
                aliases.add(norm.replace(word, ""))
        return list(aliases)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """名称规范化：全角转半角、小写、去除空格/连字符/下划线/点/斜杠。"""
        if not name:
            return ""
        result = []
        for ch in name:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                result.append(" ")
            else:
                result.append(ch)
        name = "".join(result).lower()
        for c in " -_./\\":
            name = name.replace(c, "")
        return name

    def _combine_score(self, vector_score: float, string_sim: float) -> float:
        """融合向量分数与字符串相似度。"""
        vector_weight = 1.0 - self._config.item_name_string_weight
        return vector_weight * vector_score + self._config.item_name_string_weight * string_sim


class _ItemNameExtractor:
    def __init__(self, logger: Logger, name: str):
        self._logger = logger
        self._name = name

    def extract_item_name(self, original_query: str, formatted_history_str: str) -> Dict[str, Any]:
        llm_result = {
            "item_names": [],
            "rewritten_query": original_query
        }

        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except Exception as e:
            self._logger.error(f"创建大模型对象失败,{e}")
            return llm_result

        system_message = SystemMessage(content=ITEM_NAME_SYSTEM_EXTRACT_TEMPLATE)
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=formatted_history_str,
            query=original_query
        )
        human_message = HumanMessage(content=user_prompt)

        try:
            llm_result = llm_client.invoke([system_message, human_message])
        except Exception as e:
            self._logger.error(f"调用大模型对象失败,{e}")
            return llm_result

        llm_content = llm_result.content
        if not llm_content:
            return llm_result

        return self._clean_and_parse(llm_content)

    def _clean_and_parse(self, llm_content: str) -> Dict[str, Any]:
        content = re.sub(r"^```(?:json)?\s*", "", llm_content)
        content = re.sub(r"\s*```$", "", content)

        try:
            llm_content_obj: Dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as e:
            self._logger.error(f"LLM 返回内容 JSON 解析失败:{e}, content={llm_content[:200]}")
            return {"item_names": [], "rewritten_query": ""}

        original_item_names = llm_content_obj.get("item_names")
        if not isinstance(original_item_names, list):
            item_names = []
        else:
            item_names = [item_name.strip() for item_name in original_item_names
                          if isinstance(item_name, str) and item_name.strip()]

        original_rewritten_query = llm_content_obj.get("rewritten_query")
        if not isinstance(original_rewritten_query, str):
            rewritten_query = ""
        else:
            rewritten_query = original_rewritten_query.strip()

        return {"item_names": item_names, "rewritten_query": rewritten_query}


class _DirectAnswerGenerator:
    """无商品名或助手能力类问题的直接回答生成器。"""

    def __init__(self, logger: Logger, config: QueryConfig):
        self._logger = logger
        self._config = config

    def generate(self, original_query: str, formatted_history_str: str) -> str:
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            self._logger.error(f"创建大模型对象失败,{e}")
            return self._default_answer()

        system_message = SystemMessage(content=DIRECT_ANSWER_SYSTEM_TEMPLATE)
        user_prompt = DIRECT_ANSWER_USER_TEMPLATE.format(
            history_text=formatted_history_str,
            query=original_query
        )
        human_message = HumanMessage(content=user_prompt)

        try:
            llm_result = llm_client.invoke([system_message, human_message])
            content = (llm_result.content or "").strip()
            return content if content else self._default_answer()
        except Exception as e:
            self._logger.error(f"直接回答生成失败,{e}")
            return self._default_answer()

    @staticmethod
    def _default_answer() -> str:
        return (
            "你好！我是问器知识库助手，可以帮你查询产品使用方法、技术参数、故障排查和维修指导。\n"
            "请直接告诉我具体的产品型号，例如：RS-12 数字万用表怎么测电压？"
        )


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor(self.logger, self.name)
        self._item_name_aligner = _ItemNameAligner(self.logger, self.config)
        self._intent_classifier = _QueryIntentClassifier(self.logger, self.config)
        self._direct_answer_generator = _DirectAnswerGenerator(self.logger, self.config)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state["original_query"]
        session_id = state.get("session_id", "")
        history_messages = get_recent_messages(session_id=session_id)
        state["history"] = history_messages

        formatted_history_str = ""
        for msg in history_messages:
            role = msg.get("role")
            content = msg.get("text", "")
            formatted_history_str += f"{role}: {content}\n"

        # 1. 意图识别
        intent = self._intent_classifier.classify(original_query, formatted_history_str)
        state["intent"] = intent
        self.logger.info(f"用户意图识别结果:{intent}")

        # 2. 非产品类意图直接给出友好回复
        if intent == _QueryIntentClassifier.GREETING:
            state["answer"] = (
                "你好！我是问器知识库助手，可以帮您查询产品使用方法、技术参数或维修指导。\n"
                "请直接告诉我产品型号，例如：RS-12 数字万用表怎么测电压？"
            )
            return state
        elif intent == _QueryIntentClassifier.GOODBYE:
            state["answer"] = "再见！如有其他问题，随时欢迎再次咨询。"
            return state
        elif intent == _QueryIntentClassifier.CHITCHAT:
            state["answer"] = "不客气！如果后续有产品相关的问题，随时找我。"
            return state
        elif intent == _QueryIntentClassifier.ASSISTANT_CAPABILITY:
            # 询问助手能力/身份，直接由大模型自主回答，不走检索流程
            state["answer"] = self._direct_answer_generator.generate(original_query, formatted_history_str)
            return state

        # 3. 提取商品名
        llm_result: Dict[str, Any] = self._extractor.extract_item_name(original_query, formatted_history_str)
        self.logger.info(f"大模型提取商品名的结果:{llm_result}")

        extracted_item_names = llm_result.get("item_names", [])
        rewritten_query = llm_result.get("rewritten_query") or original_query

        # 4. 对齐商品名
        confirmed, options = self._item_name_aligner.search_and_align(extracted_item_names)

        # 5. 决策
        self._dicide(confirmed, options, state, rewritten_query, original_query, intent)
        return state

    def _dicide(self, confirmed: List[str], options: List[str], state: QueryGraphState,
                rewritten_query: str, original_query: str, intent: str):
        # 5.1 有确定商品名，继续走后续检索流程
        if confirmed:
            doc_item_names = []
            section_titles = []
            for name in confirmed:
                if " > " in name:
                    # 章节标题格式：文档名 > 章节标题
                    doc, section = name.split(" > ", 1)
                    doc_item_names.append(doc)
                    section_titles.append(section)
                else:
                    doc_item_names.append(name)

            # 去重文档级 item_name
            state["item_names"] = list(dict.fromkeys(doc_item_names))

            # 如果有章节标题，把章节标题拼接到改写查询中，提升向量检索针对性
            if section_titles:
                unique_sections = list(dict.fromkeys(section_titles))
                enhanced_query = f"{rewritten_query} ({' / '.join(unique_sections)})"
                state["rewritten_query"] = enhanced_query
                self.logger.info(f"识别到章节标题，增强查询: {enhanced_query}")
            else:
                state["rewritten_query"] = rewritten_query
            return

        # 5.2 有候选但不精确，询问用户确认
        if options:
            state["answer"] = (
                f"我不能确认你指的是哪个对象，您是在询问以下选项：{'、'.join(options)} 吗？\n"
                f"如果是，请直接回复对应名称；如果不是，请提供更准确的关键词。"
            )
            return

        # 5.3 完全没有匹配到商品名
        # 5.3.1 general_question 属于"答案可能存在于知识库中的宽泛问题"，应优先走检索兜底，
        # 而不是直接让大模型自主回答（避免绕过知识库、产生幻觉）。

        # 5.3.2 未开启宽范围检索兜底，直接给出引导提示
        if not self.config.enable_broad_search_fallback:
            if intent == _QueryIntentClassifier.PRODUCT_QUERY:
                state["answer"] = (
                    "我未能识别到具体的产品名称或型号，请提供更准确的产品信息后再提问。"
                )
            else:
                state["answer"] = (
                    "我未能识别到具体的关键词或主题，请尝试提供更明确的名称、型号或关键词。"
                )
            return

        # 5.3.3 否则进入宽范围内部检索兜底模式
        state["is_broad_search"] = True
        state["item_names"] = []
        state["rewritten_query"] = rewritten_query

        # 企业私有化场景默认关闭 Web 回退，避免查询外发
        if not self.config.enable_web_fallback_when_no_item:
            state["skip_web_search"] = True

        # 记录日志：未匹配到商品名，已开启宽范围检索兜底
        self.logger.info(
            f"未匹配到商品名，已开启宽范围检索兜底: {original_query}, intent={intent}"
        )


if __name__ == '__main__':
    node = ItemNameConfirmedNode()
    test_states = [
        {"original_query": "你好", "session_id": "2"},
        {"original_query": "RS12万用表怎么使用？", "session_id": "2"},
        {"original_query": "怎么测量主板是否通电？", "session_id": "2"},
        {"original_query": "你能做什么？", "session_id": "2"},
        {"original_query": "如何学习电子维修？", "session_id": "2"},
        {"original_query": "今天天气怎么样？", "session_id": "2"},
        {"original_query": "谢谢", "session_id": "2"},
    ]
    for s in test_states:
        result_state = node.process(s)
        print("=" * 60)
        print(f"问题: {s['original_query']}")
        print(f"意图: {result_state.get('intent')}")
        print(f"商品名: {result_state.get('item_names')}")
        print(f"改写查询: {result_state.get('rewritten_query')}")
        print(f"答案/状态: {result_state.get('answer', '进入检索流程')}")
