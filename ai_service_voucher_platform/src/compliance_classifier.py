# -*- coding: utf-8 -*-
"""
【真正智能版】服务券合规性审核器（LLM深度语义判定 + RAG增强）
作者：AI审核系统核心组
目标：100%输出 1(通过) 或 2(不通过)，拒绝“无法判断”；reason 必须像专家手写。

依据：
- 《广西制造业培优育强服务券平台 AI 清单》v2.1
- 2026年最新管理办法（政策库动态加载）
- 专家审核思维链（CoT）建模
"""

import requests
import json
import logging
import re
import os
from typing import Dict, List, Optional, Tuple
from src.models import ComplianceResult

# === 配置 ===
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
POLICY_KB_PATH = os.getenv("POLICY_KB_PATH", "./data/policy_kb.json")  # 政策向量库本地路径（可扩展为 FAISS）
TIMEOUT = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceClassifier:
    def __init__(self):
        # 加载基础政策规则（可选：用于前置快速过滤）
        self.support_directions = [
            "数字化改造", "工业互联网", "智能制造", "设备上云", "数据治理",
            "绿色低碳", "检验检测", "质量管理", "供应链优化", "市场拓展",
            "专精特新培育", "技术攻关", "标准制定", "知识产权服务"
        ]
        self.excluded_keywords = {
            "游学", "旅游", "温泉", "美食", "摄影", "插花", "代办", "包过",
            "政府关系", "政策申请代写", "融资顾问（非技术类）", "党建服务"
        }

        # 预定义 Few-shot 示例（强化专家语气）
        self.few_shot_examples = [
            {
                "input": "企业数字化转型咨询：提供ERP选型、MES实施路径规划。",
                "output": '{"decision": "通过", "reason": "服务聚焦制造业数字化核心环节，要素完整，符合支持方向。", "confidence": 0.96}'
            },
            {
                "input": "中小企业财税代办服务：代账、报税、工商变更。",
                "output": '{"decision": "不通过", "reason": "属行政事务性代办，非制造业技术/管理能力提升类服务，不在支持范围。", "confidence": 0.99}'
            },
            {
                "input": "企业定制化插花培训：提升员工美学素养与团队凝聚力。",
                "output": '{"decision": "不通过", "reason": "属于休闲娱乐类活动，与制造业培优育强目标无关，违反负面清单。", "confidence": 0.98}'
            }
        ]

    def _call_llm_with_rag(self, description: str, retrieved_policies: List[str] = None) -> Dict:
        """
        LLM + RAG 联合推理
        retrieved_policies: 从政策库检索出的3条最相关政策（可为空，此时仅用Few-shot）
        """
        # 构建 Prompt（严格遵循专家角色 + CoT + Few-shot）
        policy_context = ""
        if retrieved_policies:
            policy_context = "【相关现行政策依据】\n" + "\n".join(f"- {p}" for p in retrieved_policies[:3])

        few_shot_str = "\n".join(
            f"用户输入：{ex['input']}\n系统输出：{ex['output']}"
            for ex in self.few_shot_examples
        )

        prompt = f"""你是一名资深工信审核专家，负责广西制造业培优育强服务券的上架合规性终审。
请严格依据以下原则执行深度语义校验：
1. 支持方向仅限：{', '.join(self.support_directions)}
2. 绝对禁止：{', '.join(self.excluded_keywords)}（负面清单一票否决）
3. 要素完整性：必须明确体现‘功能介绍’、‘交付清单’、‘验收标准’中至少两项，且描述具体可执行。
4. 逻辑闭环：服务内容需与制造业企业生产经营直接相关，杜绝泛化、空洞描述。

{policy_context}

【审核任务】
请对以下服务描述进行最终判定，必须输出且仅输出一个JSON对象：
{{
    "decision": "通过" 或 "不通过",
    "reason": "专业、精炼（≤120字）、体现专家审阅视角的结论性意见",
    "confidence": 0.0~1.0（数值越高越确定）,
    "missing_elements": ["缺少的要素1", ...] （若无则为空列表）
}}

【输入服务描述】
\"\"\"{description}\"\"\"

【参考示例】
{few_shot_str}

现在请开始审核：
"""

        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.05,  # 极低随机性，保证审核严肃性与一致性
                "top_p": 0.8,
                "num_ctx": 4096
            }
        }

        try:
            resp = requests.post(LLM_API_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw_resp = resp.json()
            # 提取 response 字段并解析 JSON
            json_str = raw_resp.get("response", "{}")
            # 防御性处理：去除可能的 ```json 包裹
            json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_str).strip()
            result = json.loads(json_str)
            return result
        except Exception as e:
            logger.error(f"LLM调用失败: {e} | fallback to rule-based")
            # fallback：用规则引擎兜底（确保永不返回4）
            return self._fallback_rule_based(description)

    def _fallback_rule_based(self, description: str) -> Dict:
        """规则兜底：保证永不返回4"""
        desc_l = description.lower()
        
        # 1. 负面清单一票否决
        for kw in self.excluded_keywords:
            if kw in desc_l:
                return {
                    "decision": "不通过",
                    "reason": f"命中负面清单：含'{kw}'，不属于制造业培优育强支持范围。",
                    "confidence": 0.99,
                    "missing_elements": []
                }

        # 2. 要素缺失快速判断
        has_func = bool(re.search(r"功能|服务内容|能力", desc_l))
        has_deliver = bool(re.search(r"交付|输出|成果|报告", desc_l))
        has_accept = bool(re.search(r"验收|标准|指标|确认", desc_l))
        missing = []
        if not has_func: missing.append("功能介绍")
        if not has_deliver: missing.append("交付清单")
        if not has_accept: missing.append("验收标准")

        if len(missing) >= 2:
            return {
                "decision": "不通过",
                "reason": f"关键要素缺失：{', '.join(missing)}，无法评估服务可行性与闭环性。",
                "confidence": 0.90,
                "missing_elements": missing
            }

        # 3. 支持方向模糊？默认不通过（保守策略）
        if not any(sd in desc_l for sd in [s.lower() for s in self.support_directions]):
            return {
                "decision": "不通过",
                "reason": "服务内容未体现制造业数字化、技术提升或管理优化等核心方向，不符合政策导向。",
                "confidence": 0.85,
                "missing_elements": []
            }

        # 4. 兜底通过（仅当要素基本齐全 + 方向匹配）
        return {
            "decision": "通过",
            "reason": "服务内容聚焦制造业能力提升，要素基本完整，符合支持方向。",
            "confidence": 0.82,
            "missing_elements": missing
        }

    def _retrieve_policies(self, description: str) -> List[str]:
        """
        模拟 RAG 检索（当前为占位，可替换为 FAISS / Chroma）
        实际部署时应调用 policy_retriever.py
        """
        # 简单关键词匹配模拟
        keywords = ["数字化", "制造", "检验", "标准", "转型"]
        hits = []
        if any(kw in description for kw in keywords):
            hits.append("《2026年管理办法》第三条：支持企业开展数字化改造、智能制造装备应用...")
            hits.append("《2026年管理办法》第八条：检验检测类服务须具备CMA/CNAS资质...")
        return hits[:3]

    def classify_service(self, description: str) -> ComplianceResult:
        """
        主审核入口：强制输出 1 或 2，永不 4
        """
        if not description or len(description.strip()) < 10:
            return ComplianceResult(
                conclusion="不通过",
                confidence_score=0.95,
                primary_reason="服务描述为空或过短，无法进行有效审核。",
                suggestions=["请补充具体服务内容、技术路径与交付物描述。"]
            )

        # Step 1: 检索相关条款（模拟RAG）
        relevant_policies = self._retrieve_policies(description)

        # Step 2: 调用LLM专家系统
        expert_resp = self._call_llm_with_rag(description, relevant_policies)

        # Step 3: 映射结果（确保只有1/2）
        decision = expert_resp.get("decision", "不通过")
        reason = expert_resp.get("reason", "服务内容与制造业培优育强方向不符。")
        confidence = float(expert_resp.get("confidence", 0.8))

        # 强制约束：绝不允许“无法判断”
        if decision not in ["通过", "不通过"]:
            decision = "不通过"
            reason = f"[系统强制修正] 原判定为'{decision}'，已按规程修正为'不通过'。理由：{reason}"

        # 生成 suggestions（可选，用于前端提示）
        suggestions = []
        missing = expert_resp.get("missing_elements", [])
        if missing:
            suggestions.append(f"请补充：{', '.join(missing)}")

        return ComplianceResult(
            conclusion=decision,
            confidence_score=confidence,
            primary_reason=reason[:255],  # 严格符合甲方长度限制
            suggestions=suggestions
        )