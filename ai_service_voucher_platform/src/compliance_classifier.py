import os
import json
import faiss
import requests
from sentence_transformers import SentenceTransformer
from src.models import ComplianceResult
from src.utils import load_yaml
import logging

class ComplianceClassifier:
    def __init__(self, config_path="./config/policy_rules.yaml", index_dir="./index"):
        self.rules = load_yaml(config_path)
        self.index_dir = index_dir
        self.encoder, self.faiss_index, self.index_id_map = self._load_index()

    def _load_index(self):
        try:
            encoder = SentenceTransformer("./all-MiniLM-L6-v2")
            faiss_index = faiss.read_index(os.path.join(self.index_dir, "policy_faiss.bin"))
            with open(os.path.join(self.index_dir, "index_id_map.json"), "r", encoding="utf-8") as f:
                index_id_map = json.load(f)
            return encoder, faiss_index, index_id_map
        except Exception as e:
            logging.error(f"加载索引失败: {e}")
            raise RuntimeError("未找到或无法加载知识库索引。请确保已运行 `update_knowledge_base.py`。")

    def _call_qwen_generate_suggestion(self, desc: str, reason: str) -> str:
        prompt = (f"你是一个专业的政府项目审核员，正在审核一个不完全合规的服务产品。\n"
                  f"服务描述：'{desc}'\n"
                  f"初步判定原因：'{reason}'\n"
                  f"请根据以上信息，生成一条不超过80字的、具体且有建设性的修改建议，引导服务商向制造业培优育强方向调整。")
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen3:8b", "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
                timeout=20
            )
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except requests.RequestException as e:
            logging.warning(f"调用LLM失败: {e}, 将使用预设模板。")
            return ""

    def classify_service(self, description: str) -> ComplianceResult:
        # 1. 负面清单规则：硬性一票否决
        for rule in self.rules.get('excluded_services', []):
            matched = [kw for kw in rule['keywords'] if kw in description]
            if matched:
                return ComplianceResult(
                    conclusion="不通过",
                    confidence_score=0.99,
                    primary_reason=f"命中禁止项: {rule['name']}",
                    matched_keywords=matched
                )

        # 2. 描述模糊场景：触发“部分通过”和AI建议
        for rule in self.rules.get('conditional_approval_scenarios', []):
             if any(kw in description for kw in rule['trigger_keywords']):
                suggestion = self._call_qwen_generate_suggestion(description, rule['cause'])
                return ComplianceResult(
                    conclusion="部分通过",
                    confidence_score=0.75,
                    primary_reason=rule['cause'],
                    suggestions=[suggestion or rule['suggestion_template']]
                )
        
        # 3. 向量检索，匹配支持方向
        query_embedding = self.encoder.encode([description]).astype("float32")
        distances, indices = self.faiss_index.search(query_embedding, 1)
        
        if indices[0][0] != -1:
            best_match_id = self.index_id_map[str(indices[0][0])]
            if "support_directions" in best_match_id:
                return ComplianceResult(
                    conclusion="通过",
                    confidence_score=max(0.6, 1 / (1 + distances[0][0])), # 基础分0.6
                    primary_reason="服务内容与政策支持方向高度相关。"
                )

        # 4. 默认结论：无法判断
        return ComplianceResult(
            conclusion="无法判断",
            confidence_score=0.5,
            primary_reason="服务内容与现有政策库的匹配度较低，需人工审核。"
        )