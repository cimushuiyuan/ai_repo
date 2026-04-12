import os
import re
import json
import faiss
import sqlite3
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
from src.utils import load_yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyIngestor:
    def __init__(self, config_path="./config/policy_rules.yaml", source_dir="./policy_source", index_dir="./index"):
        self.config_path = config_path
        self.source_dir = source_dir
        self.index_dir = index_dir
        self.encoder = SentenceTransformer("./all-MiniLM-L6-v2") # 使用轻量级高效模型

    def ingest_data(self):
        logger.info("开始更新知识库...")
        os.makedirs(self.index_dir, exist_ok=True)
        rules = load_yaml(self.config_path)

        all_texts, index_id_map = [], {}
        
        # 1. 结构化规则处理
        for category in ['support_directions', 'excluded_services', 'conditional_approval_scenarios']:
            for i, rule in enumerate(rules.get(category, [])):
                key = f"rule_{category}_{rule['id']}"
                text = f"{rule.get('name', '')}: {' '.join(rule.get('keywords',[]))}. {rule.get('description','')}"
                all_texts.append(text)
                index_id_map[len(all_texts) - 1] = key

        # 2. 非结构化文档处理 (PDF, TXT)
        if os.path.exists(self.source_dir):
            for filename in sorted(os.listdir(self.source_dir)):
                path = os.path.join(self.source_dir, filename)
                if filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif filename.lower().endswith(".txt"):
                    loader = TextLoader(path, encoding='utf-8')
                else:
                    continue
                
                try:
                    pages = loader.load_and_split()
                    for i, page in enumerate(pages):
                        key = f"doc_{os.path.basename(path)}_{i}"
                        all_texts.append(page.page_content)
                        index_id_map[len(all_texts) - 1] = key
                except Exception as e:
                    logger.error(f"处理文件 {filename} 失败: {e}")

        # 3. 创建并保存FAISS向量索引
        if not all_texts:
            logger.warning("未找到任何可供学习的文本，知识库为空。")
            return
            
        embeddings = self.encoder.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        
        faiss.write_index(index, os.path.join(self.index_dir, "policy_faiss.bin"))
        with open(os.path.join(self.index_dir, "index_id_map.json"), "w", encoding="utf-8") as f:
            json.dump(index_id_map, f)

        logger.info(f"知识库更新完成！共索引 {len(all_texts)} 条信息。")