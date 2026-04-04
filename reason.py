import fitz  # PyMuPDF - 用于PDF处理
import docx  # python-docx - 用于Word文档处理
import requests
import json
from typing import List, Dict, Optional
import time
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class OfflineDocumentAnalyzer:
    def __init__(self, ollama_api_base: str = "http://localhost:11434"):
        self.ollama_api_base = ollama_api_base
        self.document_chunks = []
        self.processed_document_text = ""
        self.tfidf_vectorizer = None
        self.doc_tfidf_matrix = None

    def extract_text_from_pdf(self, file_path: str) -> str:
        """从PDF文件提取文本"""
        try:
            doc = fitz.open(file_path)
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():  # 只添加非空文本
                    text_parts.append(text)
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            raise Exception(f"PDF解析失败: {str(e)}")

    def extract_text_from_docx(self, file_path: str) -> str:
        """从Word文档提取文本"""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            raise Exception(f"Word文档解析失败: {str(e)}")

    def extract_text_from_txt(self, file_path: str) -> str:
        """从文本文件提取文本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
                return content
        except Exception as e:
            raise Exception(f"文本文件解析失败: {str(e)}")

    def load_document(self, file_path: str) -> str:
        """加载文档并提取文本"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            content = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            content = self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            content = self.extract_text_from_txt(file_path)
        else:
            raise Exception(f"不支持的文件格式: {file_ext}")

        self.processed_document_text = content
        print(f"✅ 文档加载成功，共 {len(content)} 字符")
        return content

    def chunk_document(self, chunk_size: int = 500, overlap: int = 50):
        """将文档切分为小块"""
        if not self.processed_document_text:
            raise Exception("请先加载文档")

        # 按句子分割文本
        sentences = re.split(r'[。！？\n\r]+', self.processed_document_text)
        sentences = [s.strip() for s in sentences if s.strip()]  # 清理空句子

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # 确保句子不是太长以至于超过chunk_size
            if len(sentence) > chunk_size:
                # 如果句子本身太长，按长度分割
                for i in range(0, len(sentence), chunk_size - overlap):
                    sub_sentence = sentence[i:i + chunk_size - overlap]
                    if current_chunk and len(current_chunk + sub_sentence) > chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = sub_sentence
                    else:
                        if current_chunk:
                            current_chunk += " " + sub_sentence
                        else:
                            current_chunk = sub_sentence
                continue

            # 检查添加当前句子是否会超过大小限制
            if len(current_chunk) + len(sentence) <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # 如果当前块已经有内容，保存它
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 开始新的块，考虑重叠
                words = current_chunk.split()
                overlap_size = min(int(len(words) * 0.2), overlap)  # 取重叠比例或固定值的较小者
                overlap_words = " ".join(words[-overlap_size:]) if overlap_size > 0 else ""
                current_chunk = overlap_words + " " + sentence if overlap_words else sentence

        # 添加最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        self.document_chunks = chunks
        print(f"✅ 文档已切分为 {len(chunks)} 个片段")

        # 构建TF-IDF矩阵用于检索
        if chunks:
            print("📊 构建TF-IDF索引...")
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=None,  # 可以使用中文停用词
                ngram_range=(1, 2),
                max_features=10000
            )
            self.doc_tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks)
            print("✅ TF-IDF索引构建完成")

        return chunks

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关文档片段"""
        if not self.document_chunks or self.tfidf_vectorizer is None or self.doc_tfidf_matrix is None:
            raise Exception("请先分块文档")

        # 将查询转换为TF-IDF向量
        query_tfidf = self.tfidf_vectorizer.transform([query])

        # 计算余弦相似度
        similarities = cosine_similarity(query_tfidf, self.doc_tfidf_matrix).flatten()

        # 获取最相似的top_k个片段的索引
        top_indices = similarities.argsort()[-top_k:][::-1]

        # 过滤掉相似度很低的结果
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # 设定一个阈值
                relevant_chunks.append(self.document_chunks[idx])

        # 如果没有找到足够相似的片段，返回最前面的几个
        if not relevant_chunks:
            relevant_chunks = self.document_chunks[:top_k]

        return relevant_chunks

    def test_ollama_connection(self) -> bool:
        """测试与Ollama的连接"""
        try:
            url = f"{self.ollama_api_base}/api/tags"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def query_ollama(self, prompt: str, model: str = "llama3") -> str:
        """调用本地Ollama的Llama3模型"""
        url = f"{self.ollama_api_base}/api/generate"

        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 4096,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            response = requests.post(url, json=data, timeout=120)  # 增加超时时间
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()
        except requests.exceptions.Timeout:
            raise Exception("Ollama API请求超时，请检查Ollama服务是否正常运行")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API请求失败: {e}")
        except json.JSONDecodeError:
            raise Exception("Ollama API返回无效JSON格式")

    def answer_question(self, question: str, model: str = "llama3") -> str:
        """回答用户问题"""
        if not self.document_chunks:
            return "请先加载文档后再提问。"

        # 检索相关文档片段
        try:
            relevant_chunks = self.retrieve_relevant_chunks(question)
        except Exception as e:
            return f"检索文档片段时发生错误: {str(e)}"

        if not relevant_chunks:
            return "未能在文档中找到相关信息。"

        # 构建提示词
        context = "\n\n".join(relevant_chunks[:3])  # 取前3个最相关的块
        prompt = f"""你是政策文档分析助手。请基于以下政策文档内容回答用户的问题。如果文档中没有相关信息，请明确说明。

政策文档内容：
<context>
{context}
</context>

用户问题：
{question}

请给出准确的回答，并在回答中适当引用文档内容。如果涉及具体条款、数字或日期，请务必准确引用原文。回答要简洁明了。"""

        # 调用Ollama
        try:
            answer = self.query_ollama(prompt, model)
            return answer
        except Exception as e:
            return f"回答问题时发生错误: {str(e)}"

    def process_document(self, file_path: str, chunk_size: int = 500):
        """完整的文档处理流程"""
        print("🚀 开始处理文档...")

        # 1. 加载文档
        self.load_document(file_path)

        # 2. 分块
        self.chunk_document(chunk_size=chunk_size)

        print(f"✅ 文档处理完成！共 {len(self.document_chunks)} 个片段可供查询。")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='离线政策文档问答系统')
    parser.add_argument('--model', type=str, default='llama3', help='使用的Ollama模型名')
    parser.add_argument('--chunk-size', type=int, default=500, help='文档分块大小')
    args = parser.parse_args()

    analyzer = OfflineDocumentAnalyzer()

    print("=== 离线政策文档问答系统 ===")
    print("🔍 正在检测Ollama连接状态...")

    # 测试Ollama连接
    ollama_connected = analyzer.test_ollama_connection()

    if not ollama_connected:
        print("⚠️  无法连接到Ollama，请检查:")
        print("   - Ollama服务是否正在运行 (ollama serve)")
        print("   - 端口是否正确 (默认11434)")
        print("   - llama3模型是否已下载 (ollama pull llama3)")
        return

    print("✅ Ollama连接正常")

    # 获取文档路径
    while True:
        file_path = input("\n📝 请输入政策文档路径 (PDF/DOCX/TXT): ").strip()

        # 处理相对路径和用户主目录
        if file_path.startswith('~'):
            file_path = os.path.expanduser(file_path)
        elif not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if os.path.exists(file_path):
            break
        print(f"❌ 文件不存在: {file_path}")
        print("💡 请检查文件路径是否正确")

    try:
        # 处理文档
        analyzer.process_document(file_path, chunk_size=args.chunk_size)

        print("\n💬 文档已加载，现在可以提问了！输入 'quit' 或 'exit' 退出。")

        while True:
            question = input("\n❓ 请输入您的问题: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            if not question:
                continue

            print("\n🔍 正在分析...")
            start_time = time.time()

            answer = analyzer.answer_question(question, args.model)

            end_time = time.time()

            print(f"\n🤖 {args.model} 回答:")
            print(f"{answer}")
            print(f"\n⏱️  耗时: {end_time - start_time:.2f}秒")

    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n💥 发生错误: {str(e)}")
        print("\n💡 可能的原因:")
        print("- 文件格式不受支持")
        print("- 文件损坏或加密")
        print("- Ollama服务未启动")
        print("- 模型未下载")


if __name__ == "__main__":
    main()
