"""
RAG 核心流程: 加载 -> 切分 -> 向量化 -> 入库 -> 检索 -> 生成
为了可读性, 整个 pipeline 放在一个文件里, 方便阅读和改造.
"""
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama


class RAGPipeline:
    def __init__(
        self,
        docs_dir: str = "docs",
        db_dir: str = "data/chroma",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        llm_model: str = "qwen2.5:3b",
        collection_name: str = "rag_demo",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.docs_dir = Path(docs_dir)
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        self.collection_name = collection_name

        print(f"[init] 加载 embedding 模型: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        print(f"[init] 初始化 Chroma (持久化目录: {self.db_dir})")
        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- 1. 加载 ----------
    def load_documents(self) -> List[Dict]:
        """加载 docs/ 下所有 .md / .txt / .pdf 文件."""
        docs = []
        for path in sorted(self.docs_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix in {".md", ".txt"}:
                    text = path.read_text(encoding="utf-8")
                elif suffix == ".pdf":
                    import pypdf
                    reader = pypdf.PdfReader(str(path))
                    text = "\n\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                else:
                    continue
            except Exception as e:
                print(f"[load] 跳过 {path}: {e}")
                continue

            if text.strip():
                docs.append({"path": str(path), "text": text})
                print(f"[load] {path} ({len(text)} chars)")
        return docs

    # ---------- 2. 切分 ----------
    def split_text(self, text: str) -> List[str]:
        """固定长度 + 重叠的朴素切分. 生产环境建议换成按段落/句子的语义切分."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return chunks

    # ---------- 3 & 4. 向量化 + 入库 ----------
    def index(self) -> int:
        """重建索引: 清空 -> 切分 -> 向量化 -> 批量写入."""
        # demo 场景直接全量重建, 生产环境要做增量
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        docs = self.load_documents()
        if not docs:
            print("[index] docs/ 目录下没有文档")
            return 0

        all_chunks, all_ids, all_meta = [], [], []
        for doc in docs:
            for i, chunk in enumerate(self.split_text(doc["text"])):
                all_chunks.append(chunk)
                all_ids.append(f"{doc['path']}::chunk_{i}")
                all_meta.append({"source": doc["path"], "chunk_id": i})

        print(f"[index] 共 {len(all_chunks)} 个 chunk, 开始向量化...")
        embeddings = self.embedder.encode(
            all_chunks,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        # Chroma 一次 add 建议不超过几千条, 数据多的时候要分批
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_meta,
        )
        print(f"[index] 入库完成, 当前 collection 大小: {self.collection.count()}")
        return len(all_chunks)

    # ---------- 5. 检索 ----------
    def retrieve(self, query: str) -> List[Dict]:
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=self.top_k,
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "score": 1 - dist,  # cosine distance -> similarity
                }
            )
        return hits

    # ---------- 6. 生成 ----------
    def generate(self, query: str, hits: List[Dict]) -> str:
        context = "\n\n".join(
            [
                f"【片段 {i+1} | 来源: {h['source']}】\n{h['text']}"
                for i, h in enumerate(hits)
            ]
        )
        prompt = f"""你是一个严谨的问答助手. 请只根据下面提供的资料片段回答用户问题.
如果资料里没有相关信息, 直接回答"资料中未提及", 不要编造.

===资料开始===
{context}
===资料结束===

用户问题: {query}

请用中文给出简洁准确的回答, 并在末尾用方括号标注你主要参考了哪些片段编号, 例如 [片段 1, 片段 2].
"""
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return response["message"]["content"]

    # ---------- 对外入口 ----------
    def ask(self, query: str) -> Dict:
        hits = self.retrieve(query)
        answer = self.generate(query, hits)
        return {"query": query, "answer": answer, "hits": hits}
