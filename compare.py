"""对比有无 reranker 的检索结果."""
from src.rag import RAGPipeline

questions = [
    "RAG 的核心组件有哪些?",
    "什么是混合检索?",
    "怎么评估 RAG 的效果?",
]

print("=" * 60)
print("不使用 Reranker")
print("=" * 60)
rag_no = RAGPipeline(use_reranker=False)
for q in questions:
    print(f"\n问题: {q}")
    for i, h in enumerate(rag_no.retrieve(q), 1):
        preview = h["text"].replace("\n", " ")[:60]
        print(f"  [{i}] score={h['score']:.3f}  {preview}...")

print("\n" + "=" * 60)
print("使用 Reranker")
print("=" * 60)
rag_yes = RAGPipeline(use_reranker=True)
for q in questions:
    print(f"\n问题: {q}")
    for i, h in enumerate(rag_yes.retrieve(q), 1):
        preview = h["text"].replace("\n", " ")[:60]
        print(f"  [{i}] rerank={h['rerank_score']:.3f}  {preview}...")