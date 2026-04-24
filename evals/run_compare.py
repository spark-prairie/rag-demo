"""对比不同配置的 RAG 效果."""
import json
import sys
from pathlib import Path
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGPipeline
from evals.run import evaluate, summarize


testset = json.loads(
    Path(__file__).parent.joinpath("testset.json").read_text()
)

configs = [
    {"name": "baseline", "use_reranker": False, "chunk_size": 500},
    {"name": "with_reranker", "use_reranker": True, "chunk_size": 500},
    {"name": "smaller_chunks", "use_reranker": True, "chunk_size": 300},
    {"name": "larger_chunks", "use_reranker": True, "chunk_size": 800},
]

all_results = {}
for cfg in configs:
    name = cfg.pop("name")
    print(f"\n\n{'#' * 60}\n# 配置: {name}\n# {cfg}\n{'#' * 60}")
    rag = RAGPipeline(**cfg)
    # 重新建索引 (chunk_size 变了要重建)
    rag.index()
    results = evaluate(rag, testset)
    summarize(results)
    all_results[name] = {
        "config": cfg,
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in results) / len(results),
        "keyword_recall": sum(r["keyword_recall"] for r in results) / len(results),
    }

print("\n\n最终对比:")
print(f"{'配置':<20} {'检索命中':<12} {'关键词召回':<12}")
for name, r in all_results.items():
    print(f"{name:<20} {r['retrieval_hit_rate']:<12.2%} {r['keyword_recall']:<12.2%}")

Path(__file__).parent.joinpath("compare_results.json").write_text(
    json.dumps(all_results, indent=2, ensure_ascii=False)
)
