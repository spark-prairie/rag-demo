"""评估 RAG 质量. 用法: python evals/run.py"""
import json
import sys
from pathlib import Path

# 把项目根加进 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGPipeline


def evaluate(rag, cases):
    results = []
    for case in cases:
        r = rag.ask(case["question"])
        
        # 指标 1: 检索是否命中了期望的源文件
        retrieval_hit = any(
            case["expected_source"] in h["source"] for h in r["hits"]
        )
        
        # 指标 2: 答案关键词召回率
        answer_lower = r["answer"].lower()
        keywords = case.get("expected_keywords", [])
        keyword_hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        keyword_recall = keyword_hits / len(keywords) if keywords else 1.0
        
        # 指标 3: 期望短语是否出现
        contains = case.get("expected_answer_contains", "")
        phrase_hit = contains.lower() in answer_lower if contains else None
        
        results.append({
            "id": case["id"],
            "question": case["question"],
            "retrieval_hit": retrieval_hit,
            "keyword_recall": keyword_recall,
            "phrase_hit": phrase_hit,
            "answer": r["answer"][:200],
        })
    return results


def summarize(results):
    n = len(results)
    retrieval_acc = sum(r["retrieval_hit"] for r in results) / n
    avg_keyword = sum(r["keyword_recall"] for r in results) / n
    phrase_ns = [r for r in results if r["phrase_hit"] is not None]
    phrase_acc = (
        sum(r["phrase_hit"] for r in phrase_ns) / len(phrase_ns)
        if phrase_ns
        else None
    )
    print(f"\n{'='*50}")
    print(f"样本数: {n}")
    print(f"检索命中率: {retrieval_acc:.2%}")
    print(f"答案关键词召回: {avg_keyword:.2%}")
    if phrase_acc is not None:
        print(f"期望短语命中率: {phrase_acc:.2%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    testset = json.loads(
        Path(__file__).parent.joinpath("testset.json").read_text()
    )
    rag = RAGPipeline()
    results = evaluate(rag, testset)
    
    # 打印每条结果
    for r in results:
        status = "✓" if r["retrieval_hit"] else "✗"
        print(f"{status} [{r['id']}] kw_recall={r['keyword_recall']:.2f}")
        print(f"    Q: {r['question']}")
        print(f"    A: {r['answer'][:100]}...")
    
    summarize(results)
    
    # 存结果, 方便后续对比
    output = Path(__file__).parent / "results.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n结果已保存到 {output}")
