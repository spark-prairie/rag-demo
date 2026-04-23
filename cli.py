"""
命令行入口.

用法:
    python cli.py index                # 重建索引
    python cli.py ask "你的问题"        # 提问
"""
import sys

from src.rag import RAGPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    rag = RAGPipeline()
    cmd = sys.argv[1]

    if cmd == "index":
        n = rag.index()
        print(f"\n已索引 {n} 个片段")
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('请提供问题, 例如: python cli.py ask "什么是 RAG?"')
            sys.exit(1)
        query = sys.argv[2]
        result = rag.ask(query)

        print(f"\n问题: {result['query']}\n")
        print("检索到的片段:")
        for i, h in enumerate(result["hits"], 1):
            preview = h["text"].replace("\n", " ")[:80]
            print(f"  [{i}] score={h['score']:.3f}  source={h['source']}")
            print(f"      {preview}...")
        print(f"\n回答:\n{result['answer']}")
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
