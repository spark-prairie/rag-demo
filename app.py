"""
Gradio Web 界面.
运行: python app.py
打开: http://127.0.0.1:7860
"""
import gradio as gr

from src.rag import RAGPipeline

rag = RAGPipeline()


def chat(message: str, history):
    if rag.collection.count() == 0:
        return "向量库是空的, 请先点上面的 \"重建索引\" 按钮. (别忘了先往 docs/ 目录放文档)"
    result = rag.ask(message)
    sources_md = "\n\n---\n**引用片段**"
    for i, h in enumerate(result["hits"], 1):
        preview = h["text"].replace("\n", " ")[:150]
        sources_md += f"\n\n**[{i}]** `{h['source']}` · score `{h['score']:.3f}`\n> {preview}..."
    return result["answer"] + sources_md


def rebuild_index():
    n = rag.index()
    if n == 0:
        return "⚠️ docs/ 目录下没有文档"
    return f"✅ 已重建索引, 共 {n} 个片段"


with gr.Blocks(title="RAG Demo") as demo:
    gr.Markdown(
        "# 🧠 本地 RAG Demo\n"
        "把 `.md` / `.txt` 文档放进 `docs/` 目录, 点 **重建索引**, 就可以开始问了."
    )
    with gr.Row():
        rebuild_btn = gr.Button("🔄 重建索引", variant="primary", scale=1)
        status = gr.Textbox(label="状态", interactive=False, scale=3)
    rebuild_btn.click(rebuild_index, outputs=status)

    gr.ChatInterface(
        fn=chat,
        examples=["什么是 RAG?", "RAG 的核心组件有哪些?", "RAG 有什么改进方向?"],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
