"""Terminal version of IntelliRAG (same RAG engine as the website).
Run:  python main.py"""
from rag import llm
from rag.answer import Options, ask
from rag.vector_store import KnowledgeBase, ingest_folder


def main():
    print("=" * 56)
    print("            IntelliRAG - Document Search")
    print("=" * 56)
    kb = KnowledgeBase()
    kb.load()  
    ingest_folder(kb)
    if kb.size() == 0:
        print("No documents found. Add PDF or TXT files to the data folder.")
        return
    mode = f"{llm.provider()} / {llm.model_name()}" if llm.available() else "retrieval only (add a key for AI answers)"
    print(f"Documents: {len(kb.documents())}   Chunks: {kb.size()}   Mode: {mode}")
    history = []
    while True:
        question = input("\nAsk your question (or type exit): ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        result = ask(kb, question, history, Options())
        print("\nAnswer:\n" + result["answer"])
        for s in result["sources"]:
            page = f"page {s['page']}" if s["page"] is not None else "text file"
            print(f"  [{s['n']}] {s['source']} - {page} - confidence {round(s['confidence'] * 100)}%")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]


if __name__ == "__main__":
    main()
