from langchain_core.documents import Document

from src.reranker import rerank_documents


def main():
    query = "What is retrieval-augmented generation?"

    docs = [
        Document(
            page_content="RAG combines retrieval with generation by using external documents during response generation.",
            metadata={"source": "rag.pdf"}
        ),
        Document(
            page_content="ColBERT is a late interaction model for passage retrieval.",
            metadata={"source": "colbert.pdf"}
        ),
        Document(
            page_content="REALM augments language model pretraining with retrieval.",
            metadata={"source": "realm.pdf"}
        ),
    ]

    results = rerank_documents(query, docs)

    for i, doc in enumerate(results, start=1):
        print(
            f"{i}. {doc.metadata.get('source')} | "
            f"score={doc.metadata.get('rerank_score'):.4f}"
        )
        print(doc.page_content)
        print("-" * 80)


if __name__ == "__main__":
    main()