from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_RERANK_MODEL = None


def get_reranker() -> CrossEncoder:
    global _RERANK_MODEL

    if _RERANK_MODEL is None:
        _RERANK_MODEL = CrossEncoder(RERANK_MODEL_NAME)

    return _RERANK_MODEL


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    if not documents:
        return []

    model = get_reranker()

    pairs = [(query, doc.page_content or "") for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

    reranked_docs = []
    for doc, score in scored_docs:
        metadata = dict(doc.metadata or {})
        metadata["rerank_score"] = float(score)

        reranked_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    if top_k is not None:
        return reranked_docs[:top_k]

    return reranked_docs