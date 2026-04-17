from __future__ import annotations

import json
import re
from pathlib import Path
from src.reranker import rerank_documents

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from src.config import (
    INDEX_DIR,
    EMBED_MODEL,
    NORMALIZE,
    DEVICE,
    BATCH_SIZE,
    USE_FAISS,
    FAISS_INDEX_PATH,
)

try:
    import faiss
except Exception:
    faiss = None


TOP_K = 5
DENSE_TOP_N = 10
BM25_TOP_N = 10
RRF_K = 60

_EMBEDDINGS = None
_EMB_MATRIX = None
_DOCS = None
_BM25 = None
_BM25_TOKENS = None
_FAISS_INDEX = None


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": NORMALIZE,
            "batch_size": BATCH_SIZE,
        },
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = build_embeddings()
    return _EMBEDDINGS


def clean_text(text: str) -> str:
    text = (text or "").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)


def doc_debug_label(doc: Document) -> str:
    metadata = doc.metadata or {}
    source = metadata.get("source", "")
    page = metadata.get("page", None)
    chunk_id = metadata.get("chunk_id", None)

    parts = [source]
    if page is not None:
        parts.append(f"page={page}")
    if chunk_id is not None:
        parts.append(f"chunk={chunk_id}")

    return " | ".join(parts)


def doc_debug_snippet(doc: Document, max_len: int = 200) -> str:
    text = clean_text(doc.page_content or "")
    return text[:max_len]


def print_ranked_docs(title: str, docs: list[Document], limit: int = 5) -> None:
    print(f"\n[{title}]")
    for i, doc in enumerate(docs[:limit], start=1):
        print(f"{i}. {doc_debug_label(doc)}")
        print(f"   {doc_debug_snippet(doc)}")


def load_faiss_index() -> None:
    global _FAISS_INDEX

    if not USE_FAISS or faiss is None:
        _FAISS_INDEX = None
        return

    if not FAISS_INDEX_PATH.exists():
        _FAISS_INDEX = None
        return

    try:
        _FAISS_INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    except Exception:
        _FAISS_INDEX = None


def load_index() -> None:
    global _EMB_MATRIX, _DOCS, _BM25, _BM25_TOKENS

    emb_path = INDEX_DIR / "embeddings.npy"
    docs_path = INDEX_DIR / "docs.json"

    if not emb_path.exists() or not docs_path.exists():
        raise FileNotFoundError(
            f"Index files not found in {INDEX_DIR}. Run: python -m src.export_index"
        )

    _EMB_MATRIX = np.load(str(emb_path)).astype(np.float32)

    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data["documents"]
    metadatas = data["metadatas"]

    _DOCS = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(documents, metadatas)
    ]

    _BM25_TOKENS = [tokenize(doc.page_content) for doc in _DOCS]
    _BM25 = BM25Okapi(_BM25_TOKENS)

    load_faiss_index()


def ensure_loaded() -> None:
    if _EMB_MATRIX is None or _DOCS is None or _BM25 is None:
        load_index()


def embed_query(query: str) -> np.ndarray:
    vector = np.array(_get_embeddings().embed_query(query), dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


def retrieve_dense_numpy(query: str, top_k: int) -> list[Document]:
    ensure_loaded()

    if top_k <= 0:
        return []

    query_vector = embed_query(query)
    scores = _EMB_MATRIX @ query_vector

    k = min(int(top_k), len(scores))
    if k == 0:
        return []

    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [_DOCS[i] for i in top_indices]


def retrieve_dense_faiss(query: str, top_k: int) -> list[Document]:
    ensure_loaded()

    if _FAISS_INDEX is None or top_k <= 0:
        return []

    query_vector = embed_query(query).reshape(1, -1).astype(np.float32)
    k = min(int(top_k), len(_DOCS))

    if k == 0:
        return []

    _, indices = _FAISS_INDEX.search(query_vector, k)

    docs = []
    for idx in indices[0]:
        if idx >= 0:
            docs.append(_DOCS[int(idx)])

    return docs


def retrieve_dense(query: str, top_k: int) -> list[Document]:
    if USE_FAISS and _FAISS_INDEX is not None:
        docs = retrieve_dense_faiss(query, top_k)
        if docs:
            return docs

    return retrieve_dense_numpy(query, top_k)


def retrieve_bm25(query: str, top_k: int) -> list[Document]:
    ensure_loaded()

    if top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = np.array(_BM25.get_scores(query_tokens), dtype=np.float32)

    k = min(int(top_k), len(scores))
    if k == 0:
        return []

    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [_DOCS[i] for i in top_indices]


def doc_key(doc: Document) -> tuple:
    metadata = doc.metadata or {}
    return (
        metadata.get("source", ""),
        metadata.get("page", None),
        metadata.get("chunk_id", None),
        (doc.page_content or "")[:120],
    )


def rrf_fuse(result_lists: list[list[Document]], top_k: int, rrf_k: int = RRF_K) -> list[Document]:
    scores: dict[tuple, float] = {}
    docs_by_key: dict[tuple, Document] = {}

    for ranked_docs in result_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            key = doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:top_k]]


def retrieve_hybrid(query: str, top_k: int = TOP_K):
    dense_docs = retrieve_dense(query, top_k=DENSE_TOP_N)
    bm25_docs = retrieve_bm25(query, top_k=BM25_TOP_N)

    fused = rrf_fuse(
        [dense_docs, bm25_docs],
        top_k=max(15, top_k),
        rrf_k=RRF_K,
    )

    print(f"[DEBUG] Reranking {len(fused)} docs...")
    reranked = rerank_documents(query, fused)

    return reranked[:top_k]


def main() -> None:
    print("Hybrid Retriever Ready. Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ").strip()

        if not query:
            print("Please enter a query.\n")
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        print(f"\nQuery: {query}\n")

        dense_results = retrieve_dense(query, top_k=5)
        print_ranked_docs("DENSE RESULTS", dense_results, limit=5)

        bm25_results = retrieve_bm25(query, top_k=5)
        print_ranked_docs("BM25 RESULTS", bm25_results, limit=5)

        hybrid_results = retrieve_hybrid(query, top_k=5)
        print_ranked_docs("HYBRID RESULTS", hybrid_results, limit=5)
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()