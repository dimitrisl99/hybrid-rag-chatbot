import json
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    PROCESSED_DIR,
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


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": NORMALIZE,
            "batch_size": BATCH_SIZE,
        },
    )


def load_chunks() -> list[dict]:
    chunks_path = PROCESSED_DIR / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Could not find {chunks_path}. Run indexing first: python -m src.indexing"
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("chunks.json is empty.")

    return chunks


def export_docs_json(chunks: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        documents.append(text)
        metadatas.append(
            {
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "page_index": chunk.get("page_index"),
                "chunk_id": chunk.get("chunk_id"),
                "chunking": chunk.get("chunking"),
                "type": "pdf",
            }
        )
        ids.append(chunk.get("id"))

    docs_payload = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs_path = INDEX_DIR / "docs.json"

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(docs_payload, f, ensure_ascii=False, indent=2)

    print(f"[save] Wrote docs.json with {len(documents)} documents")
    return documents, metadatas, ids


def export_embeddings(documents: list[str]) -> np.ndarray:
    embeddings_model = build_embeddings()

    print(f"[embed] Encoding {len(documents)} chunks...")
    vectors = embeddings_model.embed_documents(documents)
    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {matrix.shape}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = INDEX_DIR / "embeddings.npy"
    np.save(emb_path, matrix)

    print(f"[save] Wrote embeddings.npy with shape {matrix.shape}")
    return matrix


def export_faiss_index(matrix: np.ndarray) -> None:
    if not USE_FAISS:
        print("[faiss] Skipped (USE_FAISS=False)")
        return

    if faiss is None:
        print("[faiss] Skipped (faiss not installed)")
        return

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"[save] Wrote faiss.index with {index.ntotal} vectors")


def main() -> None:
    chunks = load_chunks()
    documents, _, _ = export_docs_json(chunks)
    matrix = export_embeddings(documents)
    export_faiss_index(matrix)

    print("[main] Export completed.")
    print(f"[main] Output dir: {INDEX_DIR}")


if __name__ == "__main__":
    main()