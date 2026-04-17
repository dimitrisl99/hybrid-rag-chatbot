import json
import hashlib
import re
from pathlib import Path

import chromadb
import fitz
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    PDF_DIR,
    CHROMA_DIR,
    PROCESSED_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    NORMALIZE,
    DEVICE,
    BATCH_SIZE,
    PDF_TARGET_CHARS,
    PDF_MAX_CHARS,
    PDF_MIN_CHARS,
    PDF_SIM_THRESHOLD,
    PDF_WINDOW_SENTENCES,
    MIN_CHUNK_LENGTH,
    RESET_COLLECTION,
)


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_spaces(text: str) -> str:
    text = (text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_light_headers_footers(text: str) -> str:
    lines = (text or "").splitlines()

    if lines:
        last = lines[-1].strip().lower()
        if re.match(r"^(page)\s*\d+\s*$", last) or re.match(r"^\d{1,4}$", last):
            lines = lines[:-1]

    if lines:
        first = lines[0].strip().lower()
        if len(first) <= 40 and any(k in first for k in ["page", "chapter", "section"]):
            lines = lines[1:]

    return "\n".join(lines).strip()


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def cosine_similarity(vec_a, vec_b) -> float:
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / ((norm_a ** 0.5) * (norm_b ** 0.5))


def stable_id(*parts: str) -> str:
    raw = "||".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": NORMALIZE,
            "batch_size": BATCH_SIZE,
        },
    )


def reset_collection(persist_dir: Path, collection_name: str) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    existing_collections = {collection.name for collection in client.list_collections()}
    if collection_name in existing_collections:
        client.delete_collection(collection_name)
        print(f"[reset] Deleted collection: {collection_name}")


def get_vectordb(embeddings) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )


def semantic_split_page(
    page_text: str,
    embedder,
    target_chars: int,
    max_chars: int,
    min_chars: int,
    sim_threshold: float,
    window_sentences: int,
) -> list[str]:
    page_text = (page_text or "").strip()
    if not page_text:
        return []

    sentences = split_sentences(page_text)
    if not sentences:
        return []

    sentence_windows = []
    i = 0
    while i < len(sentences):
        window_text = " ".join(sentences[i:i + window_sentences]).strip()
        if window_text:
            sentence_windows.append(window_text)
        i += window_sentences

    if not sentence_windows:
        return []

    window_embeddings = embedder.embed_documents(sentence_windows)

    initial_chunks = []
    current_chunk = [sentence_windows[0]]
    current_length = len(sentence_windows[0])

    for idx in range(1, len(sentence_windows)):
        similarity = cosine_similarity(
            window_embeddings[idx - 1],
            window_embeddings[idx],
        )

        should_split = (similarity < sim_threshold) and (current_length >= min_chars)

        if should_split:
            initial_chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence_windows[idx]]
            current_length = len(sentence_windows[idx])
        else:
            current_chunk.append(sentence_windows[idx])
            current_length += 1 + len(sentence_windows[idx])

    if current_chunk:
        initial_chunks.append(" ".join(current_chunk).strip())

    merged_chunks = []
    buffer = ""

    for chunk in initial_chunks:
        if not buffer:
            buffer = chunk
            continue

        if len(buffer) + 1 + len(chunk) <= target_chars:
            buffer = buffer + " " + chunk
        else:
            merged_chunks.append(buffer.strip())
            buffer = chunk

    if buffer:
        merged_chunks.append(buffer.strip())

    final_chunks = []
    for chunk in merged_chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                piece = chunk[start:start + max_chars].strip()
                if piece:
                    final_chunks.append(piece)
                start += max_chars

    return [chunk for chunk in final_chunks if chunk and len(chunk) >= MIN_CHUNK_LENGTH]


def load_pdf_docs(embeddings) -> tuple[list[Document], list[str], list[dict]]:
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))

    docs_to_add: list[Document] = []
    ids_to_add: list[str] = []
    chunk_records: list[dict] = []

    print(f"[pdf] Found {len(pdf_paths)} PDFs")

    for pdf_path in pdf_paths:
        try:
            with fitz.open(pdf_path) as pdf_document:
                for page_index in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_index)
                    text = page.get_text("text")
                    text = clean_text(text)
                    text = strip_light_headers_footers(text)

                    if not text or len(text) < MIN_CHUNK_LENGTH:
                        continue

                    chunks = semantic_split_page(
                        page_text=text,
                        embedder=embeddings,
                        target_chars=PDF_TARGET_CHARS,
                        max_chars=PDF_MAX_CHARS,
                        min_chars=PDF_MIN_CHARS,
                        sim_threshold=PDF_SIM_THRESHOLD,
                        window_sentences=PDF_WINDOW_SENTENCES,
                    )

                    for chunk_index, chunk in enumerate(chunks):
                        chunk = compact_spaces(chunk)
                        if not chunk:
                            continue

                        doc_id = stable_id(
                            "pdf",
                            pdf_path.name,
                            page_index + 1,
                            chunk_index,
                        )

                        metadata = {
                            "type": "pdf",
                            "source": pdf_path.name,
                            "page": page_index + 1,
                            "page_index": page_index,
                            "chunk_id": chunk_index,
                            "chunking": "semantic_v1",
                        }

                        docs_to_add.append(
                            Document(
                                page_content=chunk,
                                metadata=metadata,
                            )
                        )

                        ids_to_add.append(doc_id)

                        chunk_records.append(
                            {
                                "id": doc_id,
                                "text": chunk,
                                "source": pdf_path.name,
                                "page": page_index + 1,
                                "page_index": page_index,
                                "chunk_id": chunk_index,
                                "chunking": "semantic_v1",
                            }
                        )

        except Exception as exc:
            print(f"[pdf][WARN] Failed {pdf_path.name}: {exc}")

    print(f"[pdf] Chunks ready: {len(docs_to_add)}")
    return docs_to_add, ids_to_add, chunk_records


def save_chunks_json(chunk_records: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "chunks.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunk_records, file, ensure_ascii=False, indent=2)

    print(f"[save] Wrote {len(chunk_records)} chunks to {output_path}")


def add_in_batches(
    vectordb,
    docs_to_add: list[Document],
    ids_to_add: list[str],
    batch_size: int = 200,
) -> None:
    for i in range(0, len(docs_to_add), batch_size):
        batch_docs = docs_to_add[i:i + batch_size]
        batch_ids = ids_to_add[i:i + batch_size]

        vectordb.add_documents(batch_docs, ids=batch_ids)
        print(f"[add] {min(i + batch_size, len(docs_to_add))}/{len(docs_to_add)}")


def main() -> None:
    embeddings = build_embeddings()

    if RESET_COLLECTION:
        reset_collection(CHROMA_DIR, COLLECTION_NAME)

    vectordb = get_vectordb(embeddings)

    pdf_docs, pdf_ids, chunk_records = load_pdf_docs(embeddings)

    if not pdf_docs:
        print("[main] No PDF chunks found.")
        return

    save_chunks_json(chunk_records)
    add_in_batches(vectordb, pdf_docs, pdf_ids, batch_size=200)

    print("[main] Indexing completed.")
    print(f"[main] Collection: {COLLECTION_NAME}")
    print(f"[main] Total indexed chunks: {len(pdf_docs)}")


if __name__ == "__main__":
    main()