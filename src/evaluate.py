import json
import re
import unicodedata
from pathlib import Path

from src.retriever import retrieve_dense, retrieve_bm25, retrieve_hybrid

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval" / "eval_queries.json"


def load_eval_data():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_PATH}")

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_source_name(text: str) -> str:
    text = (text or "").strip().lower()

    # unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # common ligatures / weird chars
    text = text.replace("ﬀ", "ff")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("", "")

    # remove .pdf for comparison robustness
    text = re.sub(r"\.pdf$", "", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def hit_at_k(results, relevant_sources, k=5):
    top_results = results[:k]
    retrieved_sources = {
        normalize_source_name(doc.metadata.get("source"))
        for doc in top_results
    }
    relevant_set = {normalize_source_name(src) for src in relevant_sources}

    return 1 if any(source in retrieved_sources for source in relevant_set) else 0


def recall_at_k(results, relevant_sources, k=5):
    top_results = results[:k]
    retrieved_sources = {
        normalize_source_name(doc.metadata.get("source"))
        for doc in top_results
    }
    relevant_set = {normalize_source_name(src) for src in relevant_sources}

    if not relevant_set:
        return 0.0

    return len(retrieved_sources & relevant_set) / len(relevant_set)


def mrr_at_k(results, relevant_sources, k=5):
    relevant_set = {normalize_source_name(src) for src in relevant_sources}

    for rank, doc in enumerate(results[:k], start=1):
        source = normalize_source_name(doc.metadata.get("source"))
        if source in relevant_set:
            return 1.0 / rank

    return 0.0


def evaluate_method(method_name, retrieval_fn, eval_data, k=5):
    hits = []
    recalls = []
    mrrs = []

    print(f"\nEvaluating: {method_name}")
    print("=" * 60)

    for item in eval_data:
        query = item["query"]
        relevant_sources = item["relevant_sources"]

        results = retrieval_fn(query, top_k=k)

        h = hit_at_k(results, relevant_sources, k=k)
        r = recall_at_k(results, relevant_sources, k=k)
        m = mrr_at_k(results, relevant_sources, k=k)

        hits.append(h)
        recalls.append(r)
        mrrs.append(m)

        print(f"\nQuery: {query}")
        print(f"Hit@{k}: {h} | Recall@{k}: {r:.3f} | MRR@{k}: {m:.3f}")

        print("Relevant sources:")
        for src in relevant_sources:
            print(f"  - {src}")

        print("Retrieved:")
        for idx, doc in enumerate(results[:k], start=1):
            print(
                f"  {idx}. {doc.metadata.get('source')} | "
                f"page={doc.metadata.get('page')} | "
                f"chunk={doc.metadata.get('chunk_id')}"
            )

    summary = {
        "method": method_name,
        f"Hit@{k}": sum(hits) / len(hits),
        f"Recall@{k}": sum(recalls) / len(recalls),
        f"MRR@{k}": sum(mrrs) / len(mrrs),
    }

    return summary


def main():
    eval_data = load_eval_data()
    k = 5

    results = []
    results.append(evaluate_method("Dense", retrieve_dense, eval_data, k=k))
    results.append(evaluate_method("BM25", retrieve_bm25, eval_data, k=k))
    results.append(evaluate_method("Hybrid", retrieve_hybrid, eval_data, k=k))

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for row in results:
        print(
            f"{row['method']}: "
            f"Hit@{k}={row[f'Hit@{k}']:.3f}, "
            f"Recall@{k}={row[f'Recall@{k}']:.3f}, "
            f"MRR@{k}={row[f'MRR@{k}']:.3f}"
        )


if __name__ == "__main__":
    main()