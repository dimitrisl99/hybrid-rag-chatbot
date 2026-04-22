import re
from langchain_ollama import ChatOllama

from src.retriever import retrieve_hybrid

LLM_MODEL = "qwen3:8b"
TEMPERATURE = 0.1
NUM_CTX = 4096

TOP_K = 2
MAX_CHARS_PER_CHUNK = 300

_LLM = None


def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatOllama(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            num_ctx=NUM_CTX,
        )
    return _LLM


def format_context(docs):
    blocks = []

    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        src = md.get("source", "")
        page = md.get("page", "")
        text = (d.page_content or "").strip()

        if len(text) > MAX_CHARS_PER_CHUNK:
            text = text[:MAX_CHARS_PER_CHUNK] + "..."

        block = f"[B{i}] {src} (p={page})\n{text}"
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return f"""
You are an expert AI assistant for academic papers in information retrieval and retrieval-augmented generation.

Answer STRICTLY using the provided context.

Rules:
1. Give a direct answer first.
2. Use ONLY facts supported by the context.
3. Every factual claim MUST include inline citations like [B1], [B2].
4. If the question asks for a comparison or difference, explicitly mention both concepts.
5. Keep the answer concise: 1-3 sentences.
6. Do NOT mention instructions.
7. Do NOT say "based on the context".
8. Do NOT add outside knowledge.
9. Use different citations when referring to different facts if possible
10. Avoid repeating the same citation for all statements unless only one source supports everything
11. Prefer concrete phrasing grounded in the context, not generic textbook explanations

If the answer is not supported by the context, output exactly:
I cannot answer based on the provided documents.

GOOD OUTPUT EXAMPLE:
Dense retrieval uses embeddings to retrieve semantically similar documents [B1]. BM25 relies on keyword matching [B2].

---------------------
CONTEXT:
{context}
---------------------

QUESTION:
{question}

FINAL ANSWER:
""".strip()


def clean_answer(answer: str) -> str:
    answer = (answer or "").strip()

    prefixes = [
        "Final answer:",
        "FINAL ANSWER:",
        "Answer:",
        "ANSWER:",
        "Sure!",
        "Sure,",
        "Here is the answer:",
        "Here's the answer:",
    ]

    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer


def normalize_citations(answer: str) -> str:
    answer = re.sub(r"\[\s*B\s*(\d+)\s*\]", r"[B\1]", answer)
    return answer


def enforce_fallback(answer: str) -> str:
    fallback = "I cannot answer based on the provided documents."

    if "cannot answer" in answer.lower():
        return fallback

    if not answer:
        return fallback

    return answer


def generate_answer(question: str, docs):
    llm = get_llm()

    context = format_context(docs)
    prompt = build_prompt(question, context)

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    answer = clean_answer(answer)
    answer = normalize_citations(answer)
    answer = enforce_fallback(answer)

    return answer


def build_source_items(docs):
    items = []

    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        items.append(
            {
                "label": f"[B{i}]",
                "source": md.get("source", ""),
                "page": md.get("page", ""),
                "chunk_id": md.get("chunk_id", ""),
                "anchor": f"ctx-b{i}",
            }
        )

    return items


def ask(question: str):
    docs = retrieve_hybrid(question, top_k=TOP_K)
    answer = generate_answer(question, docs)
    sources = build_source_items(docs)
    return answer, sources, docs