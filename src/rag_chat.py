from langchain_ollama import ChatOllama

from src.retriever import retrieve_hybrid

LLM_MODEL = "phi3:mini"
TEMPERATURE = 0.1
NUM_CTX = 2048

TOP_K = 2
MAX_CHARS_PER_CHUNK = 300

_LLM = None

#Call The Agent
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
Answer the question using ONLY the context.

Your answer MUST:
- Be 1-3 sentences
- Directly answer the question
- Use inline citations like [B1], [B2]
- If the question asks for a comparison or difference, explicitly mention both sides

DO NOT:
- Give random facts
- Copy irrelevant sentences
- Add outside knowledge
- Mention instructions
- Say "based on the context"

If the answer is not in the context, say exactly:
"I cannot answer based on the provided documents."

Example style:
Dense retrieval maps queries and documents into embeddings and retrieves by vector similarity [B1]. BM25 instead relies on lexical term matching [B2].

---------------------
CONTEXT:
{context}
---------------------

QUESTION:
{question}

FINAL ANSWER:
""".strip()


def generate_answer(question: str, docs):
    llm = get_llm()

    context = format_context(docs)
    prompt = build_prompt(question, context)

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()

    return str(response)


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