import re
import streamlit as st

from src.rag_chat import ask, LLM_MODEL

st.set_page_config(
    page_title="Hybrid RAG Chatbot",
    page_icon="💬",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
def reset_chat():
    st.session_state.messages = []
    if "_pending_question" in st.session_state:
        del st.session_state["_pending_question"]


def extract_citations(answer_text: str) -> list[str]:
    return sorted(set(re.findall(r"\[B\d+\]", answer_text)))


def make_answer_clickable(answer_text: str, sources: list[dict]) -> str:
    """
    Replace [B1], [B2] with clickable anchors to retrieved context blocks.
    """
    html = answer_text

    for src in sources:
        label = src["label"]      # e.g. [B1]
        anchor = src["anchor"]    # e.g. ctx-b1
        safe_label = re.escape(label)

        html = re.sub(
            safe_label,
            f'<a href="#{anchor}" style="color:#4ea1ff; font-weight:700; text-decoration:none;">{label}</a>',
            html,
        )

    html = html.replace("\n", "<br>")
    return html


def get_visible_sources(answer_text: str, sources: list[dict]) -> list[dict]:
    """
    Show only cited sources if citations exist in the answer.
    Otherwise fall back to all sources.
    """
    cited_labels = set(extract_citations(answer_text))

    if not cited_labels:
        return sources

    visible = [src for src in sources if src["label"] in cited_labels]
    return visible if visible else sources


def render_sources_section(answer_text: str, sources: list[dict]):
    visible_sources = get_visible_sources(answer_text, sources)

    if not visible_sources:
        return

    with st.expander("Sources used", expanded=False):
        for src in visible_sources:
            st.markdown(
                f'- <a href="#{src["anchor"]}"><b>{src["label"]}</b></a> '
                f'{src["source"]} | page={src["page"]} | chunk={src["chunk_id"]}',
                unsafe_allow_html=True,
            )


def render_context_section(docs):
    if not docs:
        return

    with st.expander("Retrieved context", expanded=False):
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}
            anchor = f"ctx-b{i}"

            st.markdown(
                f'<div id="{anchor}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'**[B{i}] {md.get("source", "unknown")}** | '
                f'page={md.get("page", "")} | '
                f'chunk={md.get("chunk_id", "")}'
            )
            st.write(
                doc.page_content[:1200]
                + ("..." if len(doc.page_content) > 1200 else "")
            )
            st.divider()


# -----------------------------
# State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Demo Controls")

    st.markdown(f"**Model:** `{LLM_MODEL}`")
    st.markdown("**Pipeline:** BM25 + Dense Retrieval + Reranker")

    if st.button("🗑️ Clear chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("Sample questions")

    sample_questions = [
        "What is dense retrieval?",
        "How does BM25 differ from dense retrieval?",
        "What is ColBERT used for?",
    ]

    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["_pending_question"] = q
            st.rerun()


# -----------------------------
# Header
# -----------------------------
st.title("💬 Hybrid RAG Chatbot")
st.caption(f"BM25 + Dense Retrieval + Reranker + {LLM_MODEL}")


# -----------------------------
# Render chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            docs = message.get("docs", [])
            answer_text = message["content"]

            answer_html = make_answer_clickable(answer_text, sources)
            st.markdown(answer_html, unsafe_allow_html=True)

            if not extract_citations(answer_text):
                st.warning("⚠️ No supporting citations found")

            render_sources_section(answer_text, sources)
            render_context_section(docs)
        else:
            st.markdown(message["content"])

# -----------------------------
# Input resolution
# -----------------------------
user_query = None

if "_pending_question" in st.session_state:
    user_query = st.session_state.pop("_pending_question")
else:
    user_query = st.chat_input("Ask a question about your RAG papers...")


# -----------------------------
# Handle new user query
# -----------------------------
if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                answer, sources, docs = ask(user_query)

                answer_html = make_answer_clickable(answer, sources)
                st.markdown(answer_html, unsafe_allow_html=True)

                if not extract_citations(answer):
                    st.warning("⚠️ No supporting citations found")

                render_sources_section(answer, sources)
                render_context_section(docs)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "docs": docs,
                    }
                )

            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )