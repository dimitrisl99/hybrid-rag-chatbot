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
def make_answer_clickable(answer_text: str, sources: list[dict]) -> str:
    """Replace [B1], [B2] with clickable anchors to retrieved context blocks."""
    html = answer_text

    for src in sources:
        label = src["label"]   # e.g. [B1]
        anchor = src["anchor"] # e.g. ctx-b1
        safe_label = re.escape(label)
        html = re.sub(
            safe_label,
            f'<a href="#{anchor}" style="text-decoration:none;">{label}</a>',
            html
        )

    # preserve line breaks
    html = html.replace("\n", "<br>")
    return html


def reset_chat():
    st.session_state.messages = []


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
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            answer_html = make_answer_clickable(
                message["content"],
                message.get("sources", []),
            )
            st.markdown(answer_html, unsafe_allow_html=True)

            sources = message.get("sources", [])
            docs = message.get("docs", [])

            if sources:
                with st.expander("Sources used", expanded=False):
                    for src in sources:
                        st.markdown(
                            f'- <a href="#{src["anchor"]}">{src["label"]}</a> '
                            f'{src["source"]} | page={src["page"]} | chunk={src["chunk_id"]}',
                            unsafe_allow_html=True,
                        )

            if docs:
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

                if sources:
                    with st.expander("Sources used", expanded=False):
                        for src in sources:
                            st.markdown(
                                f'- <a href="#{src["anchor"]}">{src["label"]}</a> '
                                f'{src["source"]} | page={src["page"]} | chunk={src["chunk_id"]}',
                                unsafe_allow_html=True,
                            )

                if docs:
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