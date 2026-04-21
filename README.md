# Hybrid RAG Chatbot with Reranking and Evaluation

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) system** for question answering over document collections. It combines classical information retrieval techniques with modern neural retrieval and reranking methods to improve answer relevance and grounding.

---

## Overview

The system integrates multiple retrieval and generation components:

- Dense retrieval using embedding models  
- Sparse retrieval (BM25) for lexical matching  
- Hybrid fusion using Reciprocal Rank Fusion (RRF)  
- Cross-encoder reranking for improved relevance  
- LLM-based answer generation using retrieved context  
- Evaluation pipeline with standard IR metrics  

The objective is to explore how combining retrieval strategies impacts performance and answer quality.

---

## System Architecture

The pipeline consists of the following stages:

### 1. Document Ingestion
- PDF loading and preprocessing (PyMuPDF)
- Cleaning and normalization
- Semantic chunking based on similarity

### 2. Indexing
- Embedding generation using HuggingFace models
- Storage of embeddings (NumPy / FAISS)
- Optional ChromaDB persistence

### 3. Retrieval
- Dense retrieval via vector similarity
- Sparse retrieval via BM25
- Hybrid fusion using Reciprocal Rank Fusion (RRF)

### 4. Reranking
- Cross-encoder model (`ms-marco-MiniLM`)
- Re-scoring and reordering of retrieved documents

### 5. Answer Generation
- Local LLM via Ollama (`phi3:mini`)
- Prompt-based generation constrained to retrieved context
- Inline citations (e.g. [B1], [B2])

---

## Evaluation

The project includes an evaluation pipeline supporting:

- Hit@K  
- Recall@K  
- Mean Reciprocal Rank (MRR@K)  

This enables comparison between:
- Dense retrieval  
- BM25 retrieval  
- Hybrid retrieval  

### Example Results

| Method  | Hit@5 | Recall@5 | MRR@5 |
|--------|------|----------|-------|
| Dense  | 0.80 | 0.72     | 0.65  |
| BM25   | 0.76 | 0.70     | 0.60  |
| Hybrid | 0.88 | 0.81     | 0.74  |


---

## Demo

A Streamlit interface is provided:

```bash
streamlit run app.py
```

Features:

- Interactive question answering
- Source attribution
- Retrieved context inspection

## Setup

1. Clone the repository

```
git clone https://github.com/dimitrisl99/hybrid-rag-chatbot.git
cd hybrid-rag-chatbot
```

2. Create virtual environment 

```
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Ollama Setup (LLM)

This project uses a local LLM via Ollama.

**Install Ollama** 

https://ollama.com/

**Pull model**
```
ollama pull phi3:mini
```
**Run model (optional test)**
```
ollama run phi3:mini
```

## Running the Full Pipeline

Follow these steps to run the full RAG pipeline:

1. Process and index documents

```
python -m src.indexing
```

2. Export embeddings / FAISS index

```
python -m src.export_index 
```

3. Run the app 

```
streamlit run app.py
```

## Project Structure

```
src/
  ├── config.py
  ├── indexing.py
  ├── export_index.py
  ├── retriever.py
  ├── reranker.py
  ├── rag_chat.py
  ├── evaluate.py

data/
  ├── raw/          # input PDFs
  ├── processed/    # generated chunks
  ├── index_numpy/  # embeddings + FAISS
  ├── chroma_db/    # vector DB
  ├── eval/         # evaluation queries
```

## Future Work

- Query expansion techniques
- Improved chunking strategies
- Larger and more advanced LLMs
- More extensive evaluation benchmarks
- Support for additional document types

## Author 

Dimitris Loukakis