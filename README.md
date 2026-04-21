# Hybrid RAG Chatbot with Reranking and Evaluation

This project implements a Hybrid Retrieval-Augmented Generation (RAG) pipeline for question answering over document collections, combining classical information retrieval techniques with modern neural retrieval and reranking.

## Overview

The system integrates multiple retrieval and generation components:

- Dense retrieval using embedding models  
- Sparse retrieval (BM25) for lexical matching  
- Hybrid fusion using Reciprocal Rank Fusion (RRF)  
- Cross-encoder reranking for improved relevance  
- LLM-based answer generation using retrieved context  
- Evaluation pipeline with standard IR metrics  

The goal is to explore how combining multiple retrieval strategies improves answer quality and retrieval performance.

## Architecture

The system consists of the following stages:

1. Document Ingestion  
   - PDF loading and preprocessing  
   - Semantic chunking  

2. Indexing  
   - Embedding generation (HuggingFace models)  
   - Storage using FAISS or NumPy  
   - Optional ChromaDB integration  

3. Retrieval  
   - Dense retrieval based on vector similarity  
   - BM25 retrieval for keyword matching  
   - Hybrid fusion using Reciprocal Rank Fusion (RRF)  

4. Reranking  
   - Cross-encoder model (ms-marco-MiniLM)  
   - Reordering of top retrieved documents  

5. Answer Generation  
   - Prompt-based generation with a local LLM (Ollama)  
   - Context-constrained answers with inline citations  

## Evaluation

The project includes an evaluation pipeline with the following metrics:

- Hit@K  
- Recall@K  
- Mean Reciprocal Rank (MRR@K)  

This allows comparison between:
- Dense retrieval  
- BM25 retrieval  
- Hybrid retrieval  

## Demo

A Streamlit interface is provided for interactive use:

```bash
streamlit run app.py
```

The interface supports:

- Interactive question answering
- Source attribution
- Inspection of retrieved context

## Setup 

```
git clone https://github.com/your-username/hybrid-rag-chatbot.git
cd hybrid-rag-chatbot

pip install -r requirements.txt
```

```
src/
  ├── indexing.py
  ├── retriever.py
  ├── reranker.py
  ├── rag_chat.py
  ├── evaluate.py

data/
  ├── raw/
  ├── processed/
  ├── eval/
  ```


## Future Work 

- Query expansion techniques
- Improved chunking strategies
- Experiments with larger language models
- More extensive evaluation benchmarks

