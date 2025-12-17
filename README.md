# BM42_Eval

This repository contains the **experimental evaluation pipeline** used in the master’s thesis **BM42 vs. Conventional Methods: Evaluating Next-Generation Hybrid Search Techniques for Information Retrieval** written by *Lucie Erazímová, 2025* as part of a master thesis internship with the company *SoftRobot*.

This project benchmarks a neural sparse retrieval method called **BM42** -- proposed by **Qdrant** -- against traditional and established lexical, dense, and hybrid retrieval approaches across several Information Retrieval (IR) tasks and domains (general, scientific, and medical).

---

## Project Overview

This repository provides a pipeline which **systematically evaluates the strengths and limitations of BM42** in comparison with:

- **BM25** (traditional sparse lexical retrieval)
- **SPLADE** (established neural sparse retrieval)
- **Dense vector retrieval** (OpenAI embeddings + ANN search)
- **Hybrid search** (experimental lexical / neural sparse + dense fusion).

Experiments are performed on selected datasets from the **BEIR benchmark**, all of which are publicly available:

- General-domain deduplication (Quora)
- Biomedical information retrieval (TREC-COVID)
- Scientific fact verification (SciFact)
- retrieve from https://github.com/beir-cellar/beir

The pipeline is constructed with the objective to evaluate and analyze **retrieval quality, ranking quality, interpretability, and robustness across sparse vs. dense relevance distributions**.

---

## What is BM42?

BM42 is a **neural sparse retriever** introduced by Qdrant that:

- Retains the **IDF-based logic of BM25**
- Replaces term-frequency (TF) with **transformer attention weights**
- Produces **interpretable sparse vectors** augmented with contextual relevance

This design aims to combine the **efficiency and interpretability of lexical search** and the **semantic awareness of transformer-based retrieval**. 

---

## Pipeline Architecture

The evaluation pipeline follows the following stages:

1. **Dataset preprocessing**  
   - Normalize IDs (into Qdrant-compatible integers)
   - Filter out long samples (≤ 512 tokens) to avoid chunking

2. **Tokenization**  
   - BM25: lexical tokenization + lemmatization  
   - BM42 / SPLADE: BERT WordPiece tokenization
   - OpeanAI embedding model: cl100k_base (API handles automatically)

3. **Indexing**  
   - Dense embeddings: OpenAI `text-embedding-ada-002`  
   - Sparse vectors: FastEmbed (BM25, BM42)  
   - SPLADE embeddings: FastEmbed (generated in batches due to computational constraints)

4. **Search**  
   - Sparse search (BM25, BM42, SPLADE)
   - Dense ANN search (OpeanAI's `ada-002`)
   - Hybrid search using Reciprocal Rank Fusion (via Qdrant's API)

5. **Evaluation**  
   - Top-10 retrieval  using traditional IR metrics

   - **Precision@10** – noise filtering ability
   - **Recall@10** – ability to retrieve all relevant documents
   - **Hit@10** – whether at least one relevant document is retrieved
   - **nDCG@10** – ranking quality

   - To understand the statistical relevance, results are reported with **95% confidence intervals** using bootstrap resampling.

---

## Technologies and Platforms Used:

- **Qdrant** – vector database and hybrid search engine
- **Tantivy** – full-text BM25 baseline
- **FastEmbed** – sparse vector generation (BM25, BM42)
- **HuggingFace Transformers** – tokenization & SPLADE
- **OpenAI Embeddings** – dense retrieval baseline
- **Azure OpenAI (via LangChain)** – support for generating dense embeddings 

---

## Key Findings (Summary points)

- **Standalone BM42 underperforms** while sparse and dense baselines show a consistently strong performance
- **Hybrid BM42 + dense retrieval** significantly improves in performance, yet does not surpass state-of-the-art models
- **Hybrid BM25 remains the strongest overall performer** across all tasks and domains
- BM42 shows **potential in semantic sensitivity and interpretability** as a hybrid retriever, but further experimentation is needed 

For detailed results, analysis, and qualitative examples, refer to the thesis.

---

## Thesis

**BM42 vs. Conventional Methods: Evaluating Next-Generation Hybrid Search Techniques for Information Retrieval**  
Lucie Erazímová, June 2025

- University: Uppsala University
- Programme: Master’s Programme in Language Technology
- Internship in collaboration with SoftRobot: https://www.softrobot.io/
- Link to the full document: https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1979031&dswid=-6131
---

## Reproducibility Notes

- Experiments are tied to **Qdrant’s BM42 and Tantivy's full text BM25 implementations**
- Results reflect **specific engine and model configurations** only
- To generalize the results, it is strongly recommended to test across other vector databases such as FAISS, Weviate, or Pinecone.

---

## Citation

Erazímová, L. (2025). BM42 vs. Conventional Methods: Evaluating Next-Generation Hybrid Search Techniques for Information Retrieval (Dissertation). Retrieved from https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-562576




