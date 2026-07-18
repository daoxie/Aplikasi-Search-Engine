"""
Semantic Search module — Mini Search Engine
============================================
Features:
  - Sentence-BERT (SBERT) embedding extraction with Streamlit caching.
  - FAISS index flat IP (Inner Product) search for fast Cosine Similarity.
  - Cross-Encoder reranking for contextual query-document matching.
"""

import streamlit as st
import numpy as np
import faiss
import math
from sentence_transformers import SentenceTransformer, CrossEncoder

SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource(show_spinner="Memuat model Sentence-BERT...")
def load_sbert_model():
    """Load SBERT model from huggingface with resource caching."""
    return SentenceTransformer(SBERT_MODEL_NAME)


@st.cache_resource(show_spinner="Memuat model Cross-Encoder Reranker...")
def load_cross_encoder_model():
    """Load Cross-Encoder model from huggingface with resource caching."""
    return CrossEncoder(RERANKER_MODEL_NAME)


@st.cache_data(show_spinner="Mengekstraksi representasi embedding dokumen...")
def get_document_embeddings(documents):
    """
    Extract embeddings for all documents and cache the results.
    documents : tuple of str
        The documents to embed. We use tuple to make it hashable for st.cache_data.
    """
    model = load_sbert_model()
    # model.encode supports list of texts, return numpy array
    embeddings = model.encode(list(documents), convert_to_numpy=True)
    return embeddings


def semantic_search(query: str, documents: list[str], doc_embeddings: np.ndarray, top_n: int = 10):
    """
    Perform semantic search using SBERT + FAISS with Cosine Similarity.
    
    Returns
    -------
    tuple : (indices, scores)
        - indices : list of doc ids (0-indexed)
        - scores : list of cosine similarity scores
    """
    model = load_sbert_model()
    # Embed query
    q_emb = model.encode([query], convert_to_numpy=True)
    
    # Normalize for Cosine Similarity
    q_emb_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    doc_embeddings_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # FAISS Index
    dimension = doc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(doc_embeddings_norm)
    
    # Search top_n
    scores, indices = faiss_index.search(q_emb_norm, top_n)
    
    # Filter out negative indices (FAISS returns -1 for missing neighbors if top_n > total documents)
    valid_indices = []
    valid_scores = []
    for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
        if idx >= 0:
            valid_indices.append(idx)
            valid_scores.append(score)
            
    return valid_indices, valid_scores


def semantic_rerank_search(query: str, documents: list[str], doc_embeddings: np.ndarray, top_n: int = 10, retrieve_k: int = 25):
    """
    Perform semantic search using SBERT + FAISS first, then rerank candidates using Cross-Encoder.
    
    Returns
    -------
    tuple : (indices, scores)
        - indices : list of reranked doc ids (0-indexed)
        - scores : list of sigmoid-normalized relevance scores [0, 1]
    """
    # 1. Retrieve candidate document IDs using semantic search
    cand_ids, _ = semantic_search(query, documents, doc_embeddings, top_n=retrieve_k)
    
    # 2. Rerank candidates using Cross-Encoder
    cross_encoder = load_cross_encoder_model()
    cand_texts = [documents[idx] for idx in cand_ids]
    pairs = [[query, text] for text in cand_texts]
    
    # Predict logits
    logits = cross_encoder.predict(pairs)
    
    # Sigmoid function to map logit scores to [0, 1] range for UI progress bar
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-float(x)))
    
    scores = [sigmoid(l) for l in logits]
    
    # Sort candidates based on score in descending order
    reranked = sorted(zip(cand_ids, scores), key=lambda x: -x[1])
    
    final_ids = [r[0] for r in reranked[:top_n]]
    final_scores = [r[1] for r in reranked[:top_n]]
    
    return final_ids, final_scores
