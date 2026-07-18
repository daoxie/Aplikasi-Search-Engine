"""
Retrieval module — Mini Search Engine
=======================================
- TF-IDF weighting (log frequency: 1 + log10(tf))
- Vector Space Model
- Cosine Similarity
"""

import math
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf(processed_docs, inverted_index):
    """
    Hitung TF-IDF dan bangun document-term matrix.

    Returns
    -------
    tuple: (doc_matrix, vocabulary, idf_dict, tf_list)
        - doc_matrix  : np.ndarray shape (N, V) — TF-IDF vectors
        - vocabulary  : list[str] — ordered list of terms
        - idf_dict    : dict[str, float] — IDF per term
        - tf_list     : list[dict[str, float]] — raw TF (log-weighted) per doc
    """
    N = len(processed_docs)

    # ---- TF (log frequency weighting) ----
    tf_list = []
    for doc_tokens in processed_docs:
        freq = Counter(doc_tokens)
        tf_doc = {}
        for term, count in freq.items():
            tf_doc[term] = 1 + math.log10(count) if count > 0 else 0
        tf_list.append(tf_doc)

    # ---- IDF ----
    idf_dict = {}
    for term, doc_ids in inverted_index.items():
        df = len(doc_ids)
        idf_dict[term] = math.log10(N / df) if df > 0 else 0

    # ---- Vocabulary & Document-Term Matrix ----
    vocabulary = sorted(inverted_index.keys())

    doc_matrix = np.zeros((N, len(vocabulary)))
    for i, tf_doc in enumerate(tf_list):
        for j, term in enumerate(vocabulary):
            doc_matrix[i, j] = tf_doc.get(term, 0) * idf_dict.get(term, 0)

    return doc_matrix, vocabulary, idf_dict, tf_list


def search(query_tokens, vocabulary, idf_dict, doc_matrix):
    """
    Cari dokumen berdasarkan query tokens menggunakan Cosine Similarity.

    Parameters
    ----------
    query_tokens : list[str]
        Token-token query yang sudah di-preprocess.
    vocabulary   : list[str]
    idf_dict     : dict[str, float]
    doc_matrix   : np.ndarray

    Returns
    -------
    tuple: (similarities, query_vector)
        - similarities : np.ndarray shape (N,) — cosine similarity per doc
        - query_vector : np.ndarray shape (V,) — query TF-IDF vector
    """
    q_freq = Counter(query_tokens)

    query_vector = np.zeros(len(vocabulary))
    for j, term in enumerate(vocabulary):
        tf = q_freq.get(term, 0)
        if tf > 0:
            tf = 1 + math.log10(tf)
        query_vector[j] = tf * idf_dict.get(term, 0)

    # Cosine similarity
    if np.linalg.norm(query_vector) == 0:
        return np.zeros(doc_matrix.shape[0]), query_vector

    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        doc_matrix
    )[0]

    return similarities, query_vector