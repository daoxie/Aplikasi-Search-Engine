"""
Evaluation module — Mini Search Engine
========================================
Menghitung metrik evaluasi IR:
  1. Precision, Recall, F1-Score (untuk satu query di UI)
  2. Average Precision (AP), Reciprocal Rank (RR), NDCG@10 (untuk 5 query standar UAS)
"""

import math
import numpy as np


def evaluate(predicted: list[int], ground_truth: list[int]) -> dict:
    """
    Hitung metrik evaluasi dasar (Precision, Recall, F1-Score) untuk top-10.
    """
    pred_set = set(predicted)
    truth_set = set(ground_truth)

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_ap(predicted: list[int], ground_truth: list[int]) -> float:
    """
    Hitung Average Precision (AP) untuk satu query.
    """
    if not ground_truth:
        return 0.0
    
    pred_list = list(predicted)
    truth_set = set(ground_truth)
    
    ap = 0.0
    hits = 0
    for i, p in enumerate(pred_list):
        if p in truth_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            ap += precision_at_i
            
    return ap / len(ground_truth)


def compute_rr(predicted: list[int], ground_truth: list[int]) -> float:
    """
    Hitung Reciprocal Rank (RR) untuk satu query.
    """
    truth_set = set(ground_truth)
    for i, p in enumerate(predicted):
        if p in truth_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(predicted: list[int], ground_truth: list[int], k: int = 10) -> float:
    """
    Hitung Normalized Discounted Cumulative Gain (NDCG) pada cut-off K.
    """
    if not ground_truth:
        return 0.0
        
    # DCG@K
    dcg = 0.0
    for i in range(min(len(predicted), k)):
        p = predicted[i]
        if p in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
            
    # IDCG@K (Ideal DCG where all relevant documents are placed at the top)
    idcg = 0.0
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg


def evaluate_all_systems(documents: list[str]) -> dict:
    """
    Melakukan evaluasi komparatif untuk 3 sistem:
      1. TF-IDF (Baseline)
      2. SBERT (Semantic)
      3. SBERT + Reranking (Hybrid)
    Menggunakan 5 query standar dan ground truth yang didefinisikan manual.
    """
    queries = [
        "efek samping imunisasi pada bayi",
        "kasus campak pada anak dan balita",
        "jadwal dan jenis vaksin lengkap anak",
        "orang tua yang menolak vaksinasi anak",
        "pentingnya imunisasi untuk kesehatan anak"
    ]

    ground_truths = [
        [1, 23, 24, 25, 29, 41],
        [9, 14, 21, 27, 34, 40],
        [3, 5, 20, 26, 32, 35],
        [6, 8, 30, 33, 37, 44],
        [10, 13, 16, 26, 36, 50]
    ]

    # Konversi ground truth (1-indexed) ke 0-indexed
    gt_0indexed = [[idx - 1 for idx in gt] for gt in ground_truths]

    # Import dependensi lokal (di dalam fungsi untuk menghindari circular imports)
    from modules.preprocessing import preprocess
    from modules.indexing import build_index
    from modules.retrieval import build_tfidf, search as tfidf_search
    from modules.semantic_search import get_document_embeddings, semantic_search, semantic_rerank_search

    # 1. Bangun TF-IDF Baseline
    processed_docs = [preprocess(doc) for doc in documents]
    inv_index = build_index(processed_docs)
    doc_matrix, vocabulary, idf_dict, tf_list = build_tfidf(processed_docs, inv_index)

    # 2. Bangun SBERT Embeddings (menggunakan tuple agar st.cache_data bisa menghitung hash)
    doc_embeddings = get_document_embeddings(tuple(documents))

    systems = ["TF-IDF (Baseline)", "SBERT", "SBERT + Reranking"]
    sys_metrics = {sys: {"ap": [], "rr": [], "ndcg": []} for sys in systems}

    # Jalankan evaluasi
    for query, gt in zip(queries, gt_0indexed):
        # A. Jalankan TF-IDF Search
        q_tokens = preprocess(query)
        tfidf_sim, _ = tfidf_search(q_tokens, vocabulary, idf_dict, doc_matrix)
        tfidf_res = sorted(enumerate(tfidf_sim), key=lambda x: (-x[1], x[0]))
        tfidf_pred = [r[0] for r in tfidf_res if r[1] > 0][:10]

        # B. Jalankan SBERT Search
        sbert_pred, _ = semantic_search(query, documents, doc_embeddings, top_n=10)

        # C. Jalankan SBERT + Reranking
        rerank_pred, _ = semantic_rerank_search(query, documents, doc_embeddings, top_n=10)

        # Hitung metrik
        preds = [tfidf_pred, sbert_pred, rerank_pred]
        for sys_name, pred in zip(systems, preds):
            sys_metrics[sys_name]["ap"].append(compute_ap(pred, gt))
            sys_metrics[sys_name]["rr"].append(compute_rr(pred, gt))
            sys_metrics[sys_name]["ndcg"].append(compute_ndcg(pred, gt, k=10))

    # Rekap hasil rata-rata (MAP, MRR, NDCG@10)
    summary = {}
    for sys_name in systems:
        summary[sys_name] = {
            "MAP": round(float(np.mean(sys_metrics[sys_name]["ap"])), 4),
            "MRR": round(float(np.mean(sys_metrics[sys_name]["rr"])), 4),
            "NDCG": round(float(np.mean(sys_metrics[sys_name]["ndcg"])), 4),
            "ap_list": [round(x, 4) for x in sys_metrics[sys_name]["ap"]],
            "rr_list": [round(x, 4) for x in sys_metrics[sys_name]["rr"]],
            "ndcg_list": [round(x, 4) for x in sys_metrics[sys_name]["ndcg"]],
        }

    return {
        "queries": queries,
        "ground_truths": ground_truths,
        "results": summary
    }