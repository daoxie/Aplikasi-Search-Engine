"""
Sentiment Analysis module — Zero-Shot berbasis Sentence-BERT
=============================================================
Mengklasifikasikan teks opini vaksinasi balita ke label:
  - Positif  : mendukung vaksinasi
  - Negatif  : menolak / khawatir terhadap vaksinasi
  - Netral   : informasi / tidak berpihak

Pendekatan: Prototype-based Zero-Shot Classification
  1. Buat embedding "prototype" kalimat representatif tiap sentimen.
  2. Embedding dokumen dibandingkan (Cosine Similarity) dengan tiap prototype.
  3. Label dengan similarity tertinggi dipilih sebagai sentimen.

Tidak memerlukan data training berlabel!
"""

import numpy as np
import streamlit as st
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Kalimat Prototipe per Sentimen (Bahasa Indonesia)
# -------------------------------------------------------
PROTOTYPE_SENTENCES = {
    "Positif": [
        "vaksinasi anak sangat penting untuk melindungi dari penyakit berbahaya",
        "saya mendukung program imunisasi balita karena terbukti aman dan efektif",
        "vaksin campak polio DPT wajib diberikan untuk kesehatan anak",
        "imunisasi dasar lengkap adalah hak setiap anak untuk hidup sehat",
        "terima kasih sudah memberikan vaksin kepada anak saya agar terlindungi",
        "vaksinasi berhasil mencegah penyakit menular dan melindungi komunitas",
        "anak yang divaksin lebih sehat dan terlindungi dari wabah penyakit",
    ],
    "Negatif": [
        "vaksin berbahaya dan menyebabkan efek samping serius pada anak",
        "saya menolak vaksinasi karena tidak percaya kandungannya aman",
        "imunisasi tidak diperlukan dan hanya membuat anak sakit demam tinggi",
        "vaksin mengandung bahan kimia berbahaya yang merusak tubuh anak",
        "program vaksinasi adalah konspirasi yang merugikan masyarakat",
        "anak saya mengalami reaksi buruk setelah disuntik vaksin",
        "menolak vaksin karena bertentangan dengan keyakinan dan agama",
        "vaksin tidak terbukti efektif dan lebih banyak bahayanya",
    ],
    "Netral": [
        "jadwal vaksinasi anak mencakup BCG DPT polio campak hepatitis",
        "informasi tentang jenis imunisasi dan waktu pemberiannya",
        "efek samping umum vaksin adalah demam ringan dan bengkak di tempat suntikan",
        "pemerintah menyediakan vaksin gratis di puskesmas untuk balita",
        "tanya jawab seputar vaksinasi dan imunisasi bayi",
    ],
}

SENTIMENT_COLORS = {
    "Positif": "#22c55e",   # green
    "Negatif": "#ef4444",   # red
    "Netral":  "#94a3b8",   # slate
}

SENTIMENT_EMOJI = {
    "Positif": "✅",
    "Negatif": "❌",
    "Netral":  "⚪",
}


@st.cache_resource(show_spinner="Memuat model Sentiment SBERT...")
def _load_model():
    """Reuse SBERT model yang sudah di-cache oleh semantic_search module."""
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_data(show_spinner="Membangun embedding prototipe sentimen...")
def _get_prototype_embeddings():
    """
    Hitung dan cache embedding rata-rata untuk setiap kelas sentimen.
    Returns dict: { label -> np.ndarray (384,) }
    """
    model = _load_model()
    prototypes = {}
    for label, sentences in PROTOTYPE_SENTENCES.items():
        embs = model.encode(sentences, convert_to_numpy=True)
        # Mean pooling → satu vektor representatif per kelas
        prototypes[label] = embs.mean(axis=0)
    return prototypes


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity antara dua vektor 1-D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def classify_sentiment(text: str) -> dict:
    """
    Klasifikasikan sentimen satu teks.

    Returns
    -------
    dict dengan keys:
        - label     : str ("Positif" / "Negatif" / "Netral")
        - scores    : dict { label -> float (cosine similarity) }
        - confidence: float (selisih skor tertinggi dengan kedua)
    """
    model = _load_model()
    prototypes = _get_prototype_embeddings()

    emb = model.encode([text], convert_to_numpy=True)[0]

    scores = {
        label: _cosine_similarity(emb, proto)
        for label, proto in prototypes.items()
    }

    label = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    return {
        "label": label,
        "scores": scores,
        "confidence": confidence,
    }


@st.cache_data(show_spinner="Menganalisis sentimen seluruh dataset...")
def classify_batch(documents: tuple) -> list[dict]:
    """
    Klasifikasikan sentimen untuk seluruh koleksi dokumen (dengan caching).

    Parameters
    ----------
    documents : tuple of str  (tuple agar hashable untuk st.cache_data)

    Returns
    -------
    list of dict (satu entry per dokumen, format sama dengan classify_sentiment)
    """
    model = _load_model()
    prototypes = _get_prototype_embeddings()

    # Encode semua dokumen sekaligus (batch, efisien)
    all_embs = model.encode(list(documents), convert_to_numpy=True)

    results = []
    for emb in all_embs:
        scores = {
            label: _cosine_similarity(emb, proto)
            for label, proto in prototypes.items()
        }
        label = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        results.append({
            "label": label,
            "scores": scores,
            "confidence": confidence,
        })

    return results
