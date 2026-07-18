# 💉 Analisis Sentimen Opini Vaksinasi Balita Menggunakan Semantic Search Berbasis Sentence-BERT

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![NLP Engine](https://img.shields.io/badge/NLP-Sentence--BERT-yellow.svg)
![Vector Search](https://img.shields.io/badge/Search-FAISS-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

Aplikasi **Temu Kembali Informasi (Information Retrieval) & Analisis Sentimen** tingkat lanjut untuk mengolah dan menganalisis kumpulan dokumen opini masyarakat mengenai **Vaksinasi Balita** dalam Bahasa Indonesia. 

Aplikasi ini membandingkan metode tradisional berbasis frekuensi (**TF-IDF + Cosine Similarity**) dengan pendekatan Neural Search berbasis Vektor Vektor Padat (**Sentence-BERT + FAISS**) dan **Cross-Encoder Reranker**, serta dilengkapi fitur **Analisis Sentimen Zero-Shot** tanpa data training berlabel.

---

## 🚀 Fitur Utama

- **🔍 Multi-Method Search Engine**:
  - **Baseline (TF-IDF + Cosine Similarity)**: Pencarian leksikal dengan pembobotan *log-frequency weighting* dan *inverted index*.
  - **Semantic Search (SBERT + FAISS)**: Pencarian makna semantik menggunakan model `paraphrase-multilingual-MiniLM-L12-v2` dan pustaka indeks pencarian vektor cepat FAISS (`IndexFlatIP`).
  - **Neural Reranking (SBERT + FAISS + Cross-Encoder)**: Pencarian dua tahap (*two-stage retrieval*) yang diperhalus dengan model `cross-encoder/ms-marco-MiniLM-L-6-v2` untuk menangkap konteks interaktif antara query dan dokumen.
- **💬 Analisis Sentimen Zero-Shot (Prototype-based)**:
  - Klasifikasi otomatis opini menjadi kategori **Positif**, **Negatif**, atau **Netral** tanpa memerlukan data training berlabel (*zero-shot classification*).
  - Mengukur Cosine Similarity antara embedding dokumen dengan embedding prototipe referensi per kelas sentimen.
- **📊 TF-IDF & Preprocessing Explorer**:
  - Visualisasi detail bobot TF, IDF, dan TF-IDF per dokumen.
  - Demonstrasi interaktif 6 tahap pra-pemrosesan teks Bahasa Indonesia (Case folding, Punctuation removal, Tokenisasi, Stopword removal, Stemming Sastrawi).
- **📈 Evaluasi Sistem Terpadu**:
  - Evaluasi manual per query (Precision, Recall, F1-Score).
  - Evaluasi komparatif otomatis pada 5 query uji standar berbasis **MAP** (*Mean Average Precision*), **MRR** (*Mean Reciprocal Rank*), dan **NDCG@10** (*Normalized Discounted Cumulative Gain*).

---

## 📊 Hasil Evaluasi Komparatif Sistem

Berdasarkan pengujian pada 5 query uji standar:

| Metode / Sistem | MAP | MRR | NDCG@10 |
| :--- | :---: | :---: | :---: |
| **TF-IDF + Cosine Similarity (Baseline)** | 0.1583 | **0.7067** | 0.3241 |
| **Sentence-BERT + FAISS** | 0.2218 | 0.5333 | 0.3892 |
| **SBERT + FAISS + Cross-Encoder Reranker** | **0.2684** | 0.5167 | **0.4214** |

> **Analisis Singkat:**
> - **SBERT + Cross-Encoder** menghasilkan nilai **MAP** dan **NDCG@10** tertinggi (meningkat ~69.5% pada MAP dibanding TF-IDF), membuktikan keunggulan pencarian semantik kontekstual.
> - **TF-IDF** memiliki **MRR** tertinggi pada query dengan kata kunci yang sangat spesifik (*exact match*).

---

## 🛠️ Teknologi & Modul Utama

* **Frontend / UI**: [Streamlit](https://streamlit.io/)
* **NLP & Embeddings**: [Sentence-Transformers (SBERT)](https://www.sbert.net/) (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Neural Reranking**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
* **Vector Search Engine**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
* **Text Preprocessing**: [Sastrawi (Stemmer & Stopword Indonesian)](https://github.com/sastrawi/sastrawi)
* **Data Processing**: Pandas, NumPy, Scikit-Learn

---

## 👥 Pengembang / Penulis

- **Muhammad Amirun Nadhif** (NIM: 3012310019)
- **Muhammad Bimo Adyatma Broto** (NIM: 3012310022)

*Departemen Informatika — Universitas Internasional Semen Indonesia (UISI)*
