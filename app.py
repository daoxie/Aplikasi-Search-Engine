# =====================================================
#  ANALISIS SENTIMEN OPINI VAKSINASI BALITA
#  Menggunakan Semantic Search Berbasis Sentence-BERT
#  Temu Kembali Informasi | Projek UTS
# =====================================================
#  Fitur:
#   • Pre-processing (case folding, punctuation removal,
#     stop-word removal, stemming Sastrawi)
#   • Inverted Index
#   • TF-IDF (log frequency weighting)
#   • Vector Space Model + Cosine Similarity
#   • Ranked Retrieval
#   • Evaluasi (Precision, Recall, F1)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from collections import Counter

# ---- Custom modules ----
from modules.preprocessing import preprocess, preprocess_steps
from modules.indexing import build_index
from modules.retrieval import build_tfidf, search
from modules.evaluation import evaluate, evaluate_all_systems
from modules.semantic_search import get_document_embeddings, semantic_search, semantic_rerank_search
from modules.sentiment import classify_sentiment, classify_batch, SENTIMENT_COLORS, SENTIMENT_EMOJI

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Analisis Sentimen Opini Vaksinasi Balita — Semantic Search Berbasis Sentence-BERT",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Load external CSS ----
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---- Google Font ----
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# =====================================================
# HEADER BANNER
# =====================================================
st.markdown(
    """
    <div class="header-banner">
        <h1 style="font-size: 2.2rem; line-height: 1.3; margin-bottom: 10px;">Analisis Sentimen Opini Vaksinasi Balita Menggunakan Semantic Search Berbasis Sentence-BERT</h1>
        <p>Sistem Temu Kembali Informasi Opini: Tradisional (TF-IDF) vs. Modern (SBERT &amp; Reranker)</p>
        <span class="badge">Sentence-BERT (MiniLM-L12) · FAISS Vector Search · Cross-Encoder Reranker · Sastrawi Stemmer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# LOAD DATASET
# =====================================================
DEFAULT_DATASET = Path(__file__).parent / "dataset_opini.csv"


@st.cache_data(show_spinner=False)
def load_dataset(file_or_path):
    """Load dataset CSV, handles header quirks."""
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    try:
        df = pd.read_csv(file_or_path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, encoding="utf-8", header=2, on_bad_lines="skip")

    # Auto-detect if real header is on row 2
    if "Data" not in df.columns and "Unnamed: 2" in df.columns:
        first_value = str(df["Unnamed: 2"].iloc[0]).strip()
        if first_value == "Data":
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding="utf-8", header=2, on_bad_lines="skip")

    return df


uploaded = st.file_uploader(" Upload Dataset CSV (opsional)", type=["csv"])

if uploaded:
    df = load_dataset(uploaded)
    st.success(" Dataset berhasil di-upload!")
else:
    if DEFAULT_DATASET.exists():
        df = load_dataset(str(DEFAULT_DATASET))
        st.info(" Menggunakan dataset default: **dataset_opini.csv**")
    else:
        st.error(" Dataset default tidak ditemukan. Silakan upload file CSV.")
        st.stop()

# ---- Determine text column ----
if "Data" in df.columns:
    TEXT_COL = "Data"
elif "Unnamed: 2" in df.columns:
    TEXT_COL = "Unnamed: 2"
else:
    st.error(" Kolom teks tidak ditemukan. Pastikan CSV memiliki kolom **'Data'**.")
    st.stop()

documents = df[TEXT_COL].fillna("").astype(str).tolist()

# ---- Precompute SBERT Embeddings (cached) ----
sbert_embeddings = get_document_embeddings(tuple(documents))

# =====================================================
# PREPROCESSING & INDEXING (cached)
# =====================================================
@st.cache_data(show_spinner=" Memproses dokumen…")
def process_all(docs):
    """Preprocess all documents, build index & TF-IDF."""
    processed = [preprocess(doc) for doc in docs]
    inv_index = build_index(processed)
    doc_matrix, vocab, idf_dict, tf_list = build_tfidf(processed, inv_index)
    return processed, inv_index, doc_matrix, vocab, idf_dict, tf_list


processed_docs, inverted_index, doc_matrix, vocabulary, idf_dict, tf_list = process_all(
    tuple(documents)
)

# ---- Stats ----
total_docs = len(documents)
total_terms = len(vocabulary)
avg_doc_len = round(np.mean([len(d) for d in processed_docs]), 1) if processed_docs else 0

# ---- Statistics Cards ----
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-value">{total_docs}</div>
            <div class="stat-label">Total Dokumen</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-value">{total_terms}</div>
            <div class="stat-label">Term Unik</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-value">{avg_doc_len}</div>
            <div class="stat-label">Rata-rata Token/Dokumen</div>
        </div>""",
        unsafe_allow_html=True,
    )
st.markdown("---")

# =====================================================
# TABS
# =====================================================
tab_search, tab_tfidf, tab_dataset, tab_eval, tab_sentiment = st.tabs(
    ["🔍 Pencarian", "📊 TF-IDF Detail", "📂 Dataset & Index", "📈 Evaluasi", "💬 Analisis Sentimen"]
)

# ---- Precompute sentimen seluruh dataset (cached) ----
doc_sentiments = classify_batch(tuple(documents))

# =====================================================
# TAB 1: PENCARIAN
# =====================================================
with tab_search:
    st.markdown(
        '<div class="section-title"> Cari Dokumen</div>',
        unsafe_allow_html=True,
    )
    st.caption("Masukkan kata kunci untuk mencari dokumen yang relevan. Sistem menggunakan **TF-IDF + Cosine Similarity** untuk ranking.")

    with st.form("search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Kueri Pencarian",
            placeholder="Contoh: imunisasi campak balita",
            label_visibility="collapsed",
        )
        
        search_method = st.selectbox(
            "Metode Pencarian",
            options=[
                "TF-IDF + Cosine Similarity (Baseline)",
                "Sentence-BERT + FAISS (Semantic Search)",
                "Sentence-BERT + FAISS + Cross-Encoder Reranking"
            ],
            index=0
        )
        
        submitted = st.form_submit_button("🔍 Cari Sekarang", use_container_width=True)

    if submitted and query_input.strip():
        query_tokens = preprocess(query_input)

        if not query_tokens and search_method == "TF-IDF + Cosine Similarity (Baseline)":
            st.warning("⚠️ Tidak ada term yang tersisa setelah preprocessing. Coba kata kunci lain.")
        else:
            if search_method == "TF-IDF + Cosine Similarity (Baseline)":
                st.markdown(
                    f"**Query setelah preprocessing:** `{' '.join(query_tokens)}`"
                )
                similarities, q_vector = search(query_tokens, vocabulary, idf_dict, doc_matrix)

                # Build results
                results_df = pd.DataFrame({
                    "doc_id": range(len(documents)),
                    "document": documents,
                    "similarity": similarities,
                })
                results_df = results_df[results_df["similarity"] > 0].sort_values(
                    "similarity", ascending=False
                ).reset_index(drop=True)
            else:
                st.markdown(
                    f"**Query asli (input model neural):** `{query_input}`"
                )
                if search_method == "Sentence-BERT + FAISS (Semantic Search)":
                    indices, scores = semantic_search(query_input, documents, sbert_embeddings, top_n=len(documents))
                else:
                    indices, scores = semantic_rerank_search(query_input, documents, sbert_embeddings, top_n=len(documents))
                
                results_df = pd.DataFrame({
                    "doc_id": indices,
                    "document": [documents[idx] for idx in indices],
                    "similarity": scores,
                })
                results_df = results_df[results_df["similarity"] > 0.0].reset_index(drop=True)

            if results_df.empty:
                st.info(" Tidak ada dokumen yang cocok dengan kueri Anda.")
            else:
                st.markdown(f"###  Hasil Pencarian — {len(results_df)} dokumen ditemukan")

                # Store in session state for evaluation tab
                st.session_state["last_results"] = results_df
                st.session_state["last_query"] = query_input

                top_n = min(10, len(results_df))

                for rank_idx in range(top_n):
                    row = results_df.iloc[rank_idx]
                    score = row["similarity"]
                    doc_id = int(row["doc_id"])
                    doc_text = row["document"]

                    # Determine level
                    if score >= 0.3:
                        level = "high"
                        badge_class = "gold" if rank_idx == 0 else "silver" if rank_idx == 1 else "bronze"
                        level_label = "Sangat Relevan"
                    elif score >= 0.1:
                        level = "medium"
                        badge_class = "bronze"
                        level_label = "Relevan"
                    else:
                        level = "low"
                        badge_class = "bronze"
                        level_label = "Kurang Relevan"

                    # Truncate long docs
                    display_text = doc_text[:300] + "…" if len(doc_text) > 300 else doc_text

                    score_pct = min(score * 100, 100)

                    # Ambil sentimen dokumen ini
                    sent_info = doc_sentiments[doc_id]
                    sent_label = sent_info["label"]
                    sent_color = SENTIMENT_COLORS[sent_label]
                    sent_emoji = SENTIMENT_EMOJI[sent_label]

                    st.markdown(
                        f"""
                        <div class="result-card {level} animate-in" style="animation-delay: {rank_idx * 0.05}s">
                            <div style="display:flex; align-items:flex-start;">
                                <div class="rank-badge {badge_class}">#{rank_idx + 1}</div>
                                <div style="flex:1;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                        <span style="font-weight:600; color:#e2e8f0;">Dokumen #{doc_id + 1}</span>
                                        <div style="display:flex; gap:8px; align-items:center;">
                                            <span style="background:{sent_color}22; color:{sent_color}; border:1px solid {sent_color}55; border-radius:20px; padding:2px 10px; font-size:0.78rem; font-weight:600;">{sent_emoji} {sent_label}</span>
                                            <span class="step-pill">{level_label} · {score:.4f}</span>
                                        </div>
                                    </div>
                                    <p style="color:rgba(255,255,255,0.7); font-size:0.9rem; line-height:1.6; margin:0;">
                                        {display_text}
                                    </p>
                                    <div class="score-bar">
                                        <div class="score-bar-fill {level}" style="width:{score_pct:.1f}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show full results table
                with st.expander(" Lihat Tabel Lengkap Semua Hasil"):
                    display_df = results_df[["doc_id", "similarity", "document"]].copy()
                    display_df.columns = ["ID Dokumen", "Cosine Similarity", "Teks Dokumen"]
                    display_df["ID Dokumen"] = display_df["ID Dokumen"] + 1
                    st.dataframe(display_df, use_container_width=True)

    elif submitted:
        st.warning("⚠️ Masukkan kata kunci terlebih dahulu.")


# =====================================================
# TAB 2: TF-IDF DETAIL
# =====================================================
with tab_tfidf:
    st.markdown(
        '<div class="section-title">Detail TF-IDF</div>',
        unsafe_allow_html=True,
    )
    st.caption("Lihat bobot TF, IDF, dan TF-IDF untuk setiap term dalam koleksi dokumen.")

    # ---- IDF Table ----
    st.markdown("####  Tabel IDF (Inverse Document Frequency)")
    idf_data = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
    idf_df = pd.DataFrame(idf_data, columns=["Term", "IDF"])
    idf_df["DF (doc freq)"] = idf_df["Term"].apply(lambda t: len(inverted_index.get(t, [])))
    idf_df = idf_df[["Term", "DF (doc freq)", "IDF"]]
    idf_df.index = range(1, len(idf_df) + 1)

    col_idf1, col_idf2 = st.columns([2, 1])
    with col_idf1:
        st.dataframe(idf_df.head(50), use_container_width=True, height=400)
    with col_idf2:
        st.markdown("**Rumus IDF:**")
        st.latex(r"IDF(t) = \log_{10}\left(\frac{N}{df(t)}\right)")
        st.markdown(f"- **N** (total dokumen) = `{total_docs}`")
        st.markdown(f"- **Total term unik** = `{total_terms}`")
        st.markdown("- Term dengan IDF tinggi → jarang muncul di banyak dokumen → lebih diskriminatif")

    # ---- TF-IDF per Document ----
    st.markdown("---")
    st.markdown("####  TF-IDF per Dokumen")
    doc_selector = st.selectbox(
        "Pilih dokumen untuk melihat detail TF-IDF:",
        options=range(total_docs),
        format_func=lambda x: f"Dokumen #{x + 1}: {documents[x][:80]}…" if len(documents[x]) > 80 else f"Dokumen #{x + 1}: {documents[x]}",
    )

    if doc_selector is not None:
        tf_doc = tf_list[doc_selector]
        detail_rows = []
        for term, tf_val in sorted(tf_doc.items(), key=lambda x: x[1], reverse=True):
            idf_val = idf_dict.get(term, 0)
            tfidf_val = tf_val * idf_val
            raw_count = Counter(processed_docs[doc_selector]).get(term, 0)
            detail_rows.append({
                "Term": term,
                "Raw TF": raw_count,
                "TF (1+log₁₀)": round(tf_val, 4),
                "IDF": round(idf_val, 4),
                "TF-IDF": round(tfidf_val, 4),
            })

        detail_df = pd.DataFrame(detail_rows)
        detail_df.index = range(1, len(detail_df) + 1)
        st.dataframe(detail_df, use_container_width=True, height=400)

        # Show formula
        st.markdown("**Rumus:**")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.latex(r"TF(t,d) = 1 + \log_{10}(f_{t,d})")
        with col_f2:
            st.latex(r"W(t,d) = TF(t,d) \times IDF(t)")

    # ---- Top Terms Bar Chart ----
    st.markdown("---")
    st.markdown("####  Top 20 Term (berdasarkan IDF)")
    top_terms = idf_df.head(20)
    st.bar_chart(top_terms.set_index("Term")["IDF"], use_container_width=True, height=350)


# =====================================================
# TAB 3: DATASET & INDEX
# =====================================================
with tab_dataset:
    st.markdown(
        '<div class="section-title"> Dataset &amp; Inverted Index</div>',
        unsafe_allow_html=True,
    )

    # ---- Dataset Preview ----
    st.markdown("####  Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True, height=400)

    st.markdown("---")

    # ---- Preprocessing Demo ----
    st.markdown(" Demo Preprocessing")
    st.caption("Pilih dokumen untuk melihat tahapan preprocessing secara detail.")
    demo_doc = st.selectbox(
        "Pilih dokumen:",
        options=range(total_docs),
        format_func=lambda x: f"Doc #{x + 1}: {documents[x][:70]}…" if len(documents[x]) > 70 else f"Doc #{x + 1}: {documents[x]}",
        key="demo_doc_select",
    )

    if demo_doc is not None:
        steps = preprocess_steps(documents[demo_doc])

        step_labels = [
            ("1️ Original", steps["original"]),
            ("2️ Case Folding", steps["case_folded"]),
            ("3️ Punctuation Removal", steps["punctuation_removed"]),
            ("4️ Tokenisasi", ", ".join(steps["tokens"])),
            ("5️ Stop-word Removal", ", ".join(steps["stopword_removed"])),
            ("6️ Stemming (Sastrawi)", ", ".join(steps["stemmed"])),
        ]

        for label, content in step_labels:
            with st.expander(label, expanded=False):
                st.text(content)

    st.markdown("---")

    # ---- Inverted Index ----
    st.markdown("####  Inverted Index")
    st.caption(f"Total **{total_terms}** term unik di-index dari **{total_docs}** dokumen.")

    search_term = st.text_input(" Cari term di index:", placeholder="Ketik term…", key="idx_search")

    if search_term:
        filtered = {k: v for k, v in inverted_index.items() if search_term.lower() in k.lower()}
    else:
        filtered = dict(list(sorted(inverted_index.items()))[:30])

    if filtered:
        idx_rows = []
        for term, doc_ids in sorted(filtered.items()):
            idx_rows.append({
                "Term": term,
                "DF": len(doc_ids),
                "Document IDs": ", ".join(str(d + 1) for d in doc_ids[:20]) + ("…" if len(doc_ids) > 20 else ""),
            })
        idx_df = pd.DataFrame(idx_rows)
        idx_df.index = range(1, len(idx_df) + 1)
        st.dataframe(idx_df, use_container_width=True, height=400)
    else:
        st.info("Tidak ditemukan term yang cocok.")


# =====================================================
# TAB 4: EVALUASI
# =====================================================
with tab_eval:
    st.markdown(
        '<div class="section-title"> Evaluasi Sistem</div>',
        unsafe_allow_html=True,
    )
    st.caption("Hitung **Precision**, **Recall**, dan **F1-Score** dengan membandingkan hasil retrieval terhadap ground truth.")

    if "last_results" in st.session_state and st.session_state["last_results"] is not None:
        last_q = st.session_state.get("last_query", "")
        results_df = st.session_state["last_results"]
        predicted_ids = results_df["doc_id"].head(10).tolist()

        st.markdown(f"**Kueri terakhir:** `{last_q}`")
        st.markdown(f"**Top-10 dokumen retrieved:** `{[int(x)+1 for x in predicted_ids]}`")

        st.markdown("---")
        st.markdown("#### Masukkan Ground Truth")
        st.caption("Tulis nomor dokumen yang benar-benar relevan (1-indexed), pisahkan dengan koma.")

        gt_input = st.text_input(
            "Ground Truth (nomor dokumen)",
            placeholder="Contoh: 1, 4, 10, 14",
            key="gt_input",
        )

        if gt_input:
            try:
                # Parse ground truth (user inputs 1-indexed, convert to 0-indexed)
                gt_ids_1indexed = [int(x.strip()) for x in gt_input.split(",") if x.strip().isdigit()]
                gt_ids = [x - 1 for x in gt_ids_1indexed if 1 <= x <= total_docs]

                if not gt_ids:
                    st.warning(" Masukkan nomor dokumen yang valid (1 s/d jumlah dokumen).")
                else:
                    metrics = evaluate(predicted_ids, gt_ids)

                    # Metric cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f"""<div class="metric-card metric-precision">
                                <div class="label">Precision</div>
                                <div class="value">{metrics['precision']}</div>
                                <div style="font-size:0.8rem; opacity:0.6;">TP / (TP + FP) = {metrics['tp']} / {metrics['tp'] + metrics['fp']}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f"""<div class="metric-card metric-recall">
                                <div class="label">Recall</div>
                                <div class="value">{metrics['recall']}</div>
                                <div style="font-size:0.8rem; opacity:0.6;">TP / (TP + FN) = {metrics['tp']} / {metrics['tp'] + metrics['fn']}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f"""<div class="metric-card metric-f1">
                                <div class="label">F1-Score</div>
                                <div class="value">{metrics['f1']}</div>
                                <div style="font-size:0.8rem; opacity:0.6;">2 × P × R / (P + R)</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    # Detail
                    st.markdown("---")
                    st.markdown("####  Detail Evaluasi")
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown(f"- **True Positives (TP):** `{metrics['tp']}` — dokumen relevan yang berhasil ditemukan")
                        st.markdown(f"- **False Positives (FP):** `{metrics['fp']}` — dokumen tidak relevan yang masuk top-10")
                        st.markdown(f"- **False Negatives (FN):** `{metrics['fn']}` — dokumen relevan yang tidak masuk top-10")
                    with detail_col2:
                        st.markdown("**Rumus:**")
                        st.latex(r"Precision = \frac{TP}{TP + FP}")
                        st.latex(r"Recall = \frac{TP}{TP + FN}")
                        st.latex(r"F_1 = \frac{2 \cdot P \cdot R}{P + R}")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info(" Lakukan pencarian terlebih dahulu di tab ** Pencarian**, lalu kembali ke sini untuk evaluasi.")

    # =====================================================
    # EVALUASI KOMPARATIF (UAS)
    # =====================================================
    st.markdown("---")
    st.markdown(
        '<div class="section-title">🏆 Evaluasi Komparatif (UAS TKI)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Perbandingan performa sistem Baseline (TF-IDF) dengan sistem baru berbasis Semantic Search (SBERT & Cross-Encoder) pada 5 query uji standar.")

    # Caching the full evaluation execution
    @st.cache_data(show_spinner="Menjalankan evaluasi komparatif pada 5 query standar...")
    def run_full_evaluation(docs):
        return evaluate_all_systems(docs)

    eval_data = run_full_evaluation(documents)
    queries_list = eval_data["queries"]
    gt_list = eval_data["ground_truths"]
    eval_results = eval_data["results"]

    # 1. Summary Table
    st.markdown("#### 📊 Tabel Ringkasan Metrik (Rata-rata 5 Query)")
    summary_rows = []
    for sys_name, res in eval_results.items():
        summary_rows.append({
            "Sistem": sys_name,
            "MAP (Mean Average Precision)": f"{res['MAP']:.4f}",
            "MRR (Mean Reciprocal Rank)": f"{res['MRR']:.4f}",
            "NDCG@10": f"{res['NDCG']:.4f}"
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # 2. Grouped Bar Chart
    st.markdown("#### 📈 Visualisasi Perbandingan Performa")
    chart_data = pd.DataFrame({
        "Sistem": list(eval_results.keys()),
        "MAP": [res["MAP"] for res in eval_results.values()],
        "MRR": [res["MRR"] for res in eval_results.values()],
        "NDCG@10": [res["NDCG"] for res in eval_results.values()]
    })
    
    # Render grouped bar chart by setting index to Sistem
    st.bar_chart(chart_data.set_index("Sistem"), height=350, use_container_width=True)

    # Display key highlights
    st.success(
        "💡 **Analisis Singkat:**\n\n"
        "- **SBERT + Reranking** memiliki performa **MAP tertinggi (0.2684)** dan **NDCG@10 tertinggi (0.4214)**, menunjukkan bahwa penggabungan pemahaman makna (Semantic Search) dengan penilaian konteks secara mendalam (Cross-Encoder) terbukti efektif dalam menyempurnakan kualitas pencarian dokumen opini.\n"
        "- **TF-IDF** memiliki **MRR tertinggi (0.7067)** karena pada beberapa query dengan kata kunci yang sangat spesifik (seperti 'campak'), dokumen relevan yang mengandung kata persis diletakkan di posisi pertama, namun ia gagal menemukan variasi sinonim (lemah pada MAP secara keseluruhan)."
    )

    # 3. Query Details
    st.markdown("---")
    st.markdown("#### 🔍 Detail Performa Per Query")
    
    for i, (q, gt) in enumerate(zip(queries_list, gt_list)):
        with st.expander(f"Query #{i+1}: \"{q}\"", expanded=False):
            st.markdown(f"**Ground Truth (Dokumen Relevan):** `{gt}`")
            
            # Table comparing the metrics for this query
            q_metrics = []
            for sys_name, res in eval_results.items():
                q_metrics.append({
                    "Sistem": sys_name,
                    "AP": f"{res['ap_list'][i]:.4f}",
                    "RR": f"{res['rr_list'][i]:.4f}",
                    "NDCG@10": f"{res['ndcg_list'][i]:.4f}"
                })
            st.table(pd.DataFrame(q_metrics))

# =====================================================
# TAB 5: ANALISIS SENTIMEN
# =====================================================
with tab_sentiment:
    st.markdown(
        '<div class="section-title">💬 Analisis Sentimen Opini Vaksinasi</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Klasifikasi otomatis setiap opini ke label **Positif**, **Negatif**, atau **Netral** "
        "menggunakan pendekatan **Zero-Shot berbasis Sentence-BERT** — tanpa data training berlabel."
    )

    # ---- Distribusi Sentimen Dataset ----
    st.markdown("---")
    st.markdown("#### 📊 Distribusi Sentimen Seluruh Dataset")

    label_counts = {"Positif": 0, "Negatif": 0, "Netral": 0}
    for s in doc_sentiments:
        label_counts[s["label"]] += 1
    total_sent = len(doc_sentiments)

    col_pos, col_neg, col_net = st.columns(3)
    for col, label, color in zip(
        [col_pos, col_neg, col_net],
        ["Positif", "Negatif", "Netral"],
        ["#22c55e", "#ef4444", "#94a3b8"],
    ):
        cnt = label_counts[label]
        pct = cnt / total_sent * 100 if total_sent > 0 else 0
        emoji = SENTIMENT_EMOJI[label]
        with col:
            st.markdown(
                f"""<div class="stat-card" style="border-top:3px solid {color};">
                    <div class="stat-value" style="color:{color};">{cnt}</div>
                    <div class="stat-label">{emoji} {label} ({pct:.1f}%)</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # Bar chart distribusi
    import pandas as pd
    dist_df = pd.DataFrame(
        {"Sentimen": list(label_counts.keys()), "Jumlah": list(label_counts.values())}
    ).set_index("Sentimen")
    st.bar_chart(dist_df, height=280, use_container_width=True)

    # ---- Demo: Klasifikasi Teks Manual ----
    st.markdown("---")
    st.markdown("#### 🔬 Coba Klasifikasikan Teks Opini")
    st.caption("Masukkan teks opini vaksinasi balita, sistem akan mendeteksi sentimennya secara otomatis.")

    demo_text = st.text_area(
        "Teks Opini",
        placeholder="Contoh: Anak saya demam tinggi setelah divaksin, saya jadi takut...",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("🔍 Analisis Sentimen", use_container_width=True, key="btn_sentiment"):
        if demo_text.strip():
            with st.spinner("Menganalisis sentimen..."):
                result = classify_sentiment(demo_text.strip())

            s_label = result["label"]
            s_color = SENTIMENT_COLORS[s_label]
            s_emoji = SENTIMENT_EMOJI[s_label]
            s_scores = result["scores"]
            s_conf = result["confidence"]

            st.markdown(
                f"""<div style="background:{s_color}18; border:2px solid {s_color}55;
                    border-radius:14px; padding:20px 24px; margin:12px 0;">
                    <div style="font-size:2rem; margin-bottom:6px;">{s_emoji}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:{s_color}; margin-bottom:4px;">{s_label}</div>
                    <div style="color:rgba(255,255,255,0.6); font-size:0.85rem;">Confidence gap: {s_conf:.4f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("**Skor Kemiripan per Kelas (Cosine Similarity dengan prototipe):**")
            for lbl in ["Positif", "Negatif", "Netral"]:
                sc = s_scores[lbl]
                clr = SENTIMENT_COLORS[lbl]
                emoji = SENTIMENT_EMOJI[lbl]
                bar_pct = max(0, min(sc * 100, 100))
                st.markdown(
                    f"""<div style="margin:6px 0;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                            <span style="color:{clr}; font-weight:600;">{emoji} {lbl}</span>
                            <span style="color:rgba(255,255,255,0.7);">{sc:.4f}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.08); border-radius:6px; height:8px;">
                            <div style="background:{clr}; width:{bar_pct:.1f}%; height:8px;
                                border-radius:6px; transition:width 0.4s;"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("⚠️ Masukkan teks opini terlebih dahulu.")

    # ---- Tabel Sentimen Seluruh Dokumen ----
    st.markdown("---")
    st.markdown("#### 📋 Label Sentimen Per Dokumen")
    st.caption("Tabel di bawah menampilkan hasil klasifikasi sentimen untuk seluruh dokumen dalam dataset.")

    filter_sent = st.selectbox(
        "Filter berdasarkan sentimen:",
        options=["Semua", "Positif", "Negatif", "Netral"],
        key="filter_sentiment",
    )

    sent_rows = []
    for i, (doc, sent) in enumerate(zip(documents, doc_sentiments)):
        if filter_sent != "Semua" and sent["label"] != filter_sent:
            continue
        sent_rows.append({
            "No": i + 1,
            "Sentimen": f"{SENTIMENT_EMOJI[sent['label']]} {sent['label']}",
            "Skor Positif": round(sent["scores"]["Positif"], 4),
            "Skor Negatif": round(sent["scores"]["Negatif"], 4),
            "Skor Netral": round(sent["scores"]["Netral"], 4),
            "Teks Opini": doc[:120] + "…" if len(doc) > 120 else doc,
        })

    if sent_rows:
        sent_df = pd.DataFrame(sent_rows).set_index("No")
        st.dataframe(sent_df, use_container_width=True, height=450)
    else:
        st.info("Tidak ada dokumen dengan sentimen tersebut.")

    # ---- Penjelasan Metode ----
    st.markdown("---")
    with st.expander("ℹ️ Bagaimana cara kerja analisis sentimen ini?", expanded=False):
        st.markdown("""
        **Pendekatan: Zero-Shot Prototype-based Classification**

        Sistem ini **tidak memerlukan data training berlabel**. Caranya:

        1. **Prototipe Sentimen:** Untuk setiap kelas (Positif/Negatif/Netral),
           dibuat sekumpulan kalimat referensi yang representatif dalam Bahasa Indonesia.

        2. **Embedding Prototipe:** Kalimat-kalimat referensi di-encode menggunakan
           model SBERT (`paraphrase-multilingual-MiniLM-L12-v2`), lalu dirata-ratakan
           (*mean pooling*) menjadi satu vektor representatif per kelas.

        3. **Klasifikasi:** Dokumen opini di-encode, lalu dihitung **Cosine Similarity**
           antara embedding dokumen dengan setiap vektor prototipe.
           Kelas dengan similarity tertinggi dipilih sebagai label sentimen.

        **Formula:**
        ```
        label = argmax { cosine_sim(embed(dokumen), prototype(kelas)) }
                 kelas ∈ {Positif, Negatif, Netral}
        ```
        """)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:20px 0 10px; color:rgba(255,255,255,0.3); font-size:0.8rem;">
        Analisis Sentimen Opini Vaksinasi Balita — Projek UTS Temu Kembali Informasi<br>
        Semantic Search Berbasis Sentence-BERT · TF-IDF · Cosine Similarity · Sastrawi Stemmer
    </div>
    """,
    unsafe_allow_html=True,
)
