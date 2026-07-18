"""
Indexing module — Mini Search Engine
=====================================
Membangun Inverted Index dari koleksi dokumen yang sudah di-preprocess.
"""

from collections import defaultdict


def build_index(processed_docs: list[list[str]]) -> dict[str, list[int]]:
    """
    Bangun inverted index.

    Parameters
    ----------
    processed_docs : list of list of str
        Setiap elemen adalah list token hasil preprocessing.

    Returns
    -------
    dict[str, list[int]]
        Mapping term → list of document indices yang mengandung term tsb.
    """
    inverted = defaultdict(list)

    for doc_id, doc_tokens in enumerate(processed_docs):
        for term in set(doc_tokens):  # set() agar setiap term hanya dicatat 1x per doc
            inverted[term].append(doc_id)

    return dict(inverted)