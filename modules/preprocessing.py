"""
Pre-processing module — Mini Search Engine
============================================
Tahapan:
  1. Case folding        → lower()
  2. Punctuation removal → regex
  3. Stop-word removal   → Sastrawi
  4. Stemming            → Sastrawi (Bahasa Indonesia)
"""

import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---- cache di module-level agar hanya diinisialisasi sekali ----
_stemmer = StemmerFactory().create_stemmer()
_stopwords = set(StopWordRemoverFactory().get_stop_words())


def preprocess(text: str) -> list[str]:
    """
    Full preprocessing pipeline.
    Return list of stemmed tokens.
    """
    # 1. Case folding
    text = str(text).lower()

    # 2. Punctuation removal (hapus semua non-alphanumeric kecuali spasi)
    text = re.sub(r"[^\w\s]", "", text)

    # 3. Tokenisasi
    words = text.split()

    # 4. Stop-word removal
    words = [w for w in words if w not in _stopwords]

    # 5. Stemming (Sastrawi)
    words = [_stemmer.stem(w) for w in words]

    return words


def preprocess_steps(text: str) -> dict:
    """
    Return detail tiap tahap preprocessing (untuk ditampilkan di UI).
    """
    original = str(text)

    # Case folding
    folded = original.lower()

    # Punctuation removal
    cleaned = re.sub(r"[^\w\s]", "", folded)

    # Tokenisasi
    tokens = cleaned.split()

    # Stop-word removal
    filtered = [w for w in tokens if w not in _stopwords]

    # Stemming
    stemmed = [_stemmer.stem(w) for w in filtered]

    return {
        "original": original,
        "case_folded": folded,
        "punctuation_removed": cleaned,
        "tokens": tokens,
        "stopword_removed": filtered,
        "stemmed": stemmed,
    }