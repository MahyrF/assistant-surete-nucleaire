"""
Construction de l'index hybride : BM25 (sparse/lexical) + embeddings
(dense/sémantique).

On sauvegarde les deux index séparément plutôt que de les fusionner dès
maintenant : la fusion (RRF) se fait au moment de la recherche, pas de
l'indexation, car le poids relatif dense/sparse peut être ajusté sans
tout réindexer.

Modèle d'embeddings choisi : un modèle multilingue léger, correct sur du
français technique, exécutable sur CPU pour un projet perso. À faire
évoluer vers un modèle plus fort (ex: camembert-based) une fois la
mécanique validée.
"""

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config
from ingestion.chunking import Chunk, chunk_all
from ingestion.loaders import load_all_pdfs

RAW_DIR = Path(config.paths.raw_dir)
PROCESSED_DIR = Path(config.paths.processed_dir)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenisation simple pour BM25 : minuscule + découpage sur non-alphanum.

    BM25 n'a pas besoin d'une tokenisation sophistiquée : il compte des
    occurrences de mots. L'important est d'appliquer exactement la même
    fonction au moment de l'indexation et au moment de la requête.
    """
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def build_dense_index(chunks: list[Chunk]) -> np.ndarray:
    model = SentenceTransformer(config.indexing.embedding_model)
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # pour pouvoir utiliser un simple produit scalaire
    )
    return np.array(embeddings)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Chargement des PDF depuis {RAW_DIR}...")
    docs = load_all_pdfs(RAW_DIR)
    print(f"{len(docs)} document(s) chargé(s).")

    print("Découpage en chunks...")
    chunks = chunk_all(docs)
    print(f"{len(chunks)} chunks générés.")

    print("Construction de l'index BM25...")
    bm25_index = build_bm25_index(chunks)

    print("Construction de l'index dense (peut prendre quelques minutes)...")
    dense_embeddings = build_dense_index(chunks)

    print("Sauvegarde des artefacts...")
    with open(PROCESSED_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(PROCESSED_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    np.save(PROCESSED_DIR / "dense_embeddings.npy", dense_embeddings)

    print(f"Terminé. Artefacts écrits dans {PROCESSED_DIR}/")
    print(f"  - chunks.pkl            : {len(chunks)} chunks avec métadonnées")
    print(f"  - bm25_index.pkl        : index lexical BM25")
    print(f"  - dense_embeddings.npy  : matrice ({dense_embeddings.shape})")


if __name__ == "__main__":
    main()
