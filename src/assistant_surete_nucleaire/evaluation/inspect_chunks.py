"""
Utilitaire d'inspection des chunks générés par l'ingestion.

Sert uniquement à t'aider à écrire le golden dataset à partir de contenu
réel : affiche les chunks avec leur chunk_id exact (à copier dans
golden_dataset.json) et leur texte.

Usage :
    python -m evaluation.inspect_chunks                  # tous les chunks
    python -m evaluation.inspect_chunks --doc mon_guide   # un document précis
    python -m evaluation.inspect_chunks --search "incendie"  # recherche texte
"""

import argparse
import pickle
from pathlib import Path

from ..config import config as cfg


def load_chunks():
    path = Path(cfg.paths.processed_dir) / "chunks.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord l'ingestion : "
            "python -m ingestion.indexer"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", help="Filtrer sur un doc_id précis")
    parser.add_argument("--search", help="Filtrer sur un mot-clé présent dans le texte")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"{len(chunks)} chunks au total dans le corpus.\n")

    for chunk in chunks:
        if args.doc and chunk.doc_id != args.doc:
            continue
        if args.search and args.search.lower() not in chunk.text.lower():
            continue

        print(f"--- {chunk.chunk_id} ---")
        print(f"Document : {chunk.doc_title} (page {chunk.page})")
        print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
        print()


if __name__ == "__main__":
    main()
