"""
Validation de l'alignement entre le golden dataset et l'index.

Compare les expected_chunk_ids du dataset avec les chunk_id réels
présents dans l'index généré par l'ingestion.
"""

import json
import pickle
from pathlib import Path

from assistant_surete_nucleaire.config import config as cfg


def load_chunks() -> list:
    """Charge les chunks depuis le fichier pickle."""
    path = Path(cfg.paths.processed_dir) / "chunks.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord l'ingestion : "
            "python -m assistant_surete_nucleaire.ingestion.indexer"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_golden_dataset() -> dict:
    """Charge le golden dataset depuis le fichier JSON."""
    path = Path(cfg.paths.data_dir) / "golden_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    chunks = load_chunks()
    existing_ids = {c.chunk_id for c in chunks}

    dataset = load_golden_dataset()
    questions = dataset["questions"]

    errors = []
    for question in questions:
        qid = question["id"]
        expected_ids = question.get("expected_chunk_ids", [])

        for expected_id in expected_ids:
            if expected_id not in existing_ids:
                errors.append({
                    "question": qid,
                    "missing_id": expected_id
                })

    if errors:
        print(f"Erreurs : {len(errors)} expected_chunk_ids introuvables")
        for err in errors:
            print(f"  {err['question']} : {err['missing_id']} absent de l'index")
    else:
        print("Tous les expected_chunk_ids sont valides")
        print(f"  {len(questions)} questions vérifiées")
        print(f"  {len(existing_ids)} chunks dans l'index")


if __name__ == "__main__":
    main()