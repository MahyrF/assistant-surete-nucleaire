"""
Chargement des documents source.

Étape volontairement simple : on part de PDF stockés localement dans
data/raw/. Pas de scraping automatique du site ASN pour rester dans les
clous des CGU du site (les documents sont à télécharger manuellement).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class RawDocument:
    doc_id: str
    source_path: str
    title: str
    pages: list[str]                      # texte de chaque page (sans les pages de sommaire)
    pages_with_numbers: list[tuple[int, str]]  # (numéro de page PDF, texte)


def is_table_of_contents_page(text: str) -> bool:
    """
    Heuristique simple pour détecter une page de sommaire.

    On considère qu'une page est un sommaire si elle commence par le mot
    'Sommaire' ou 'Table des matières', OU si le mot 'Sommaire' apparaît
    dans les 200 premiers caractères (ce qui indique généralement un titre
    de sommaire en haut de page). Pour l'instant, on se contente de cette
    détection sur le titre en première ligne.
    """
    clean = " ".join(text.split()).lower()

    if re.search(r"^(sommaire|table des matières)", clean):
        return True

    if re.search(r"sommaire", clean[:200]):
        return True

    return False


def simplify_doc_id(filename: str) -> str:
    """
    Transforme le nom de fichier en un identifiant court compatible
    avec le golden dataset.

    Exemples :
      - "Guide de l'ASN n°6 - ..." -> "Guide-de-l-ASN-6"
      - "Guide de l'ASN n°23 - ..." -> "Guide-de-l-ASN-23"
    """
    match = re.search(r"n°\s*(\d+)", filename)
    if match:
        number = match.group(1)
        return f"Guide-de-l-ASN-{number}"
    return filename.replace(" ", "-")


def load_pdf(path: Path) -> RawDocument:
    """Extrait le texte d'un PDF, page par page, en excluant les pages de sommaire."""
    reader = PdfReader(str(path))
    pages_texts = []
    pages_with_numbers = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())

        if is_table_of_contents_page(cleaned):
            print(f"  [INFO] Page {page_number} ignorée (sommaire détecté) dans {path.name}")
            continue

        pages_texts.append(cleaned)
        pages_with_numbers.append((page_number, cleaned))

    doc_id = simplify_doc_id(path.stem)

    return RawDocument(
        doc_id=doc_id,
        source_path=str(path),
        title=path.stem.replace("_", " ").replace("-", " "),
        pages=pages_texts,
        pages_with_numbers=pages_with_numbers,
    )


def load_all_pdfs(raw_dir: Path) -> list[RawDocument]:
    """Charge tous les PDF présents dans un dossier."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {raw_dir}. "
            "Place tes documents ASN dans data/raw/ avant de lancer l'ingestion."
        )
    return [load_pdf(p) for p in pdf_paths]