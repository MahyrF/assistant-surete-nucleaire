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
    doc_id: str          # nom de fichier sans extension, utilisé comme identifiant
    source_path: str      # chemin du fichier source
    title: str            # titre du document (à défaut, le nom de fichier)
    pages: list[str]      # texte brut de chaque page, dans l'ordre


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

    # Vérifie si la page commence par "sommaire" ou "table des matières"
    # (certains PDF ajoutent des caractères avant, d'où le re.search en début de chaîne)
    if re.search(r"^(sommaire|table des matières)", clean):
        return True

    # Si le mot "sommaire" apparaît dans les 200 premiers caractères,
    # c'est probablement une page de sommaire (car le titre est en haut)
    if re.search(r"sommaire", clean[:200]):
        return True

    return False


def load_pdf(path: Path) -> RawDocument:
    """Extrait le texte d'un PDF, page par page, en excluant les pages de sommaire."""
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())

        # Ignorer les pages qui sont des sommaires
        if is_table_of_contents_page(cleaned):
            print(f"  [INFO] Page {page_number} ignorée (sommaire détecté) dans {path.name}")
            continue

        pages.append(cleaned)

    return RawDocument(
        doc_id=path.stem,
        source_path=str(path),
        title=path.stem.replace("_", " ").replace("-", " "),
        pages=pages,
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