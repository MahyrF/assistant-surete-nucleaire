"""
Chargement des documents source.

Étape volontairement simple : on part de PDF stockés localement dans
data/raw/. Pas de scraping automatique du site ASN pour rester dans les
clous des CGU du site — les documents sont à télécharger manuellement.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class RawDocument:
    doc_id: str          # nom de fichier sans extension, utilisé comme identifiant
    source_path: str      # chemin du fichier source
    title: str            # titre du document (à défaut, le nom de fichier)
    pages: list[str]      # texte brut de chaque page, dans l'ordre


def load_pdf(path: Path) -> RawDocument:
    """Extrait le texte d'un PDF, page par page.

    On garde le découpage par page à ce stade : c'est le chunking (étape
    suivante) qui décidera comment redécouper ce texte en unités plus
    pertinentes pour la recherche.
    """
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Nettoyage minimal : on ne veut pas complexifier le chunking avec
        # des espaces multiples ou des retours à la ligne parasites issus
        # de l'extraction PDF.
        cleaned = " ".join(text.split())
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
