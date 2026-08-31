"""
Découpage des documents en chunks.

Principe : on ne coupe jamais au milieu d'une phrase. On regroupe des
phrases jusqu'à atteindre une taille cible (en caractères), avec un
recouvrement entre chunks consécutifs pour ne pas perdre le contexte à la
frontière (une exigence réglementaire qui commence à la fin d'un chunk et
finit au début du suivant doit rester compréhensible dans les deux).

Chaque chunk porte des métadonnées : elles servent plus tard à afficher une
citation exacte (document + page) et à filtrer la recherche si besoin.
"""

import re
from dataclasses import dataclass, field

from assistant_surete_nucleaire.config import config as cfg
from assistant_surete_nucleaire.ingestion.loaders import RawDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str
    metadata: dict = field(default_factory=dict)


def split_into_sentences(text: str) -> list[str]:
    """Découpage naïf en phrases.

    Suffisant pour du français réglementaire (peu d'abréviations ambiguës
    hors sigles). Une vraie librairie NLP (spacy, nltk) ferait mieux sur du
    texte plus varié — à envisager si tu observes des coupures aberrantes
    sur ton corpus.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(doc: RawDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    # Utilise pages_with_numbers si disponible, sinon tombe en compatibilité
    if hasattr(doc, "pages_with_numbers") and doc.pages_with_numbers:
        page_iter = doc.pages_with_numbers
    else:
        # Fallback pour les anciennes versions sans pages_with_numbers
        page_iter = [(i + 1, p) for i, p in enumerate(doc.pages)]

    for page_number, page_text in page_iter:
        sentences = split_into_sentences(page_text)
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= cfg.chunking.target_chunk_chars:
                current = candidate
                continue

            if current:
                chunks.append(_make_chunk(doc, page_number, chunk_index, current))
                chunk_index += 1
                overlap_text = current[-cfg.chunking.overlap_chars:]
                current = f"{overlap_text} {sentence}".strip()
            else:
                current = candidate

        if current:
            chunks.append(_make_chunk(doc, page_number, chunk_index, current))
            chunk_index += 1

    return chunks


def _make_chunk(doc: RawDocument, page_number: int, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::p{page_number}::c{index}",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        page=page_number,
        text=text,
        metadata={"source_path": doc.source_path},
    )


def chunk_all(docs: list[RawDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
