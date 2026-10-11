import logging
from pathlib import Path

from loaders import load_pdf, load_web, load_web_batch, load_csv
from processing import clean_documents, chunk_documents, ChunkStrategy
from embeddings import embed_documents, embed_query
from store import add_documents, query, collection_info
from models import Document

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str | Path) -> int:
    """
    Processa e indexa um arquivo PDF.

    Returns:
        Número de chunks indexados.
    """
    logger.info("Ingerindo PDF: %s", file_path)
    docs = load_pdf(file_path)
    return _process_and_index(docs)


def ingest_web(urls: str | list[str]) -> int:
    """
    Processa e indexa uma ou mais URLs.

    Returns:
        Número de chunks indexados.
    """
    if isinstance(urls, str):
        urls = [urls]

    logger.info("Ingerindo %d URL(s)...", len(urls))
    docs = load_web_batch(urls)
    return _process_and_index(docs)


def ingest_csv(file_path: str | Path, columns: list[str] | None = None) -> int:
    """
    Processa e indexa um arquivo CSV.

    Args:
        columns: colunas a incluir (None = todas)

    Returns:
        Número de chunks indexados.
    """
    logger.info("Ingerindo CSV: %s", file_path)
    docs = load_csv(file_path, columns=columns)
    return _process_and_index(docs)


def search(query_text: str, n_results: int = 5) -> list[Document]:
    """
    Busca os chunks mais relevantes para uma pergunta.

    Args:
        query_text: pergunta em linguagem natural
        n_results:  número de resultados

    Returns:
        Lista de Documents ordenados por relevância.
    """
    logger.info("Buscando: '%s'", query_text)
    query_vector = embed_query(query_text)
    results = query(query_vector, n_results=n_results)

    for i, doc in enumerate(results, start=1):
        score = doc.metadata.get("score", "?")
        source = doc.metadata.get("source", "?")
        logger.info("  [%d] score=%s | source=%s", i, score, source)

    return results


def status() -> None:
    """Exibe informações sobre o estado atual do índice."""
    info = collection_info()
    logger.info("Collection: %s | Chunks indexados: %s", info["name"], info["count"])


# ── função interna ─────────────────────────────────────────────────────

def _process_and_index(
    docs: list[Document],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
) -> int:
    """
    Aplica limpeza, chunking e indexação em uma lista de Documents.

    Returns:
        Número de chunks indexados.
    """
    if not docs:
        logger.warning("Nenhum documento encontrado.")
        return 0

    logger.info("%d documento(s) carregado(s).", len(docs))

    cleaned = clean_documents(docs)
    logger.info("%d documento(s) após limpeza.", len(cleaned))

    chunks = chunk_documents(cleaned, strategy=strategy)
    logger.info("%d chunk(s) gerado(s).", len(chunks))

    embedded = embed_documents(chunks)
    add_documents(embedded)

    return len(chunks)