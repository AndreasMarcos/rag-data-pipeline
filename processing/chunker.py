from enum import Enum
from models import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


class ChunkStrategy(Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


def chunk_document(
    doc: Document,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Divide um Document em chunks menores.

    Args:
        doc:          Document limpo
        strategy:     estratégia de chunking
        chunk_size:   tamanho máximo de cada chunk
        chunk_overlap: sobreposição entre chunks

    Returns:
        Lista de Documents com metadados herdados + índice do chunk.
    """
    if strategy == ChunkStrategy.FIXED:
        chunks = _fixed_split(doc.page_content, chunk_size, chunk_overlap)
    elif strategy == ChunkStrategy.RECURSIVE:
        chunks = _recursive_split(doc.page_content, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}. Use ChunkStrategy.FIXED ou ChunkStrategy.RECURSIVE.")

    return [
        Document(
            page_content=chunk,
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def chunk_documents(
    docs: list[Document],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Aplica chunking em uma lista de Documents."""
    result = []
    for doc in docs:
        result.extend(chunk_document(doc, strategy, chunk_size, chunk_overlap))
    return result


# ── funções internas ───────────────────────────────────────────────────

def _fixed_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Corta o texto a cada chunk_size caracteres com overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Tenta quebrar o texto pelos separadores em ordem de prioridade.
    Só avança para o próximo separador se o chunk ainda for grande demais.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    separator = separators[-1]
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # separador vazio: divide caractere a caractere (str.split("") lança ValueError)
    parts = list(text) if separator == "" else text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # parte maior que chunk_size — divide recursivamente
            if len(part) > chunk_size and len(separators) > 1:
                sub = _recursive_split(part, chunk_size, chunk_overlap, separators[1:])
                chunks.extend(sub[:-1])
                current = sub[-1] if sub else ""
            else:
                current = part

    if current:
        chunks.append(current)

    # aplica overlap entre chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, chunk_overlap)

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Adiciona o final do chunk anterior no início do próximo."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        result.append(tail + " " + chunks[i])
    return result