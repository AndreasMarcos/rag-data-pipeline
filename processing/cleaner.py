import re
from models import Document


def clean_document(doc: Document) -> Document:
    """
    Aplica limpeza de texto em um Document.

    Args:
        doc: Document com texto bruto

    Returns:
        Novo Document com texto limpo e mesmos metadados.
    """
    text = doc.page_content
    text = _remove_non_printable(text)
    text = _normalize_whitespace(text)
    text = _remove_repeated_punctuation(text)
    text = text.strip()

    return Document(
        page_content=text,
        metadata=doc.metadata
    )


def clean_documents(docs: list[Document]) -> list[Document]:
    """
    Aplica limpeza em uma lista de Documents,
    descartando os que ficarem vazios após a limpeza.
    """
    cleaned = [clean_document(doc) for doc in docs]
    return [doc for doc in cleaned if doc.page_content]


# ── funções internas ───────────────────────────────────────────────────

def _remove_non_printable(text: str) -> str:
    """Remove caracteres de controle e não imprimíveis."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _normalize_whitespace(text: str) -> str:
    """Colapsa espaços múltiplos e normaliza quebras de linha."""
    text = re.sub(r"[ \t]+", " ", text)          # espaços/tabs múltiplos → um espaço
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ quebras → duas quebras
    return text


def _remove_repeated_punctuation(text: str) -> str:
    """Remove repetições de pontuação como '....' ou '----'."""
    text = re.sub(r"\.{3,}", "...", text)         # '.....' → '...'
    text = re.sub(r"-{3,}", "---", text)          # '------' → '---'
    text = re.sub(r"_{3,}", "___", text)          # '______' → '___'
    return text