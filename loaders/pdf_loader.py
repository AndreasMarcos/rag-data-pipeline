import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from models import Document



def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Extrai texto de um PDF página por página.

    Args:
        file_path: caminho para o arquivo .pdf

    Returns:
        Lista de Documents, um por página com conteúdo não vazio.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se o arquivo não for um PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Esperado arquivo .pdf, recebido: {path.suffix}")

    documents = []

    with fitz.open(path) as pdf:
        total_pages = len(pdf)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()

            # ignora páginas sem conteúdo (ex: páginas de imagem pura)
            if not text.strip():
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                }
            ))

    return documents