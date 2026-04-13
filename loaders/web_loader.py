import requests
from bs4 import BeautifulSoup
from pathlib import Path
from models import Document

from config import REQUEST_TIMEOUT, REQUEST_HEADERS


def load_web(url: str) -> Document:
    """
    Extrai texto limpo de uma página web.

    Args:
        url: endereço completo da página (com https://)

    Returns:
        Document com o texto extraído e metadados da fonte.

    Raises:
        ValueError: se a URL for inválida ou vazia.
        ConnectionError: se a requisição falhar.
    """
    if not url or not url.startswith("http"):
        raise ValueError(f"URL inválida: {url}")

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers=REQUEST_HEADERS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Falha ao acessar {url}: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # remove tags que não contribuem com conteúdo
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # tenta pegar o título da página para metadado
    title = soup.title.string.strip() if soup.title else url

    text = soup.get_text(separator="\n")
    text = _clean_whitespace(text)

    return Document(
        page_content=text,
        metadata={
            "source": url,
            "title": title,
        }
    )


def load_web_batch(urls: list[str]) -> list[Document]:
    """
    Processa uma lista de URLs, ignorando as que falharem.

    Args:
        urls: lista de endereços

    Returns:
        Lista de Documents das URLs que carregaram com sucesso.
    """
    documents = []
    for url in urls:
        try:
            documents.append(load_web(url))
        except (ValueError, ConnectionError) as e:
            print(f"[web_loader] Pulando {url} — {e}")
    return documents


def _clean_whitespace(text: str) -> str:
    """Remove linhas em branco excessivas e espaços desnecessários."""
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)