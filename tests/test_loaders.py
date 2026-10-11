import logging
from unittest.mock import MagicMock

import fitz
import pytest
import requests

from loaders import csv_loader, pdf_loader, web_loader


# ── CSV ────────────────────────────────────────────────────────────────

def _write_csv(tmp_path, content: str, name: str = "produtos.csv"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_csv_um_document_por_linha(tmp_path):
    path = _write_csv(tmp_path, "nome,preco,estoque\nNotebook,3500,\nMouse,80,10\n,,\n")
    docs = csv_loader.load_csv(path)

    # linha totalmente vazia é descartada; valores NaN são omitidos.
    # colunas numéricas com NaN são lidas como float pelo pandas.
    assert [d.page_content for d in docs] == [
        "nome: Notebook | preco: 3500.0",
        "nome: Mouse | preco: 80.0 | estoque: 10.0",
    ]
    assert docs[0].metadata == {
        "source": str(path),
        "filename": "produtos.csv",
        "row": 0,
        "total_rows": 2,
    }

    # filtro de colunas e separador customizado
    docs = csv_loader.load_csv(path, columns=["nome", "preco"], row_separator="; ")
    assert docs[1].page_content == "nome: Mouse; preco: 80.0"


def test_load_csv_erros_de_validacao(tmp_path):
    with pytest.raises(FileNotFoundError):
        csv_loader.load_csv(tmp_path / "inexistente.csv")

    with pytest.raises(ValueError, match=r"\.csv"):
        csv_loader.load_csv(_write_csv(tmp_path, "a,b\n1,2\n", name="dados.txt"))

    with pytest.raises(ValueError, match="vazio"):
        csv_loader.load_csv(_write_csv(tmp_path, "a,b\n", name="vazio.csv"))

    with pytest.raises(ValueError, match="Colunas não encontradas"):
        csv_loader.load_csv(_write_csv(tmp_path, "a,b\n1,2\n"), columns=["a", "c"])


# ── PDF ────────────────────────────────────────────────────────────────

def test_load_pdf_um_document_por_pagina_ignorando_vazias(tmp_path):
    path = tmp_path / "doc.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Primeira pagina")
        pdf.new_page()  # página sem texto
        pdf.new_page().insert_text((72, 72), "Terceira pagina")
        pdf.save(path)

    docs = pdf_loader.load_pdf(path)

    assert len(docs) == 2
    assert "Primeira pagina" in docs[0].page_content
    assert "Terceira pagina" in docs[1].page_content
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert all(d.metadata["total_pages"] == 3 for d in docs)
    assert docs[0].metadata["filename"] == "doc.pdf"


def test_load_pdf_erros_de_validacao(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_loader.load_pdf(tmp_path / "inexistente.pdf")

    txt = tmp_path / "doc.txt"
    txt.write_text("não é pdf")
    with pytest.raises(ValueError, match=r"\.pdf"):
        pdf_loader.load_pdf(txt)


# ── Web ────────────────────────────────────────────────────────────────

HTML = """
<html>
  <head><title>  Página de Teste  </title><style>body { color: red; }</style></head>
  <body>
    <header>Cabeçalho</header>
    <nav>Menu</nav>
    <main>
      <h1>Título</h1>


      <p>   Conteúdo principal.   </p>
      <script>alert('x')</script>
    </main>
    <aside>Lateral</aside>
    <footer>Rodapé</footer>
  </body>
</html>
"""


def _fake_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.raise_for_status.return_value = None
    return response


def test_load_web_extrai_texto_limpo(monkeypatch):
    fake_get = MagicMock(return_value=_fake_response(HTML))
    monkeypatch.setattr(web_loader.requests, "get", fake_get)

    doc = web_loader.load_web("https://exemplo.com/artigo")

    assert doc.page_content == "Página de Teste\nTítulo\nConteúdo principal."
    assert doc.metadata == {"source": "https://exemplo.com/artigo", "title": "Página de Teste"}
    fake_get.assert_called_once_with(
        "https://exemplo.com/artigo",
        timeout=web_loader.REQUEST_TIMEOUT,
        headers=web_loader.REQUEST_HEADERS,
    )


def test_load_web_erros_de_validacao_e_requisicao(monkeypatch):
    for url in ["", "ftp://exemplo.com", "exemplo.com"]:
        with pytest.raises(ValueError, match="URL inválida"):
            web_loader.load_web(url)

    def fake_get(*args, **kwargs):
        raise requests.exceptions.Timeout("tempo esgotado")

    monkeypatch.setattr(web_loader.requests, "get", fake_get)
    with pytest.raises(ConnectionError, match="Falha ao acessar"):
        web_loader.load_web("https://exemplo.com")


def test_load_web_batch_pula_urls_com_falha(monkeypatch, caplog):
    def fake_get(url, **kwargs):
        if "quebrada" in url:
            raise requests.exceptions.ConnectionError("recusada")
        return _fake_response("<html><title>Ok</title><body><p>Texto</p></body></html>")

    monkeypatch.setattr(web_loader.requests, "get", fake_get)

    with caplog.at_level(logging.WARNING, logger="loaders.web_loader"):
        docs = web_loader.load_web_batch(
            ["https://ok.com", "https://quebrada.com", "invalida"]
        )

    assert [d.metadata["source"] for d in docs] == ["https://ok.com"]
    skipped = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(skipped) == 2
