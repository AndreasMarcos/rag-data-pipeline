from unittest.mock import MagicMock

import pytest

import pipeline
from models import Document
from store import chroma_store


@pytest.fixture
def mock_index(monkeypatch):
    """Substitui embedding e vector store por mocks."""
    fake_embed = MagicMock(side_effect=lambda docs: [(d, [0.0, 1.0]) for d in docs])
    fake_add = MagicMock()
    monkeypatch.setattr(pipeline, "embed_documents", fake_embed)
    monkeypatch.setattr(pipeline, "add_documents", fake_add)
    return fake_embed, fake_add


def test_ingest_csv_limpa_chunka_e_indexa(tmp_path, mock_index):
    fake_embed, fake_add = mock_index
    path = tmp_path / "produtos.csv"
    path.write_text("nome,descricao\nNotebook,Leve    e   rápido\nMouse,Sem fio\n", encoding="utf-8")

    n_chunks = pipeline.ingest_csv(path)

    assert n_chunks == 2
    chunks = fake_embed.call_args.args[0]
    assert [c.page_content for c in chunks] == [
        "nome: Notebook | descricao: Leve e rápido",
        "nome: Mouse | descricao: Sem fio",
    ]
    assert all("chunk_index" in c.metadata for c in chunks)
    indexed = fake_add.call_args.args[0]
    assert len(indexed) == 2
    assert indexed[0][1] == [0.0, 1.0]


def test_ingest_web_aceita_string_unica(monkeypatch, mock_index):
    fake_batch = MagicMock(return_value=[Document("conteúdo da página", {"source": "https://a.com"})])
    monkeypatch.setattr(pipeline, "load_web_batch", fake_batch)

    assert pipeline.ingest_web("https://a.com") == 1
    fake_batch.assert_called_once_with(["https://a.com"])


def test_process_and_index_sem_documentos_nao_indexa(mock_index):
    fake_embed, fake_add = mock_index
    assert pipeline._process_and_index([]) == 0
    # documentos que ficam vazios após a limpeza também não geram chunks
    assert pipeline._process_and_index([Document("   \x00  ")]) == 0
    fake_add.assert_called_once_with([])


def test_search_converte_distancia_em_score(monkeypatch):
    monkeypatch.setattr(pipeline, "embed_query", lambda q: [0.1, 0.2])
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["texto a", "texto b"]],
        "metadatas": [[{"source": "a.pdf"}, {"source": "b.csv"}]],
        "distances": [[0.25, 0.123456]],
    }
    monkeypatch.setattr(chroma_store, "_get_collection", lambda: fake_collection)

    results = pipeline.search("pergunta", n_results=2)

    fake_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )
    assert [r.page_content for r in results] == ["texto a", "texto b"]
    assert results[0].metadata == {"source": "a.pdf", "score": 0.75}
    assert results[1].metadata["score"] == 0.8765



def test_erros_de_validacao_de_query_e_estrategia(mock_index):
    from embeddings import embedder

    with pytest.raises(ValueError, match="vazia"):
        embedder.embed_query("   ")

    with pytest.raises(ValueError, match="Estratégia desconhecida"):
        pipeline._process_and_index([Document("texto")], strategy="invalida")
