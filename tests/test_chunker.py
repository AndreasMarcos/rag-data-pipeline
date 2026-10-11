import pytest

from models import Document
from processing.chunker import ChunkStrategy, chunk_document


def test_fixed_split_respeita_tamanho_e_overlap():
    doc = Document(page_content="abcdefghij")
    chunks = chunk_document(doc, ChunkStrategy.FIXED, chunk_size=4, chunk_overlap=1)
    assert [c.page_content for c in chunks] == ["abcd", "defg", "ghij", "j"]

    # sem overlap, os chunks são contíguos
    chunks = chunk_document(doc, ChunkStrategy.FIXED, chunk_size=4, chunk_overlap=0)
    assert [c.page_content for c in chunks] == ["abcd", "efgh", "ij"]


def test_recursive_split_prioriza_paragrafos_e_respeita_chunk_size():
    text = "Primeiro parágrafo.\n\nSegundo parágrafo.\n\nTerceiro parágrafo."
    chunks = chunk_document(
        Document(page_content=text), ChunkStrategy.RECURSIVE, chunk_size=25, chunk_overlap=0
    )
    assert [c.page_content for c in chunks] == [
        "Primeiro parágrafo.",
        "Segundo parágrafo.",
        "Terceiro parágrafo.",
    ]

    # texto menor que chunk_size gera um único chunk
    short = chunk_document(Document(page_content="texto curto"), chunk_size=100, chunk_overlap=10)
    assert [c.page_content for c in short] == ["texto curto"]

    text = " ".join(f"palavra{i}" for i in range(200))
    chunks = chunk_document(
        Document(page_content=text), ChunkStrategy.RECURSIVE, chunk_size=50, chunk_overlap=0
    )
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 50 for c in chunks)
    # nenhuma palavra é perdida
    assert " ".join(c.page_content for c in chunks).split() == text.split()


def test_recursive_split_aplica_overlap_com_final_do_chunk_anterior():
    text = " ".join(f"palavra{i}" for i in range(50))
    overlap = 8
    chunks = chunk_document(
        Document(page_content=text), ChunkStrategy.RECURSIVE, chunk_size=40, chunk_overlap=overlap
    )
    without_overlap = chunk_document(
        Document(page_content=text), ChunkStrategy.RECURSIVE, chunk_size=40, chunk_overlap=0
    )
    assert len(chunks) == len(without_overlap) > 1
    assert chunks[0].page_content == without_overlap[0].page_content
    for i in range(1, len(chunks)):
        tail = without_overlap[i - 1].page_content[-overlap:]
        assert chunks[i].page_content == tail + " " + without_overlap[i].page_content


def test_metadados_herdados_e_indice_do_chunk():
    doc = Document(page_content="abcdefgh", metadata={"source": "a.txt", "page": 3})
    chunks = chunk_document(doc, ChunkStrategy.FIXED, chunk_size=4, chunk_overlap=0)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata == {
            "source": "a.txt",
            "page": 3,
            "chunk_index": i,
            "total_chunks": 2,
        }
    # o metadado original não é modificado
    assert doc.metadata == {"source": "a.txt", "page": 3}


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkStrategy.FIXED,
        ChunkStrategy.RECURSIVE,
    ],
)
def test_texto_vazio_nao_gera_chunks(strategy):
    assert chunk_document(Document(page_content=""), strategy) == []

