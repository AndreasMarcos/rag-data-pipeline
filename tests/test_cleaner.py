from models import Document
from processing.cleaner import clean_document, clean_documents


def _clean(text: str) -> str:
    return clean_document(Document(page_content=text)).page_content


def test_remove_caracteres_nao_imprimiveis():
    assert _clean("ol\x00á\x07 mun\x1fdo\x7f") == "olá mundo"
    # \n não é tratado como caractere de controle
    assert _clean("a\nb") == "a\nb"


def test_colapsa_espacos_tabs_e_quebras_de_linha_excessivas():
    assert _clean("a    b \t c\n\n\n\n\nd") == "a b c\n\nd"
    assert _clean("a\tb") == "a b"


def test_reduz_pontuacao_repetida():
    assert _clean("fim......") == "fim..."
    assert _clean("------") == "---"
    assert _clean("nome: ________") == "nome: ___"
    # repetições curtas não são alteradas
    assert _clean("a.. b-- c__") == "a.. b-- c__"


def test_clean_documents_faz_strip_preserva_metadados_e_descarta_vazios():
    docs = [
        Document(page_content="   conteúdo   ", metadata={"source": "x.pdf", "page": 2}),
        Document(page_content="   \n\n\t "),
        Document(page_content="\x00\x01"),
    ]
    result = clean_documents(docs)
    assert len(result) == 1
    assert result[0].page_content == "conteúdo"
    assert result[0].metadata == {"source": "x.pdf", "page": 2}
