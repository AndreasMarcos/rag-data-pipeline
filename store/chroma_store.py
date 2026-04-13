import uuid
import chromadb
from chromadb.config import Settings
from models import Document
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


_client: chromadb.Client | None = None


def _get_client() -> chromadb.Client:
    """Retorna o cliente ChromaDB, inicializando uma única vez (singleton)."""
    global _client
    if _client is None:
        print(f"[chroma_store] Conectando ao ChromaDB em: {CHROMA_PERSIST_DIR}")
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
        )
    return _client


def _get_collection() -> chromadb.Collection:
    """Retorna a collection, criando se não existir."""
    client = _get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # similaridade por cosseno
    )


def add_documents(
    docs_and_embeddings: list[tuple[Document, list[float]]]
) -> None:
    """
    Indexa documents e seus embeddings no ChromaDB.

    Args:
        docs_and_embeddings: saída do embed_documents()
    """
    if not docs_and_embeddings:
        return

    collection = _get_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for doc, embedding in docs_and_embeddings:
        ids.append(str(uuid.uuid4()))
        embeddings.append(embedding)
        documents.append(doc.page_content)
        metadatas.append(doc.metadata)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[chroma_store] {len(ids)} chunks indexados com sucesso.")


def query(
    query_embedding: list[float],
    n_results: int = 5,
) -> list[Document]:
    """
    Busca os chunks mais relevantes para uma query.

    Args:
        query_embedding: embedding da query gerado por embed_query()
        n_results:       número de resultados a retornar

    Returns:
        Lista de Documents ordenados por relevância.
    """
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        documents.append(Document(
            page_content=text,
            metadata={
                **metadata,
                "score": round(1 - distance, 4),  # converte distância em similaridade
            }
        ))

    return documents


def delete_collection() -> None:
    """Remove toda a collection — útil para resetar o índice durante testes."""
    client = _get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"[chroma_store] Collection '{CHROMA_COLLECTION_NAME}' removida.")
    except Exception:
        print(f"[chroma_store] Collection '{CHROMA_COLLECTION_NAME}' ainda não existe, pulando.")


def collection_info() -> dict:
    """Retorna informações básicas sobre a collection atual."""
    collection = _get_collection()
    return {
        "name": collection.name,
        "count": collection.count(),
    }