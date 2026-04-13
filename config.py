from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE = 512        # tamanho de cada chunk em tokens/caracteres
CHUNK_OVERLAP = 64      # sobreposição entre chunks consecutivos

# ── Embedding ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # modelo leve e eficiente para portfólio

# ── ChromaDB ───────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "rag_pipeline"

# ── Web Loader ─────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 10    # segundos
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGPipeline/1.0)"
}