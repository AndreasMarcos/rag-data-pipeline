"""
Configuração compartilhada dos testes.

- Coloca a raiz do repositório no sys.path para importar os módulos do projeto.
- Substitui `sentence_transformers`, `torch` e `chromadb` por stubs leves
  (mesmo que estejam instalados), para que os testes rodem offline, rápidos,
  sem baixar modelos e sem criar um banco Chroma persistente.
"""
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _install_stub(name: str, module: types.ModuleType) -> None:
    """Registra o stub em sys.modules antes de qualquer import do projeto."""
    sys.modules[name] = module


# ── stub: sentence_transformers ────────────────────────────────────────
_st = types.ModuleType("sentence_transformers")


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("SentenceTransformer não deve ser instanciado nos testes.")


_st.SentenceTransformer = _FakeSentenceTransformer
_install_stub("sentence_transformers", _st)
_install_stub("torch", types.ModuleType("torch"))


# ── stub: chromadb ─────────────────────────────────────────────────────
_chroma = types.ModuleType("chromadb")
_chroma_config = types.ModuleType("chromadb.config")


class _FakeClient:
    pass


class _FakeCollection:
    pass


def _fake_persistent_client(*args, **kwargs):
    raise RuntimeError("PersistentClient não deve ser criado nos testes.")


_chroma.Client = _FakeClient
_chroma.Collection = _FakeCollection
_chroma.PersistentClient = _fake_persistent_client
_chroma_config.Settings = type("Settings", (), {})
_chroma.config = _chroma_config

_install_stub("chromadb", _chroma)
_install_stub("chromadb.config", _chroma_config)
