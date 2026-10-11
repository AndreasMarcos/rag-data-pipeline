# Exemplo de uso do pipeline completo.
# Ingere dados de três fontes (PDF, CSV, Web) e executa buscas de teste.
#
# Execução:
#   cd rag_pipeline
#   python examples/run_pipeline.py

import logging

from pipeline import ingest_pdf, ingest_csv, ingest_web, search, status
from store import delete_collection


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # limpa o índice antes de começar
    print("=== Resetando índice ===")
    delete_collection()

    # ── ingestão ──────────────────────────────────────────────────────
    print("\n=== Ingestão ===")

    chunks_pdf = ingest_pdf("data/attention.pdf")
    chunks_csv = ingest_csv("data/produtos.csv")
    chunks_web = ingest_web("https://pt.wikipedia.org/wiki/Inteligência_artificial")

    print(f"\nTotal indexado:")
    print(f"  PDF : {chunks_pdf} chunks")
    print(f"  CSV : {chunks_csv} chunks")
    print(f"  Web : {chunks_web} chunks")

    status()

    # ── buscas ────────────────────────────────────────────────────────
    print("\n=== Buscas ===")

    queries = [
        "how does the attention mechanism work in transformers?",
        "qual o preço do notebook?",
        "o que é inteligência artificial?",
    ]

    for q in queries:
        print(f"\n{'─'*60}")
        results = search(q, n_results=3)
        print(f"\nTop resultado:")
        print(f"  {results[0].page_content[:300]}")
        print(f"  metadata: {results[0].metadata}")


if __name__ == "__main__":
    main()