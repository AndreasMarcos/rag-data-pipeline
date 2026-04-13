# RAG Data Pipeline

Pipeline modular para preparação e indexação de dados em sistemas de
Retrieval-Augmented Generation (RAG).

## Visão Geral

Este projeto implementa um fluxo completo de preparação de dados para RAG,
cobrindo todas as etapas desde a ingestão de fontes brutas até o armazenamento
vetorial — com foco em qualidade de código, modularidade e aplicabilidade real.

```
Fontes → Extração → Limpeza → Chunking → Embedding → Vector Store
```

## Arquitetura

```
rag_pipeline/
├── loaders/          # Extração por tipo de fonte
│   ├── pdf_loader.py
│   ├── web_loader.py
│   └── csv_loader.py
├── processing/       # Limpeza e chunking
│   ├── cleaner.py
│   └── chunker.py
├── embeddings/       # Interface com modelo de embedding
│   └── embedder.py
├── store/            # Integração com ChromaDB
│   └── chroma_store.py
├── examples/         # Exemplos de uso
│   └── test_pipeline.py
├── pipeline.py       # Camada de orquestração
├── models.py         # Estruturas de dados compartilhadas
└── config.py         # Configurações centralizadas
```

## Funcionalidades

- **Ingestão multi-fonte** — PDFs, páginas web e CSVs
- **Design modular** — cada etapa é independente e facilmente extensível
- **Chunking configurável** — estratégias fixed-size e recursiva com controle de overlap
- **Embedding agnóstico** — troca o modelo de embedding sem alterar a lógica do pipeline
- **Integração com ChromaDB** — vector store local com similaridade por cosseno, sem infraestrutura adicional

## Tecnologias

- `Python 3.10+`
- `sentence-transformers` — geração de embeddings (`all-MiniLM-L6-v2`)
- `ChromaDB` — vector store local
- `PyMuPDF` — extração de texto de PDFs
- `BeautifulSoup4` — parsing de HTML
- `pandas` — ingestão de CSVs

## Como Executar

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/rag-data-pipeline.git
cd rag-data-pipeline
```

**2. Crie e ative o ambiente virtual**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Execute o exemplo**
```bash
python examples/test_pipeline.py
```

## Uso

```python
from pipeline import ingest_pdf, ingest_csv, ingest_web, search

# ingere dados de múltiplas fontes
ingest_pdf("data/documento.pdf")
ingest_csv("data/produtos.csv")
ingest_web("https://exemplo.com/artigo")

# busca em todo o conteúdo indexado
results = search("como funciona o mecanismo de atenção?", n_results=5)

for doc in results:
    print(doc.metadata["score"], doc.page_content[:200])
```

## Decisões de Design

**Por que um Document por página do PDF?**
Preservar a granularidade por página nos metadados permite que o sistema RAG
cite a localização exata de cada trecho recuperado.

**Por que chunking recursivo como padrão?**
O split recursivo respeita a estrutura natural do texto — quebrando por
parágrafos antes de linhas, linhas antes de frases — produzindo chunks
semanticamente mais coerentes do que o split de tamanho fixo.

**Por que similaridade por cosseno?**
Para embeddings de texto, a similaridade por cosseno mede o ângulo entre
vetores em vez de sua magnitude, tornando-a mais robusta a diferenças no
tamanho dos textos.

## Limitações Conhecidas

**Vocabulary mismatch** — queries em linguagem natural podem ter scores mais
baixos contra documentos técnicos devido a diferenças de vocabulário. Técnicas
como HyDE (Hypothetical Document Embeddings) ou query expansion podem mitigar isso.

**PDFs escaneados** — o loader de PDF extrai texto apenas da camada de texto.
Documentos escaneados precisam de uma etapa de OCR antes da ingestão.

**Alinhamento de idioma** — o modelo de embedding padrão performa melhor quando
queries e documentos estão no mesmo idioma.