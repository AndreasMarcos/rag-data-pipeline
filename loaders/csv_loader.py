import pandas as pd
from pathlib import Path
from models import Document


def load_csv(
    file_path: str | Path,
    columns: list[str] | None = None,
    row_separator: str = " | ",
) -> list[Document]:
    """
    Converte cada linha de um CSV em um Document.

    Args:
        file_path:     caminho para o arquivo .csv
        columns:       colunas a incluir (None = todas)
        row_separator: separador entre os pares coluna:valor

    Returns:
        Lista de Documents, um por linha.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se o arquivo não for .csv ou estiver vazio.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Esperado arquivo .csv, recebido: {path.suffix}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Arquivo CSV vazio: {path}")

    # filtra colunas se especificado
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas não encontradas: {missing}")
        df = df[columns]

    # remove linhas completamente vazias
    df = df.dropna(how="all")

    documents = []

    for idx, row in df.iterrows():
        text = row_separator.join(
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val)
        )

        documents.append(Document(
            page_content=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "row": int(idx),
                "total_rows": len(df),
            }
        ))

    return documents