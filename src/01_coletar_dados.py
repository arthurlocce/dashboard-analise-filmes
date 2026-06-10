from pathlib import Path
import shutil

import kagglehub


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_DIR = RAW_DIR / "the-movies-dataset"
KAGGLE_DATASET = "rounakbanik/the-movies-dataset"


def baixar_dataset() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    arquivos_esperados = [
        "movies_metadata.csv",
        "credits.csv",
    ]
    if all((DATASET_DIR / nome).exists() for nome in arquivos_esperados):
        print("Dataset Kaggle ja existe localmente. Pulando download.")
    else:
        print("Baixando The Movies Dataset do Kaggle...")
        origem = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        for nome in arquivos_esperados:
            arquivo = origem / nome
            if arquivo.exists():
                shutil.copy2(arquivo, DATASET_DIR / arquivo.name)

    print("Arquivos brutos disponiveis:")
    for arquivo in sorted(DATASET_DIR.glob("*")):
        print(f"- {arquivo.name} ({arquivo.stat().st_size:,} bytes)")


if __name__ == "__main__":
    baixar_dataset()
