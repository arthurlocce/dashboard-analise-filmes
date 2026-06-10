from __future__ import annotations

from pathlib import Path
import ast
import json
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "the-movies-dataset"
PROCESSED_DIR = ROOT / "data" / "processed"


def safe_literal_eval(value):
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def nomes_lista(value, limite: int | None = None) -> list[str]:
    itens = safe_literal_eval(value)
    nomes = [item.get("name") for item in itens if isinstance(item, dict) and item.get("name")]
    return nomes[:limite] if limite else nomes


def primeiro_nome(value) -> str:
    nomes = nomes_lista(value, limite=1)
    return nomes[0] if nomes else "Não informado"


def diretor_principal(crew_value) -> str:
    for item in safe_literal_eval(crew_value):
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name") or "Não informado"
    return "Não informado"


def texto_lista(nomes: list[str]) -> str:
    return ", ".join(nomes) if nomes else "Não informado"


def limpar_moeda(serie: pd.Series) -> pd.Series:
    valores = pd.to_numeric(serie, errors="coerce")
    return valores.where(valores > 0)


def encurtar_periodo(serie: pd.Series) -> pd.Series:
    return pd.to_datetime(serie, errors="coerce").dt.year.astype("Int64")


def ler_metadados() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(RAW_DIR / "movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(RAW_DIR / "credits.csv")
    return movies, credits


def preparar_filmes(
    movies: pd.DataFrame,
    credits: pd.DataFrame,
) -> pd.DataFrame:
    movies = movies.copy()
    movies["tmdb_id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["tmdb_id"]).copy()
    movies["tmdb_id"] = movies["tmdb_id"].astype(int)
    movies = movies.drop_duplicates(subset=["tmdb_id"])

    movies["title"] = movies["title"].fillna(movies["original_title"]).fillna("Sem título")
    movies["budget_usd"] = limpar_moeda(movies["budget"])
    movies["revenue_usd"] = limpar_moeda(movies["revenue"])
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce")
    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce")
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce").fillna(0)
    movies["runtime"] = pd.to_numeric(movies["runtime"], errors="coerce")
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["release_year"] = movies["release_date"].dt.year.astype("Int64")
    movies["release_decade"] = (movies["release_year"] // 10 * 10).astype("Int64")
    movies["original_language"] = movies["original_language"].fillna("Não informado")

    movies["genre_list"] = movies["genres"].apply(nomes_lista)
    movies["genres_text"] = movies["genre_list"].apply(texto_lista)
    movies["primary_genre"] = movies["genre_list"].apply(lambda nomes: nomes[0] if nomes else "Não informado")
    movies["genre_count"] = movies["genre_list"].apply(len)
    movies["country"] = movies["production_countries"].apply(primeiro_nome)
    movies["company"] = movies["production_companies"].apply(primeiro_nome)
    movies["spoken_language"] = movies["spoken_languages"].apply(primeiro_nome)

    movies["profit_usd"] = movies["revenue_usd"] - movies["budget_usd"]
    movies["roi"] = movies["revenue_usd"] / movies["budget_usd"]
    movies.loc[~np.isfinite(movies["roi"]), "roi"] = np.nan
    movies["has_financials"] = movies["budget_usd"].notna() & movies["revenue_usd"].notna()

    credits = credits.copy()
    credits["tmdb_id"] = pd.to_numeric(credits["id"], errors="coerce")
    credits = credits.dropna(subset=["tmdb_id"]).copy()
    credits["tmdb_id"] = credits["tmdb_id"].astype(int)
    credits["director"] = credits["crew"].apply(diretor_principal)
    credits["cast_list"] = credits["cast"].apply(lambda value: nomes_lista(value, limite=5))
    credits["main_cast"] = credits["cast_list"].apply(texto_lista)
    credits = credits[["tmdb_id", "director", "cast_list", "main_cast"]]
    credits = credits.drop_duplicates(subset=["tmdb_id"])

    resumo = movies.merge(credits, on="tmdb_id", how="left")

    resumo["director"] = resumo["director"].fillna("Não informado")
    resumo["main_cast"] = resumo["main_cast"].fillna("Não informado")

    media_global = resumo.loc[resumo["vote_count"] > 0, "vote_average"].mean()
    minimo_votos = max(100, int(resumo["vote_count"].quantile(0.75)))
    votos = resumo["vote_count"]
    nota = resumo["vote_average"].fillna(media_global)
    resumo["weighted_score"] = (votos / (votos + minimo_votos)) * nota + (
        minimo_votos / (votos + minimo_votos)
    ) * media_global

    resumo["budget_log"] = np.log1p(resumo["budget_usd"].fillna(0))
    resumo["revenue_log"] = np.log1p(resumo["revenue_usd"].fillna(0))
    resumo["popularity_log"] = np.log1p(resumo["popularity"].clip(lower=0).fillna(0))
    resumo["faixa_popularidade"] = pd.qcut(
        resumo["popularity"].fillna(0).rank(method="first"),
        q=4,
        labels=["baixa", "média", "alta", "muito alta"],
    ).astype(str)
    resumo["faixa_orcamento"] = pd.cut(
        resumo["budget_usd"].fillna(0),
        bins=[-1, 0, 1_000_000, 10_000_000, 50_000_000, math.inf],
        labels=["sem orçamento", "baixo", "médio", "alto", "muito alto"],
    ).astype(str)

    generos = {genero: idx for idx, genero in enumerate(sorted(resumo["primary_genre"].astype(str).unique()))}
    resumo["primary_genre_encoded"] = resumo["primary_genre"].astype(str).map(generos).fillna(0)
    genre_dummies = resumo["genres_text"].str.get_dummies(sep=", ").add_prefix("genre_")
    resumo = pd.concat([resumo, genre_dummies], axis=1)

    return resumo


def gerar_agregacoes(movie_summary: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    exploded_genres = (
        movie_summary[[
            "tmdb_id",
            "title",
            "release_year",
            "release_decade",
            "vote_average",
            "vote_count",
            "weighted_score",
            "popularity",
            "budget_usd",
            "revenue_usd",
            "profit_usd",
            "roi",
            "genre_list",
        ]]
        .explode("genre_list")
        .rename(columns={"genre_list": "genre"})
    )
    exploded_genres["genre"] = exploded_genres["genre"].fillna("Não informado")

    genre_summary = (
        exploded_genres.groupby("genre")
        .agg(
            movie_count=("tmdb_id", "nunique"),
            avg_vote=("vote_average", "mean"),
            avg_weighted_score=("weighted_score", "mean"),
            total_vote_count=("vote_count", "sum"),
            avg_popularity=("popularity", "mean"),
            total_revenue=("revenue_usd", "sum"),
            total_budget=("budget_usd", "sum"),
            movies_with_financials=("revenue_usd", "count"),
        )
        .reset_index()
        .sort_values("movie_count", ascending=False)
    )
    genre_summary["profit"] = genre_summary["total_revenue"] - genre_summary["total_budget"]

    genre_year_summary = (
        exploded_genres.dropna(subset=["release_year"])
        .groupby(["genre", "release_decade"])
        .agg(
            movie_count=("tmdb_id", "nunique"),
            avg_vote=("vote_average", "mean"),
            avg_popularity=("popularity", "mean"),
            total_revenue=("revenue_usd", "sum"),
        )
        .reset_index()
    )

    director_summary = (
        movie_summary[movie_summary["director"] != "Não informado"]
        .groupby("director")
        .agg(
            movie_count=("tmdb_id", "nunique"),
            avg_vote=("vote_average", "mean"),
            avg_weighted_score=("weighted_score", "mean"),
            total_revenue=("revenue_usd", "sum"),
            total_budget=("budget_usd", "sum"),
            avg_popularity=("popularity", "mean"),
        )
        .reset_index()
    )
    director_summary["profit"] = director_summary["total_revenue"] - director_summary["total_budget"]
    director_summary = director_summary.sort_values(["total_revenue", "movie_count"], ascending=False)

    metrics = {
        "fonte": "The Movies Dataset - Kaggle",
        "kaggle_dataset": "rounakbanik/the-movies-dataset",
        "total_movies": int(movie_summary.shape[0]),
        "total_tmdb_votes": int(movie_summary["vote_count"].fillna(0).sum()),
        "total_genres": int(genre_summary.shape[0]),
        "total_directors": int(director_summary.shape[0]),
        "movies_with_budget": int(movie_summary["budget_usd"].notna().sum()),
        "movies_with_revenue": int(movie_summary["revenue_usd"].notna().sum()),
        "movies_with_financials": int(movie_summary["has_financials"].sum()),
        "release_year_min": int(movie_summary["release_year"].dropna().min()),
        "release_year_max": int(movie_summary["release_year"].dropna().max()),
    }

    return {
        "movie_summary": movie_summary,
        "genre_summary": genre_summary,
        "genre_year_summary": genre_year_summary,
        "director_summary": director_summary,
        "metrics": metrics,
    }


def salvar_resultados(resultados: dict[str, pd.DataFrame | dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for arquivo in PROCESSED_DIR.glob("*"):
        if arquivo.suffix.lower() in [".csv", ".json"]:
            arquivo.unlink()

    for nome, dados in resultados.items():
        if isinstance(dados, pd.DataFrame):
            dados.to_csv(PROCESSED_DIR / f"{nome}.csv", index=False)
    with (PROCESSED_DIR / "dashboard_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(resultados["metrics"], file, ensure_ascii=False, indent=2)


def main() -> None:
    movies, credits = ler_metadados()
    movie_summary = preparar_filmes(movies, credits)
    resultados = gerar_agregacoes(movie_summary)
    salvar_resultados(resultados)

    print("Pipeline Kaggle concluído.")
    print(f"Filmes: {resultados['metrics']['total_movies']:,}")
    print(f"Avaliações TMDb: {resultados['metrics']['total_tmdb_votes']:,}")
    print(f"Diretores: {resultados['metrics']['total_directors']:,}")
    print(f"Arquivos processados em: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
