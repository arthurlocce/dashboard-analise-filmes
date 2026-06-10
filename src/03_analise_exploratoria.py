from pathlib import Path
import json

import pandas as pd


ROOT = Path(_file_).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    movies = pd.read_csv(PROCESSED_DIR / "movie_summary.csv", low_memory=False)
    genres = pd.read_csv(PROCESSED_DIR / "genre_summary.csv")
    directors = pd.read_csv(PROCESSED_DIR / "director_summary.csv")
    with (PROCESSED_DIR / "dashboard_metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    return movies, genres, directors, metrics


def gerar_insights() -> str:
    movies, genres, directors, metrics = carregar_dados()

    financial = movies[movies["has_financials"].astype(str).str.lower().isin(["true", "1"])].copy()
    financial["budget_usd"] = pd.to_numeric(financial["budget_usd"], errors="coerce")
    financial["revenue_usd"] = pd.to_numeric(financial["revenue_usd"], errors="coerce")
    financial["roi"] = pd.to_numeric(financial["roi"], errors="coerce")
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce")
    movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce")
    movies["vote_count"] = pd.to_numeric(movies["vote_count"], errors="coerce")

    genero_mais_filmes = genres.sort_values("movie_count", ascending=False).iloc[0]
    genero_maior_receita = genres.sort_values("total_revenue", ascending=False).iloc[0]
    genero_melhor_nota = genres[genres["movie_count"] >= 300].sort_values("avg_weighted_score", ascending=False).iloc[0]
    filme_maior_receita = movies.sort_values("revenue_usd", ascending=False).iloc[0]
    filme_maior_roi = financial[financial["budget_usd"] >= 1_000_000].sort_values("roi", ascending=False).iloc[0]
    diretor_maior_receita = directors.sort_values("total_revenue", ascending=False).iloc[0]
    filmes_financeiros = int(metrics["movies_with_financials"])

    corr_budget_revenue = financial["budget_usd"].corr(financial["revenue_usd"])
    corr_pop_vote = movies[movies["vote_count"] >= 100]["popularity"].corr(movies[movies["vote_count"] >= 100]["vote_average"])

    linhas = [
        "# Insights principais - The Movies Dataset",
        "",
        "## Fonte e volume",
        "- Fonte: The Movies Dataset, Kaggle.",
        f"- Base com {metrics['total_movies']:,} filmes e {metrics['total_tmdb_votes']:,} avaliações TMDb somadas.",
        f"- Arquivos integrados: movies_metadata.csv e credits.csv.",
        f"- Também foram identificados {metrics['total_directors']:,} diretores.",
        "",
        "## Achados para apresentar",
        f"1. O gênero com mais filmes é *{genero_mais_filmes['genre']}*, com {int(genero_mais_filmes['movie_count']):,} títulos.",
        f"2. O gênero com maior receita total é *{genero_maior_receita['genre']}*, somando cerca de US$ {genero_maior_receita['total_revenue'] / 1_000_000_000:.1f} bilhões.",
        f"3. Entre gêneros com pelo menos 300 filmes, o melhor score médio ponderado é de *{genero_melhor_nota['genre']}*.",
        f"4. A correlação entre orçamento e receita é {corr_budget_revenue:.2f}; isso indica uma relação positiva, mas não perfeita, entre investimento e bilheteria.",
        f"5. A correlação entre popularidade e nota média é {corr_pop_vote:.2f}; popularidade não significa automaticamente melhor avaliação.",
        f"6. O filme de maior receita na base é *{filme_maior_receita['title']}*, com aproximadamente US$ {filme_maior_receita['revenue_usd'] / 1_000_000_000:.1f} bilhões.",
        f"7. Considerando orçamento mínimo de US$ 1 milhão, o maior retorno sobre orçamento foi de *{filme_maior_roi['title']}*.",
        f"8. O diretor com maior receita total é *{diretor_maior_receita['director']}*.",
        f"9. A base possui *{filmes_financeiros:,}* filmes com orçamento e receita informados para análise financeira.",
        "",
        "## Técnicas de aula aplicadas",
        "- Leitura de CSV com Pandas.",
        "- Integração com merge entre metadados e créditos.",
        "- Concatenação para adicionar variáveis codificadas de gênero.",
        "- Limpeza de IDs, datas, valores numéricos, orçamento, receita e campos ausentes.",
        "- Criação de novas variáveis: década, gênero principal, lucro, ROI, score confiável e faixas discretizadas.",
        "- Agregações por gênero, década e diretor.",
        "- Dashboard em Dash com filtros interativos, comparação de variáveis e múltiplos gráficos.",
    ]

    return "\n".join(linhas)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    texto = gerar_insights()
    caminho = DOCS_DIR / "insights_automaticos.md"
    caminho.write_text(texto, encoding="utf-8")
    print(f"Insights gerados em: {caminho}")


if _name_ == "_main_":
    main()