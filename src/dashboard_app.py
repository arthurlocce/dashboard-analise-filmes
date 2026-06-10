from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
}
SCATTER_RENDER_MODE = "webgl"
MAX_SCATTER_POINTS = 1800
MAX_OVERVIEW_SCATTER_POINTS = 2500
PALETTE = [
    "#245f68",
    "#9f433b",
    "#b58a24",
    "#576879",
    "#80635e",
    "#4f7a73",
    "#8b4156",
    "#68763d",
    "#96633d",
    "#3f5870",
    "#aa6f34",
    "#6b6f8a",
]
CONTINUOUS_SCALE = ["#245f68", "#d8c36a", "#9f433b"]


def carregar_csv(nome: str) -> pd.DataFrame:
    caminho = PROCESSED_DIR / nome
    if not caminho.exists():
        raise FileNotFoundError("Execute primeiro: python src/02_pipeline_preprocessamento.py")
    return pd.read_csv(caminho, low_memory=False)


def carregar_metricas() -> dict:
    with (PROCESSED_DIR / "dashboard_metrics.json").open(encoding="utf-8") as arquivo:
        return json.load(arquivo)


metrics = carregar_metricas()
movies = carregar_csv("movie_summary.csv")
genre_summary = carregar_csv("genre_summary.csv")
genre_year_summary = carregar_csv("genre_year_summary.csv")
director_summary = carregar_csv("director_summary.csv")


NUMERIC_COLUMNS = [
    "budget_usd",
    "revenue_usd",
    "roi",
    "popularity",
    "vote_average",
    "vote_count",
    "runtime",
    "release_year",
    "release_decade",
]
for column in NUMERIC_COLUMNS:
    if column in movies:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")

for frame, columns in [
    (genre_summary, ["movie_count", "avg_vote", "avg_weighted_score", "total_revenue", "total_budget", "profit"]),
    (genre_year_summary, ["movie_count", "avg_vote", "avg_popularity", "total_revenue", "release_decade"]),
    (director_summary, ["movie_count", "avg_vote", "avg_weighted_score", "total_revenue", "profit", "avg_popularity"]),
]:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

movies["genres_text"] = movies["genres_text"].fillna("Não informado")
movies["primary_genre"] = movies["primary_genre"].fillna("Não informado")
movies["original_language"] = movies["original_language"].fillna("Não informado")
movies["director"] = movies["director"].fillna("Não informado")
movies["main_cast"] = movies["main_cast"].fillna("Não informado")
movies["title"] = movies["title"].fillna("Sem título")
movies["has_financials"] = movies["has_financials"].astype(str).str.lower().isin(["true", "1"])

GENRES = sorted(genre for genre in genre_summary["genre"].dropna().unique() if genre != "Não informado")
GENRE_COLUMNS = {genre: f"genre_{genre}" for genre in GENRES if f"genre_{genre}" in movies.columns}
YEAR_MIN = int(metrics["release_year_min"])
YEAR_MAX = int(metrics["release_year_max"])

METRIC_OPTIONS = {
    "revenue_usd": "Receita",
    "vote_average": "Nota média",
    "roi": "Retorno sobre orçamento",
}
FINANCIAL_METRICS = {"revenue_usd", "roi"}


def encurtar(texto: str, limite: int = 44) -> str:
    texto = str(texto)
    return texto if len(texto) <= limite else texto[: limite - 1] + "..."


def dinheiro(valor) -> str:
    if pd.isna(valor):
        return "-"
    valor = float(valor)
    if abs(valor) >= 1_000_000_000:
        return f"US$ {valor / 1_000_000_000:.1f} bi".replace(".", ",")
    if abs(valor) >= 1_000_000:
        return f"US$ {valor / 1_000_000:.1f} mi".replace(".", ",")
    if abs(valor) >= 1_000:
        return f"US$ {valor / 1_000:.1f} mil".replace(".", ",")
    return f"US$ {valor:.0f}"


def adicionar_colunas_financeiras_mi(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado["budget_mi"] = resultado["budget_usd"] / 1_000_000
    resultado["revenue_mi"] = resultado["revenue_usd"] / 1_000_000
    return resultado


def numero_pt_br(valor: float, casas: int = 1) -> str:
    texto = f"{valor:.{casas}f}".replace(".", ",")
    return texto.rstrip("0").rstrip(",") if casas else texto


def dinheiro_curto(valor) -> str:
    if pd.isna(valor):
        return "-"
    valor = float(valor)
    absoluto = abs(valor)
    if absoluto >= 1_000_000_000:
        return f"US$ {numero_pt_br(valor / 1_000_000_000)} bi"
    if absoluto >= 1_000_000:
        return f"US$ {numero_pt_br(valor / 1_000_000)} mi"
    if absoluto >= 1_000:
        return f"US$ {numero_pt_br(valor / 1_000)} mil"
    return f"US$ {valor:.0f}"


def decimal_curto(valor, casas: int = 2) -> str:
    if pd.isna(valor):
        return "-"
    return numero_pt_br(float(valor), casas)


def contador_curto(valor) -> str:
    if pd.isna(valor):
        return "-"
    return f"{int(round(float(valor))):,}".replace(",", ".")


def retorno_curto(valor) -> str:
    return f"{decimal_curto(valor)}x"


def formatar_valor_metrica(valor, metrica: str) -> str:
    if metrica in ["budget_usd", "revenue_usd", "profit_usd", "total_revenue", "total_budget", "profit"]:
        return dinheiro_curto(valor)
    if metrica == "roi":
        return retorno_curto(valor)
    if metrica in ["vote_count", "movie_count"]:
        return contador_curto(valor)
    return decimal_curto(valor)


def adicionar_formatos_hover(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    if "budget_usd" in resultado.columns:
        resultado["budget_fmt"] = resultado["budget_usd"].apply(dinheiro_curto)
    if "revenue_usd" in resultado.columns:
        resultado["revenue_fmt"] = resultado["revenue_usd"].apply(dinheiro_curto)
    if "total_revenue" in resultado.columns:
        resultado["total_revenue_fmt"] = resultado["total_revenue"].apply(dinheiro_curto)
    if "roi" in resultado.columns:
        resultado["roi_fmt"] = resultado["roi"].apply(retorno_curto)
    if "popularity" in resultado.columns:
        resultado["popularity_fmt"] = resultado["popularity"].apply(decimal_curto)
    if "avg_popularity" in resultado.columns:
        resultado["avg_popularity_fmt"] = resultado["avg_popularity"].apply(decimal_curto)
    if "vote_average" in resultado.columns:
        resultado["vote_average_fmt"] = resultado["vote_average"].apply(decimal_curto)
    if "avg_vote" in resultado.columns:
        resultado["avg_vote_fmt"] = resultado["avg_vote"].apply(decimal_curto)
    if "vote_count" in resultado.columns:
        resultado["vote_count_fmt"] = resultado["vote_count"].apply(contador_curto)
    if "movie_count" in resultado.columns:
        resultado["movie_count_fmt"] = resultado["movie_count"].apply(contador_curto)
    return resultado


def passo_bonito(maximo: float, divisoes: int = 4) -> float:
    bruto = maximo / divisoes
    if bruto <= 0:
        return 1
    base = 10 ** np.floor(np.log10(bruto))
    for multiplicador in [1, 2, 2.5, 5, 10]:
        passo = multiplicador * base
        if bruto <= passo:
            return passo
    return 10 * base


def formatar_eixo_milhoes(fig: go.Figure, eixo: str = "x") -> go.Figure:
    valores = []
    for trace in fig.data:
        dados = getattr(trace, eixo, None)
        if dados is not None:
            for valor in dados:
                if valor is not None and not pd.isna(valor):
                    valores.append(float(valor))
    maximo = max(valores) if valores else 0
    if maximo <= 0:
        return fig

    passo = passo_bonito(maximo)
    limite = np.ceil(maximo / passo) * passo
    ticks = np.arange(0, limite + passo * 0.1, passo)
    textos = [dinheiro_curto(valor * 1_000_000) for valor in ticks]
    if eixo == "x":
        fig.update_xaxes(tickvals=ticks, ticktext=textos, tickangle=0, rangemode="tozero")
    else:
        fig.update_yaxes(tickvals=ticks, ticktext=textos, tickangle=0, rangemode="tozero")
    return fig


def formatar_eixo_dinheiro(fig: go.Figure, eixo: str = "y") -> go.Figure:
    valores = []
    for trace in fig.data:
        dados = getattr(trace, eixo, None)
        if dados is not None:
            for valor in dados:
                if valor is not None and not pd.isna(valor):
                    valores.append(float(valor))
    maximo = max(valores) if valores else 0

    if maximo >= 1_000_000_000:
        escala = 1_000_000_000
        sufixo = " bi"
    elif maximo >= 1_000_000:
        escala = 1_000_000
        sufixo = " mi"
    elif maximo >= 1_000:
        escala = 1_000
        sufixo = " mil"
    else:
        escala = 1
        sufixo = ""

    ticks = np.linspace(0, maximo, 5) if maximo > 0 else [0]
    maximo_escalado = maximo / escala if escala else maximo
    casas = 1 if 0 < maximo_escalado < 10 else 0
    textos = []
    for valor in ticks:
        escalado = valor / escala
        if valor == 0:
            textos.append(f"US$ 0{sufixo}")
        else:
            textos.append(f"US$ {escalado:.{casas}f}{sufixo}".replace(".", ","))
    if eixo == "x":
        fig.update_xaxes(tickvals=ticks, ticktext=textos)
    else:
        fig.update_yaxes(tickvals=ticks, ticktext=textos)
    return fig


def formatar_eixo_log_dinheiro(fig: go.Figure, eixo: str = "x") -> go.Figure:
    valores = []
    for trace in fig.data:
        dados = getattr(trace, eixo, None)
        if dados is not None:
            for valor in dados:
                if valor is not None and not pd.isna(valor) and float(valor) > 0:
                    valores.append(float(valor))
    if not valores:
        return fig

    ticks_padrao = [
        1_000_000,
        5_000_000,
        10_000_000,
        50_000_000,
        100_000_000,
        500_000_000,
        1_000_000_000,
        2_000_000_000,
    ]
    minimo, maximo = min(valores), max(valores)
    ticks = [valor for valor in ticks_padrao if minimo <= valor <= maximo]
    if not ticks:
        ticks = [minimo, maximo]
    textos = [dinheiro(valor) for valor in ticks]
    if eixo == "x":
        fig.update_xaxes(tickvals=ticks, ticktext=textos)
    else:
        fig.update_yaxes(tickvals=ticks, ticktext=textos)
    return fig


def inteiro(valor) -> str:
    return f"{int(valor):,}".replace(",", ".")


def card_indicador(titulo: str, valor: str, detalhe: str = "") -> html.Div:
    return html.Div(
        className="metric-card",
        children=[
            html.Span(titulo, className="metric-label"),
            html.Strong(valor, className="metric-value"),
            html.Span(detalhe, className="metric-detail"),
        ],
    )


def grafico(figure=None, graph_id: str | None = None) -> html.Div:
    props = {
        "figure": figure,
        "config": GRAPH_CONFIG,
        "className": "chart-graph",
        "style": {"height": "420px", "width": "100%"},
    }
    if graph_id:
        props["id"] = graph_id
    return html.Div(className="chart-card", children=dcc.Graph(**props))


def aplicar_estilo(fig: go.Figure, height: int = 420, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=height,
        margin=dict(l=62, r=24, t=48, b=48),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        title=dict(font=dict(size=15, color="#1d252d"), x=0.02, xanchor="left"),
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#263238"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#c7ced6", font_size=12),
        dragmode="zoom",
        showlegend=legend,
    )
    fig.update_xaxes(
        automargin=True,
        title_standoff=12,
        title_font=dict(size=12, color="#333b44"),
        tickfont=dict(size=11, color="#59636e"),
        gridcolor="#edf0f2",
        zerolinecolor="#d7dce1",
        linecolor="#d7dce1",
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=16,
        title_font=dict(size=12, color="#333b44"),
        tickfont=dict(size=11, color="#59636e"),
        gridcolor="#edf0f2",
        zerolinecolor="#d7dce1",
        linecolor="#d7dce1",
    )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="center",
                x=0.5,
                font=dict(size=11),
            )
        )
    return fig


def figura_vazia(titulo: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="Sem dados para os filtros selecionados", x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=titulo)
    return aplicar_estilo(fig)


def normalizar_intervalo_anos(inicio, fim) -> list[int]:
    ano_inicio = YEAR_MIN if inicio in [None, ""] else int(inicio)
    ano_fim = YEAR_MAX if fim in [None, ""] else int(fim)
    ano_inicio = max(YEAR_MIN, min(YEAR_MAX, ano_inicio))
    ano_fim = max(YEAR_MIN, min(YEAR_MAX, ano_fim))
    if ano_inicio > ano_fim:
        ano_inicio, ano_fim = ano_fim, ano_inicio
    return [ano_inicio, ano_fim]


def normalizar_minimo_avaliacoes(valor) -> int:
    try:
        minimo = int(float(str(valor or "0").strip().replace(",", ".")))
    except (TypeError, ValueError):
        minimo = 0
    return max(0, minimo)


def limitar_pontos_scatter(
    df: pd.DataFrame,
    maximo: int = MAX_SCATTER_POINTS,
    ordenar_por: list[str] | None = None,
) -> pd.DataFrame:
    if df.shape[0] <= maximo:
        return df.copy()

    colunas = [coluna for coluna in (ordenar_por or []) if coluna in df.columns]
    if colunas:
        return df.sort_values(colunas, ascending=False).head(maximo).copy()
    return df.head(maximo).copy()


def filtrar_filmes(genres, year_start, year_end, min_votes, financial_only) -> pd.DataFrame:
    intervalo = normalizar_intervalo_anos(year_start, year_end)
    df = movies[movies["release_year"].between(intervalo[0], intervalo[1], inclusive="both")].copy()

    if genres:
        colunas_genero = [GENRE_COLUMNS[genre] for genre in genres if genre in GENRE_COLUMNS]
        if colunas_genero:
            df = df[df[colunas_genero].astype(bool).any(axis=1)]
        else:
            df = df.iloc[0:0]
    min_votes = normalizar_minimo_avaliacoes(min_votes)
    if min_votes > 0:
        df = df[df["vote_count"] >= min_votes]
    if financial_only:
        df = df[df["has_financials"]]
    return df


def grafico_receita_decada() -> go.Figure:
    df = (
        movies.dropna(subset=["release_decade"])
        .groupby("release_decade", as_index=False)
        .agg(total_revenue=("revenue_usd", "sum"), movie_count=("tmdb_id", "nunique"))
    )
    df = adicionar_formatos_hover(df)
    fig = px.bar(
        df,
        x="release_decade",
        y="total_revenue",
        custom_data=["total_revenue_fmt", "movie_count_fmt"],
        color_discrete_sequence=["#355C7D"],
        title="Receita total por década de lançamento",
        labels={"release_decade": "Década", "total_revenue": "Receita"},
    )
    fig.update_traces(
        hovertemplate="<b>Década %{x}</b><br><br>Receita=%{customdata[0]}<br>Filmes=%{customdata[1]}<extra></extra>"
    )
    formatar_eixo_dinheiro(fig, "y")
    return aplicar_estilo(fig)


def grafico_generos_geral() -> go.Figure:
    df = adicionar_formatos_hover(genre_summary.sort_values("movie_count", ascending=False).head(12).sort_values("movie_count"))
    fig = px.bar(
        df,
        x="movie_count",
        y="genre",
        color="avg_vote",
        custom_data=["movie_count_fmt", "avg_vote_fmt"],
        color_continuous_scale=CONTINUOUS_SCALE,
        orientation="h",
        title="Gêneros com maior quantidade de filmes",
        labels={"movie_count": "Filmes", "genre": "Gênero", "avg_vote": "Nota média"},
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br><br>Filmes=%{customdata[0]}<br>Nota média=%{customdata[1]}<extra></extra>"
    )
    return aplicar_estilo(fig)


def grafico_orcamento_receita_geral() -> go.Figure:
    base = limitar_pontos_scatter(
        movies[movies["has_financials"] & (movies["vote_count"] >= 50)],
        maximo=MAX_OVERVIEW_SCATTER_POINTS,
        ordenar_por=["vote_count", "revenue_usd", "popularity"],
    )
    df = adicionar_formatos_hover(adicionar_colunas_financeiras_mi(base))
    fig = px.scatter(
        df,
        x="budget_mi",
        y="revenue_mi",
        color="primary_genre",
        size="popularity",
        hover_name="title",
        custom_data=["primary_genre", "budget_fmt", "revenue_fmt", "popularity_fmt"],
        render_mode=SCATTER_RENDER_MODE,
        color_discrete_sequence=PALETTE,
        title="Orçamento x bilheteria",
        labels={
            "budget_mi": "Orçamento",
            "revenue_mi": "Receita",
            "primary_genre": "Gênero",
            "popularity": "Popularidade",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>Gênero=%{customdata[0]}"
            "<br>Orçamento=%{customdata[1]}<br>Receita=%{customdata[2]}"
            "<br>Popularidade=%{customdata[3]}<extra></extra>"
        )
    )
    fig = aplicar_estilo(fig)
    fig.update_layout(margin=dict(l=82, r=30, t=54, b=74))
    formatar_eixo_milhoes(fig, "x")
    formatar_eixo_milhoes(fig, "y")
    return fig


def grafico_top_receita() -> go.Figure:
    df = adicionar_formatos_hover(movies.sort_values("revenue_usd", ascending=False).head(10).sort_values("revenue_usd"))
    df["title_chart"] = df["title"].apply(encurtar)
    fig = px.bar(
        df,
        x="revenue_usd",
        y="title_chart",
        color="primary_genre",
        custom_data=["primary_genre", "revenue_fmt"],
        orientation="h",
        hover_name="title",
        color_discrete_sequence=PALETTE,
        title="Filmes com maior receita",
        labels={"revenue_usd": "Receita", "title_chart": "Filme", "primary_genre": "Gênero"},
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>Gênero=%{customdata[0]}<br>Receita=%{customdata[1]}<extra></extra>"
    )
    formatar_eixo_dinheiro(fig, "x")
    fig = aplicar_estilo(fig)
    fig.update_layout(showlegend=False, margin=dict(l=210, r=24, t=54, b=46))
    return fig


def grafico_popularidade_nota() -> go.Figure:
    base = limitar_pontos_scatter(
        movies[(movies["vote_count"] >= 50) & movies["popularity"].notna()],
        maximo=MAX_OVERVIEW_SCATTER_POINTS,
        ordenar_por=["vote_count", "popularity"],
    )
    df = adicionar_formatos_hover(base)
    fig = px.scatter(
        df,
        x="popularity",
        y="vote_average",
        color="primary_genre",
        size="vote_count",
        hover_name="title",
        custom_data=["primary_genre", "popularity_fmt", "vote_average_fmt", "vote_count_fmt"],
        log_x=True,
        render_mode=SCATTER_RENDER_MODE,
        color_discrete_sequence=PALETTE,
        title="Popularidade x nota média",
        labels={"popularity": "Popularidade", "vote_average": "Nota média", "primary_genre": "Gênero"},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>Gênero=%{customdata[0]}"
            "<br>Popularidade=%{customdata[1]}<br>Nota média=%{customdata[2]}"
            "<br>Avaliações TMDb=%{customdata[3]}<extra></extra>"
        )
    )
    return aplicar_estilo(fig)


def layout_visao_geral() -> html.Div:
    receita_total = movies["revenue_usd"].sum()
    cards = [
        card_indicador("Filmes", inteiro(metrics["total_movies"]), "movies_metadata.csv"),
        card_indicador("Avaliações TMDb", inteiro(metrics["total_tmdb_votes"]), "vote_count somado"),
        card_indicador("Diretores", inteiro(metrics["total_directors"]), "credits.csv"),
        card_indicador("Com orçamento e receita", inteiro(metrics["movies_with_financials"]), "para análise financeira"),
        card_indicador("Receita total", dinheiro(receita_total), "filmes com bilheteria informada"),
    ]
    return html.Div(
        className="dashboard-section",
        children=[
            html.Div(cards, className="metric-grid"),
            html.Div(className="grid-2", children=[grafico(grafico_receita_decada()), grafico(grafico_generos_geral())]),
            html.Div(className="grid-2", children=[grafico(grafico_orcamento_receita_geral()), grafico(grafico_top_receita())]),
            grafico(grafico_popularidade_nota()),
        ],
    )


def layout_exploracao() -> html.Div:
    return html.Div(
        className="dashboard-section",
        children=[
            html.Div(
                className="filters-panel",
                children=[
                    html.Div(
                        className="filter-control",
                        children=[
                            html.Label("Gêneros"),
                            dcc.Dropdown(
                                id="genre-filter",
                                options=[{"label": genre, "value": genre} for genre in GENRES],
                                value=[],
                                multi=True,
                                placeholder="Todos",
                                clearable=True,
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-control filter-years",
                        children=[
                            html.Label("Ano de lançamento"),
                            html.Div(
                                className="year-inputs",
                                children=[
                                    dcc.Input(id="year-start", type="number", min=YEAR_MIN, max=YEAR_MAX, value=1980, debounce=True),
                                    html.Span("até", className="range-separator"),
                                    dcc.Input(id="year-end", type="number", min=YEAR_MIN, max=YEAR_MAX, value=YEAR_MAX, debounce=True),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-control compact",
                        children=[
                            html.Label("Mínimo de avaliações TMDb"),
                            html.Div(
                                className="number-stepper",
                                children=[
                                    html.Button("-", id="min-votes-minus", type="button", className="number-stepper-button"),
                                    html.Div("100", id="min-votes", className="number-stepper-value"),
                                    html.Button("+", id="min-votes-plus", type="button", className="number-stepper-button"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-control compact",
                        children=[
                            html.Label("Métrica do ranking"),
                            dcc.Dropdown(
                                id="metric-select",
                                options=[{"label": label, "value": value} for value, label in METRIC_OPTIONS.items()],
                                value="revenue_usd",
                                clearable=False,
                                searchable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-control compact",
                        children=[
                            html.Label("Financeiro"),
                            dcc.Checklist(
                                id="financial-only",
                                options=[{"label": "Somente com orçamento e receita", "value": "yes"}],
                                value=[],
                                className="check-row",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="filtered-kpis", className="metric-grid"),
            html.Div(className="grid-2", children=[grafico(graph_id="top-movies-filtered"), grafico(graph_id="budget-revenue-filtered")]),
            html.Div(className="grid-2", children=[grafico(graph_id="popularity-rating-filtered"), grafico(graph_id="genre-trend-filtered")]),
            grafico(graph_id="directors-filtered"),
        ],
    )


app = Dash(__name__, assets_folder=str(ROOT / "assets"))
app.title = "Análise de Filmes"
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" href="data:,">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
server = app.server

app.layout = html.Div(
    className="page-shell",
    children=[
        html.Header(
            className="app-header",
            children=[
                html.Div(
                    [
                        html.Span("Kaggle: The Movies Dataset", className="eyebrow"),
                        html.H1("Análise do mercado cinematográfico e das avaliações de filmes"),
                    ]
                ),
            ],
        ),
        dcc.Tabs(
            id="dashboards",
            value="geral",
            className="tabs",
            children=[
                dcc.Tab(label="Dashboard 1 - Visão Geral", value="geral", children=layout_visao_geral()),
                dcc.Tab(label="Dashboard 2 - Exploração Interativa", value="exploracao", children=layout_exploracao()),
            ],
        ),
    ],
)


@app.callback(
    Output("min-votes", "children"),
    Input("min-votes-minus", "n_clicks"),
    Input("min-votes-plus", "n_clicks"),
    State("min-votes", "children"),
    prevent_initial_call=True,
)
def ajustar_minimo_avaliacoes(_, __, valor_atual):
    valor = normalizar_minimo_avaliacoes(valor_atual)

    if ctx.triggered_id == "min-votes-plus":
        return str(valor + 50)
    if ctx.triggered_id == "min-votes-minus":
        return str(max(0, valor - 50))
    return str(valor)


@app.callback(
    Output("filtered-kpis", "children"),
    Output("top-movies-filtered", "figure"),
    Output("budget-revenue-filtered", "figure"),
    Output("popularity-rating-filtered", "figure"),
    Output("genre-trend-filtered", "figure"),
    Output("directors-filtered", "figure"),
    Input("genre-filter", "value"),
    Input("year-start", "value"),
    Input("year-end", "value"),
    Input("min-votes", "children"),
    Input("metric-select", "value"),
    Input("financial-only", "value"),
)
def atualizar_exploracao(genres, year_start, year_end, min_votes, metric, financial_only):
    metric = metric if metric in METRIC_OPTIONS else "revenue_usd"
    filtro_financeiro_ativo = metric in FINANCIAL_METRICS or "yes" in (financial_only or [])
    df = filtrar_filmes(genres, year_start, year_end, min_votes, filtro_financeiro_ativo)
    if df.empty:
        cards = [
            card_indicador("Filmes", "0"),
            card_indicador("Receita", "-"),
            card_indicador("Nota média", "-"),
            card_indicador("Gênero líder", "-"),
        ]
        vazia = figura_vazia("Filtros sem resultado")
        return cards, vazia, vazia, vazia, vazia, vazia

    receita = df["revenue_usd"].sum()
    nota = df["vote_average"].mean()
    genero_lider = df["primary_genre"].value_counts().index[0]
    detalhe_filmes = "com orçamento e receita" if filtro_financeiro_ativo else "após filtros"
    cards = [
        card_indicador("Filmes", inteiro(df.shape[0]), detalhe_filmes),
        card_indicador("Receita", dinheiro(receita), "total filtrado"),
        card_indicador("Nota média", f"{nota:.2f}", "TMDb"),
        card_indicador("Gênero líder", str(genero_lider), "maior volume"),
        card_indicador("Diretores", inteiro(df["director"].nunique()), "no filtro"),
    ]

    top_base = df.copy()
    titulo_top = f"Top filmes por {METRIC_OPTIONS[metric].lower()}"
    rotulo_metrica = METRIC_OPTIONS[metric]
    if metric == "roi":
        top_base = top_base[top_base["budget_usd"] >= 1_000_000]
        titulo_top = "Top filmes por retorno sobre orçamento (orçamento >= US$ 1 mi)"
        rotulo_metrica = "Retorno (vezes o orçamento)"

    top = adicionar_formatos_hover(top_base.sort_values(metric, ascending=False).head(15).sort_values(metric))
    top["title_chart"] = top["title"].apply(encurtar)
    top["metric_fmt"] = top[metric].apply(lambda valor: formatar_valor_metrica(valor, metric))
    if top.empty:
        fig_top = figura_vazia(titulo_top)
    else:
        fig_top = px.bar(
            top,
            x=metric,
            y="title_chart",
            color="primary_genre",
            orientation="h",
            hover_name="title",
            custom_data=["primary_genre", "metric_fmt"],
            color_discrete_sequence=PALETTE,
            title=titulo_top,
            labels={metric: rotulo_metrica, "title_chart": "Filme", "primary_genre": "Gênero"},
        )
        fig_top.update_traces(
            hovertemplate=(
                f"<b>%{{hovertext}}</b><br><br>Gênero=%{{customdata[0]}}"
                f"<br>{rotulo_metrica}=%{{customdata[1]}}<extra></extra>"
            )
        )
        if metric == "revenue_usd":
            formatar_eixo_dinheiro(fig_top, "x")
        if metric == "roi":
            fig_top.update_xaxes(ticksuffix="x")
        fig_top = aplicar_estilo(fig_top)
        fig_top.update_layout(showlegend=False, margin=dict(l=210, r=24, t=54, b=46))

    financeiro_base = limitar_pontos_scatter(
        df[df["has_financials"]],
        ordenar_por=["vote_count", "revenue_usd", "popularity"],
    )
    financeiro = adicionar_formatos_hover(adicionar_colunas_financeiras_mi(financeiro_base))
    if financeiro.empty:
        fig_budget = figura_vazia("Orçamento x bilheteria")
    else:
        fig_budget = px.scatter(
            financeiro,
            x="budget_mi",
            y="revenue_mi",
            color="primary_genre",
            size="popularity",
            hover_name="title",
            custom_data=["primary_genre", "budget_fmt", "revenue_fmt", "popularity_fmt"],
            render_mode=SCATTER_RENDER_MODE,
            color_discrete_sequence=PALETTE,
            title="Orçamento x bilheteria",
            labels={
                "budget_mi": "Orçamento",
                "revenue_mi": "Receita",
                "primary_genre": "Gênero",
                "popularity": "Popularidade",
            },
        )
        fig_budget.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>Gênero=%{customdata[0]}"
                "<br>Orçamento=%{customdata[1]}<br>Receita=%{customdata[2]}"
                "<br>Popularidade=%{customdata[3]}<extra></extra>"
            )
        )
        fig_budget = aplicar_estilo(fig_budget)
        fig_budget.update_layout(margin=dict(l=82, r=30, t=54, b=74))
        formatar_eixo_milhoes(fig_budget, "x")
        formatar_eixo_milhoes(fig_budget, "y")

    pop_base = limitar_pontos_scatter(
        df[df["popularity"].notna()],
        ordenar_por=["vote_count", "popularity"],
    )
    pop_df = adicionar_formatos_hover(pop_base)
    fig_pop = px.scatter(
        pop_df,
        x="popularity",
        y="vote_average",
        color="primary_genre",
        size="vote_count",
        hover_name="title",
        custom_data=["primary_genre", "popularity_fmt", "vote_average_fmt", "vote_count_fmt"],
        log_x=True,
        render_mode=SCATTER_RENDER_MODE,
        color_discrete_sequence=PALETTE,
        title="Popularidade x nota média",
        labels={"popularity": "Popularidade", "vote_average": "Nota média", "primary_genre": "Gênero"},
    )
    fig_pop.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>Gênero=%{customdata[0]}"
            "<br>Popularidade=%{customdata[1]}<br>Nota média=%{customdata[2]}"
            "<br>Avaliações TMDb=%{customdata[3]}<extra></extra>"
        )
    )
    fig_pop = aplicar_estilo(fig_pop)

    trend_base = df[["tmdb_id", "genres_text", "release_decade", "vote_average", "popularity", "revenue_usd"]].copy()
    trend_base["genre"] = trend_base["genres_text"].str.split(", ")
    trend_base = trend_base.explode("genre")
    if genres:
        trend_base = trend_base[trend_base["genre"].isin(genres)]
    else:
        top_genres = df["primary_genre"].value_counts().head(5).index
        trend_base = trend_base[trend_base["genre"].isin(top_genres)]
    trend_base = (
        trend_base.dropna(subset=["release_decade"])
        .groupby(["genre", "release_decade"], as_index=False)
        .agg(
            movie_count=("tmdb_id", "nunique"),
            avg_vote=("vote_average", "mean"),
            total_revenue=("revenue_usd", "sum"),
        )
    )
    if trend_base.empty:
        fig_trend = figura_vazia("Evolução dos gêneros por década")
    else:
        trend_base = adicionar_formatos_hover(trend_base)
        fig_trend = px.line(
            trend_base,
            x="release_decade",
            y="movie_count",
            color="genre",
            markers=True,
            custom_data=["genre", "movie_count_fmt", "avg_vote_fmt", "total_revenue_fmt"],
            color_discrete_sequence=PALETTE,
            title="Evolução dos gêneros por década",
            labels={"release_decade": "Década", "movie_count": "Quantidade de filmes", "genre": "Gênero"},
        )
        fig_trend.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>Década=%{x}"
                "<br>Filmes=%{customdata[1]}<br>Nota média=%{customdata[2]}"
                "<br>Receita=%{customdata[3]}<extra></extra>"
            )
        )
        fig_trend = aplicar_estilo(fig_trend, legend=True)

    directors = (
        df[df["director"] != "Não informado"]
        .groupby("director")
        .agg(movie_count=("tmdb_id", "nunique"), total_revenue=("revenue_usd", "sum"), avg_vote=("vote_average", "mean"))
        .reset_index()
        .sort_values(["total_revenue", "movie_count"], ascending=False)
        .head(12)
        .sort_values("total_revenue")
    )
    if directors.empty:
        fig_directors = figura_vazia("Diretores mais relevantes")
    else:
        directors = adicionar_formatos_hover(directors)
        fig_directors = px.bar(
            directors,
            x="total_revenue",
            y="director",
            orientation="h",
            color="avg_vote",
            custom_data=["total_revenue_fmt", "avg_vote_fmt", "movie_count_fmt"],
            color_continuous_scale=CONTINUOUS_SCALE,
            title="Diretores por receita total",
            labels={"total_revenue": "Receita", "director": "Diretor", "avg_vote": "Nota média"},
        )
        fig_directors.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br><br>Receita=%{customdata[0]}"
                "<br>Nota média dos filmes=%{customdata[1]}<br>Filmes=%{customdata[2]}<extra></extra>"
            )
        )
        formatar_eixo_dinheiro(fig_directors, "x")
        fig_directors = aplicar_estilo(fig_directors)

    return cards, fig_top, fig_budget, fig_pop, fig_trend, fig_directors


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8050")), debug=False)
