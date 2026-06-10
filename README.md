# Trabalho Final - The Movies Dataset

Tema: **Análise do mercado cinematográfico e do comportamento de avaliações de filmes**.

O projeto usa o dataset público **The Movies Dataset**, do Kaggle. A base atende ao enunciado porque possui mais de **46 mil filmes**, avaliações TMDb agregadas nos metadados e dois arquivos CSV integráveis.

## Arquivos Usados

- `movies_metadata.csv`: metadados dos filmes, orçamento, receita, idioma, popularidade, nota, data de lançamento.
- `credits.csv`: elenco e equipe, incluindo diretores.

## Estrutura

- `data/raw/the-movies-dataset`: dados brutos do Kaggle.
- `data/processed`: tabelas processadas pelo pipeline.
- `src/01_coletar_dados.py`: coleta automática via Kaggle.
- `src/02_pipeline_preprocessamento.py`: integração, limpeza e transformação.
- `src/03_analise_exploratoria.py`: gera insights automáticos.
- `src/dashboard_app.py`: dashboard em Dash.

## Como Rodar

```powershell
cd trabalho_filmes_movielens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para gerar os dados processados a partir dos arquivos brutos:

```powershell
python src\01_coletar_dados.py
python src\02_pipeline_preprocessamento.py
```

Para abrir o dashboard:

```powershell
python src\dashboard_app.py
```

Depois acesse:

```text
http://127.0.0.1:8050
```

Para rodar também a análise exploratória:

```powershell
python src\03_analise_exploratoria.py
```

## O Que Foi Desenvolvido

O pipeline realiza:

1. Aquisição automática dos dados do Kaggle.
2. Integração com `merge` entre filmes e créditos.
3. Limpeza de IDs, datas, orçamento, receita, popularidade, notas e valores ausentes.
4. Transformações: década, gênero principal, lucro, ROI, score confiável, faixas e variáveis codificadas.
5. Agregações por gênero, década e diretor.
6. Dois dashboards em Dash.

## Dashboards

**Dashboard 1 - Visão Geral**

Painel executivo com indicadores, receita por década, gêneros com mais filmes, orçamento x bilheteria, ranking de maior receita e popularidade x nota média.

**Dashboard 2 - Exploração Interativa**

Permite filtrar por gênero, ano, mínimo de avaliações TMDb, métrica de ranking e presença de dados financeiros. As métricas do ranking são receita, nota média e retorno sobre orçamento. Mostra ranking de filmes, orçamento x bilheteria, popularidade x nota, evolução dos gêneros e diretores por receita.
