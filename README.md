# Trabalho Prático 1 - Recuperação de Informação (SCC0282)

Sistema de recuperação textual sobre a coleção **Cranfield**, com **Modelo
Vetorial** (TF-IDF + similaridade do cosseno) e **Modelo Probabilístico**
(**BM25** implementado explicitamente), avaliados com Precision@10, Recall@10,
F1@10, MAP, MRR e NDCG@10.

ICMC / USP - 2º semestre de 2026 - Prof. Marcelo G. Manzato

---

## 1. Integrantes do grupo

| Nome completo | Nº USP | E-mail |
|---|---|---|
| Victor Silva Botelho | 15645421 | victorbotelho@usp.br |
| João Pedro Barbosa Madeira | 13683038 | Joaopedro08madeira@gmail.com |

Os mesmos dados constam na capa do relatório (`report/relatorio.tex`).

---

## 2. Instalação e execução

### Requisitos

* **Python 3.12** ou superior (desenvolvido e testado em 3.12.13)
* Conexão com a internet no primeiro uso, para baixar a coleção Cranfield e a
  lista de stopwords do NLTK

### Passo a passo

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

No Linux/macOS, use `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

```bash
python run_all.py
```

O comando acima baixa a coleção, executa **todos** os experimentos e gera
todas as figuras. Tempo típico: cerca de **15 segundos** após o download.

### Opções

| Comando | Efeito |
|---|---|
| `python run_all.py` | baixa a coleção (se necessário) e roda tudo |
| `python run_all.py --no-download` | usa os arquivos já presentes em `data/cran` |
| `python run_all.py --skip-figures` | apenas os resultados numéricos |
| `python tests/test_ir_tp1.py` | executa os 16 testes de correção |
| `python report/build_report.py` | regera `report/relatorio.pdf` |

### Aviso para Windows

O projeto cria caminhos relativamente profundos dentro de `.venv`. Se a pasta
do projeto já estiver em um caminho muito longo (por exemplo, dentro de uma
pasta sincronizada do Google Drive), o limite de 260 caracteres do Windows
(`MAX_PATH`) pode corromper a instalação de pacotes como o SciPy, causando
erros do tipo `ModuleNotFoundError: No module named
'scipy.linalg._cythonized_array_utils'`. Nesse caso, crie o ambiente virtual
em um caminho curto (por exemplo, `C:\venvs\ri`) ou habilite o suporte a
caminhos longos no Windows.

---

## 3. Versões da linguagem e das bibliotecas

| Componente | Versão usada | Papel no projeto |
|---|---|---|
| Python | 3.12.13 | linguagem |
| NumPy | 2.5.3 | álgebra vetorial |
| SciPy | 1.18.1 | matrizes esparsas (CSR/CSC), correlação de Spearman |
| scikit-learn | 1.9.1 | **apenas** na verificação cruzada dos testes |
| NLTK | 3.10.3 | lista de stopwords e stemmer de Porter |
| pandas | 3.0.6 | tabelas de resultados e escrita dos CSVs |
| matplotlib | 3.11.2 | figuras |
| ReportLab | 5.0.1 | geração do relatório em PDF |

As versões exatas estão fixadas em `requirements.txt`.

**O que foi implementado do zero:** a ponderação TF-IDF, a similaridade do
cosseno, a função de score do BM25, o índice invertido e **todas** as métricas
de avaliação. As bibliotecas são usadas para tarefas auxiliares
(tokenização auxiliar, stopwords, stemming, álgebra esparsa, gráficos). O
`TfidfVectorizer` do scikit-learn aparece **exclusivamente** em
`tests/test_ir_tp1.py`, como verificação independente do nosso modelo
vetorial, nunca no pipeline de resultados.

---

## 4. Base de dados

**Coleção Cranfield** (1400 documentos, 225 consultas, 1837 julgamentos de
relevância).

* **Obtenção:** automática. `run_all.py` baixa
  `http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz`
  (University of Glasgow) e extrai em `data/cran/`. O SHA-256 do arquivo é
  conferido (`8c48a841...0652d64`).
* **Fonte alternativa:** `ir_datasets` (<https://ir-datasets.com/cranfield.html>)
  e Hugging Face (`irds/cranfield`). Os três fornecem os mesmos dados.
* **Arquivos:** `cran.all.1400` (documentos), `cran.qry` (consultas),
  `cranqrel` (julgamentos).

### Dois detalhes da coleção que o código trata explicitamente

1. **Os identificadores das consultas não são sequenciais.** O campo `.I` de
   `cran.qry` vale `001, 002, 004, 008, …, 365`, mas o arquivo `cranqrel`
   numera as consultas de **1 a 225 pela posição no arquivo**. Tratar o `.I`
   como chave do qrel desalinha silenciosamente consultas e julgamentos e
   produz métricas muito baixas. O código renumera por posição (mesma
   convenção do `ir_datasets`) e guarda o `.I` original em
   `Query.original_id`. Há um teste que verifica esse alinhamento.

2. **A escala de relevância é invertida.** Segundo `cranqrel.readme`, o grau
   **1 é o melhor** ("resposta completa") e o **4 é o pior** ("interesse
   mínimo"); `-1` significa "sem interesse". Para as métricas binárias
   seguimos o enunciado (relevante ⇔ grau ≥ 1; `-1` e não julgados contam
   como não relevantes). Para o NDCG, que usa relevância graduada, mapeamos
   grau → ganho invertendo a escala (1→4, 2→3, 3→2, 4→1), pois usar o grau
   bruto daria o maior ganho ao documento **menos** relevante. As duas
   convenções são reportadas em `results/07_ndcg_gain_mappings.csv`.

---

## 5. Estrutura do projeto

```
.
├── run_all.py                     # ponto de entrada: reproduz tudo
├── requirements.txt
├── README.md
├── src/ir_tp1/
│   ├── dataset.py                 # download e parsing do Cranfield
│   ├── preprocess.py              # tokenização, stopwords, stemming (req. 1)
│   ├── indexing.py                # índice invertido compartilhado
│   ├── vsm.py                     # modelo vetorial TF-IDF + cosseno (req. 2)
│   ├── bm25.py                    # BM25 explícito (req. 3)
│   ├── metrics.py                 # P@k, R@k, F1, MAP, MRR, NDCG, curva P×R (req. 4)
│   ├── query_reformulations.py    # as 5 reformulações manuais (req. 8)
│   ├── experiments.py             # todos os experimentos (req. 1–9)
│   └── plots.py                   # figuras
├── tests/test_ir_tp1.py           # 16 testes de correção
├── report/ 
│   └── relatorio.pdf              # relatório final (6 páginas)
├── data/cran/                     # coleção (baixada automaticamente)
├── results/                       # todos os resultados (CSV + JSON)
└── figures/                       # figuras em PNG
```

---

## 6. Resultados completos

Tudo o que foi usado para produzir as tabelas e os gráficos do relatório está
em `results/`, **inclusive os resultados por consulta**:

| Arquivo | Conteúdo |
|---|---|
| `00_run_metadata.json` | parâmetros e resumo da execução |
| `01_preprocessing_summary.csv` | 4 configurações × 2 modelos, agregado (req. 1) |
| `01_preprocessing_per_query.csv` | **o mesmo, por consulta** (1800 linhas) |
| `01_preprocessing_vocab.csv` | tamanho de vocabulário e comprimento médio |
| `02_per_query_metrics.csv` | **todas as métricas por consulta** (req. 4) |
| `02_model_summary.csv` / `.json` | agregado e contagem de vitórias (req. 5) |
| `02_model_comparison_per_query.csv` | VSM × BM25 lado a lado, por consulta |
| `02_precision_recall_curves.json` | curvas interpoladas em 11 pontos |
| `03_query_analysis.json` / `_top5.csv` | as 6 consultas do req. 6, com Top-5 anotado |
| `04_bm25_parameter_grid.csv` | grade k1 × b agregada (req. 7) |
| `04_bm25_parameter_per_query.csv` | **a grade, por consulta** (2025 linhas) |
| `04_effect_of_b_per_query.csv` | efeito de b em cada consulta |
| `04_effect_of_b_showcase.json` | a consulta escolhida para ilustrar b |
| `05_query_reformulation.csv` / `_wide.csv` / `_detail.json` | req. 8 |
| `06_error_analysis.json` / `_false_*.csv` | req. 9, com decomposição do score |
| `07_idf_variants.csv` | Robertson × BIM |
| `07_ndcg_gain_mappings.csv` | ganho invertido × bruto |
| `08_length_analysis_*` | teste da hipótese do comprimento dos documentos |

### Resumo (225 consultas, pré-processamento `stopwords + stemming`)

| Modelo | P@10 | R@10 | F1@10 | MAP | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| Modelo Vetorial | 0,2169 | 0,3785 | 0,2488 | 0,2763 | 0,4869 | 0,3263 |
| BM25 (k1=1,2; b=0,75) | **0,2342** | **0,3976** | **0,2662** | **0,3048** | **0,5391** | **0,3546** |

Melhor configuração da grade: **k1 = 2,0; b = 0,75** → MAP **0,3115**.

---

## 7. Reprodutibilidade

* Os rankings são **determinísticos**: empates de score são desfeitos pelo
  `doc_id` crescente (`_rank_from_scores`), e não há nenhuma fonte de
  aleatoriedade no pipeline. Duas execuções produzem CSVs idênticos.
* Os **qrels são usados exclusivamente para avaliar**. Nenhuma etapa de
  indexação, ponderação, escolha de parâmetros ou reformulação de consultas
  consulta os julgamentos de relevância.
* A escolha da configuração de pré-processamento principal e a seleção das
  consultas analisadas nos requisitos 6, 7 e 9 são **automáticas e
  determinísticas** (critérios documentados em `experiments.py`), não
  escolhas manuais feitas após ver os resultados.
* As cinco reformulações do requisito 8 foram escritas manualmente, e a
  hipótese de cada uma está registrada no código (campo `hypothesis` em
  `query_reformulations.py`) **antes** de executar os modelos.

### Testes

```bash
python tests/test_ir_tp1.py
```

Os 16 testes cobrem três frentes:

1. **Métricas contra valores calculados à mão**, usando o exercício do slide 41
   da Aula 05 (P@5, MRR, AP@3, NDCG@5), de modo que os números conferem com o
   que foi visto em aula.
2. **Implementação de referência ingênua**: uma versão em Python puro, escrita
   com laços diretamente a partir das fórmulas, é comparada com a versão
   vetorizada em SciPy sobre a coleção real (`atol=1e-10`). Isso valida a
   otimização com matrizes esparsas tanto do modelo vetorial quanto do BM25,
   em três combinações de `(k1, b)`.
3. **Verificação cruzada com o scikit-learn**: correlação de Spearman > 0,95
   entre o nosso ranking e o do `TfidfVectorizer` + `cosine_similarity`
   (não se espera igualdade exata, pois o scikit-learn usa
   `idf = ln(N/df) + 1`, enquanto a Aula 04 define `idf = log₁₀(N/n)`).

---

## 8. Uso de ferramentas de IA

Consta na seção correspondente do relatório
(`report/relatorio.pdf`, seção 7).
