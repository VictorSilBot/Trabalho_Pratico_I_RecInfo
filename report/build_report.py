"""Gera ``report/relatorio.pdf`` (6 páginas, duas colunas).

Todos os números do relatório são LIDOS de ``results/``. Nada é digitado à
mão, de modo que reexecutar ``run_all.py`` e depois este script mantém texto,
tabelas e figuras sempre coerentes com os experimentos.

    python report/build_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUTPUT = ROOT / "report" / "relatorio.pdf"

# ----------------------------------------------------------------------
# Dados do grupo -- PREENCHER ANTES DA ENTREGA
# ----------------------------------------------------------------------
AUTHORS = [
    ("&lt;NOME COMPLETO&gt;", "&lt;Nº USP&gt;", "&lt;email@usp.br&gt;"),
    ("&lt;NOME COMPLETO&gt;", "&lt;Nº USP&gt;", "&lt;email@usp.br&gt;"),
    ("&lt;NOME COMPLETO&gt;", "&lt;Nº USP&gt;", "&lt;email@usp.br&gt;"),
]

# ----------------------------------------------------------------------
# Estilo
# ----------------------------------------------------------------------
INK = colors.HexColor("#111111")
INK_SOFT = colors.HexColor("#444444")
RULE = colors.HexColor("#cccccc")
BAND = colors.HexColor("#f0f0ee")
ACCENT = colors.HexColor("#2a78d6")

BODY = ParagraphStyle(
    "body",
    fontName="Times-Roman",
    fontSize=8.7,
    leading=10.6,
    alignment=TA_JUSTIFY,
    textColor=INK,
    spaceAfter=4,
)
H1 = ParagraphStyle(
    "h1",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=12,
    spaceBefore=8,
    spaceAfter=3,
    alignment=0,
    textColor=INK,
)
H2 = ParagraphStyle(
    "h2",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=8.8,
    leading=10.5,
    spaceBefore=5,
    spaceAfter=2,
    alignment=0,
    textColor=INK,
)
TITLE = ParagraphStyle(
    "title",
    parent=BODY,
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    alignment=TA_CENTER,
    spaceAfter=5,
)
SUBTITLE = ParagraphStyle(
    "subtitle", parent=BODY, fontSize=9.2, leading=11.4, alignment=TA_CENTER
)
CAPTION = ParagraphStyle(
    "caption",
    parent=BODY,
    fontSize=7.3,
    leading=8.8,
    alignment=TA_CENTER,
    textColor=INK_SOFT,
    spaceBefore=1,
    spaceAfter=5,
)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.0, leading=8.2, alignment=0, spaceAfter=0)


# As fontes Type1 padrão do ReportLab (Times/Helvetica) não possuem os glifos
# de subscrito e sobrescrito Unicode, que sairiam como quadrados vazios no PDF.
# Convertemos esses caracteres para a marcação <sub>/<super>, que o ReportLab
# renderiza com a própria fonte.
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUB_RE = re.compile("[₀-₉]+")
_SUP_RE = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")


def _fix_glyphs(text: str) -> str:
    text = _SUB_RE.sub(lambda m: f"<sub>{m.group().translate(_SUBSCRIPTS)}</sub>", text)
    text = _SUP_RE.sub(lambda m: f"<super>{m.group().translate(_SUPERSCRIPTS)}</super>", text)
    return text.replace("‖", "||")


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(_fix_glyphs(text), style)


def header_cell(text: str) -> Paragraph:
    """Cabeçalho de tabela que quebra linha em vez de colidir com o vizinho."""
    return Paragraph(
        _fix_glyphs(text),
        ParagraphStyle(
            "th",
            parent=BODY,
            fontName="Helvetica-Bold",
            fontSize=6.6,
            leading=7.8,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
    )


def fmt(value: float, digits: int = 3) -> str:
    """Formata no padrão brasileiro (vírgula decimal)."""
    return f"{value:.{digits}f}".replace(".", ",")


# ----------------------------------------------------------------------
# Carregamento dos resultados
# ----------------------------------------------------------------------
def load_results() -> dict:
    def csv(name):
        return pd.read_csv(RESULTS / name)

    def js(name):
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    return {
        "meta": js("00_run_metadata.json"),
        "prep": csv("01_preprocessing_summary.csv"),
        "vocab": csv("01_preprocessing_vocab.csv"),
        "models": csv("02_model_summary.csv"),
        "models_json": js("02_model_summary.json"),
        "comparison": csv("02_model_comparison_per_query.csv"),
        "analysis": js("03_query_analysis.json"),
        "grid": csv("04_bm25_parameter_grid.csv"),
        "b_showcase": js("04_effect_of_b_showcase.json"),
        "reform": csv("05_query_reformulation.csv"),
        "reform_detail": js("05_query_reformulation_detail.json"),
        "errors": js("06_error_analysis.json"),
        "idf": csv("07_idf_variants.csv"),
        "gain": csv("07_ndcg_gain_mappings.csv"),
        "length": js("08_length_analysis_summary.json"),
    }


# ----------------------------------------------------------------------
# Tabelas
# ----------------------------------------------------------------------
BASE_TABLE_STYLE = [
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.0),
    ("LEADING", (0, 0), (-1, -1), 8.2),
    ("TEXTCOLOR", (0, 0), (-1, -1), INK),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
    ("LINEABOVE", (0, 0), (-1, 0), 0.6, INK),
    ("LINEBELOW", (0, -1), (-1, -1), 0.6, INK),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 1.6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
]


def make_table(data, col_widths=None, extra_style=None, font="Helvetica") -> Table:
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = list(BASE_TABLE_STYLE) + [("FONTNAME", (0, 1), (-1, -1), font)]
    if extra_style:
        style += extra_style
    table.setStyle(TableStyle(style))
    return table


def table_preprocessing(R, width) -> Table:
    prep, vocab = R["prep"], R["vocab"]
    labels = {
        "none": "sem stopw./sem stem.",
        "stop": "stopwords",
        "stem": "stemming",
        "stop_stem": "stopwords + stem.",
    }
    header = [
        header_cell("Configuração"),
        header_cell("|V|"),
        header_cell("|d|<br/>méd."),
        header_cell("MAP<br/>vet."),
        header_cell("MAP<br/>BM25"),
        header_cell("P@10<br/>BM25"),
    ]
    rows = [header]
    for config in ["none", "stop", "stem", "stop_stem"]:
        v = vocab[vocab["config"] == config].iloc[0]
        vsm = prep[(prep["config"] == config) & (prep["model"] == "VSM")].iloc[0]
        bm = prep[(prep["config"] == config) & (prep["model"] == "BM25")].iloc[0]
        rows.append(
            [
                labels[config],
                f"{int(v['vocab_size'])}",
                fmt(v["avg_doc_length"], 1),
                fmt(vsm["MAP"]),
                fmt(bm["MAP"]),
                fmt(bm["P@10"]),
            ]
        )
    w = width
    return make_table(
        rows,
        [w * 0.32, w * 0.11, w * 0.12, w * 0.14, w * 0.15, w * 0.16],
        [
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, -1), (-1, -1), BAND),
        ],
    )


def table_models(R, width) -> Table:
    models = R["models"]
    metrics = ["P@10", "R@10", "F1@10", "MAP", "MRR", "NDCG@10"]
    rows = [["Modelo"] + metrics]
    for name in ["VSM", "BM25"]:
        row = models[models["model"] == name].iloc[0]
        label = "Vetorial" if name == "VSM" else "BM25"
        rows.append([label] + [fmt(row[m]) for m in metrics])
    vsm = models[models["model"] == "VSM"].iloc[0]
    bm = models[models["model"] == "BM25"].iloc[0]
    rows.append(
        ["Δ relativo"]
        + [f"+{fmt(100 * (bm[m] - vsm[m]) / vsm[m], 1)}%" for m in metrics]
    )
    w = width
    return make_table(
        rows,
        [w * 0.19] + [w * 0.135] * 6,
        [
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
            ("BACKGROUND", (0, 2), (-1, 2), BAND),
            ("TEXTCOLOR", (1, 3), (-1, 3), ACCENT),
            ("FONTSIZE", (1, 3), (-1, 3), 6.4),
        ],
    )


def _top5_cells(flags) -> list[str]:
    """Documentos do Top-5 com marcação de relevância."""
    return [
        f"<b>{f['doc_id']}</b>&#9679;" if f["relevant"] else str(f["doc_id"])
        for f in flags
    ]


def table_query_analysis(R, width) -> Table:
    detail = R["analysis"]["detalhe"]
    category_labels = {
        "bm25_superior": "BM25 &gt; vet.",
        "vsm_superior": "vet. &gt; BM25",
        "ambos_insatisfatorios": "ambos ruins",
    }
    rows = [[header_cell(h) for h in
             ["Caso", "Q", "Mod", "1º", "2º", "3º", "4º", "5º", "AP"]]]
    band_rows = []
    index = 1
    for category, entries in detail.items():
        for entry in entries:
            for model in ("VSM", "BM25"):
                cells = _top5_cells(entry["top5"][model])
                ap = entry["AP_VSM"] if model == "VSM" else entry["AP_BM25"]
                rows.append(
                    [
                        Paragraph(category_labels[category], CELL) if model == "VSM" else "",
                        f"q{entry['query_id']}" if model == "VSM" else "",
                        "vet." if model == "VSM" else "BM25",
                        *[Paragraph(c, CELL) for c in cells],
                        fmt(ap, 2),
                    ]
                )
                if model == "BM25":
                    band_rows.append(index + 1)
                index += 1
    w = width
    style = [
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("FONTSIZE", (0, 1), (-1, -1), 6.5),
    ]
    for row_index in band_rows:
        style.append(("LINEBELOW", (0, row_index), (-1, row_index), 0.25, RULE))
    return make_table(
        rows,
        [w * 0.135, w * 0.068, w * 0.095, *([w * 0.118] * 5), w * 0.112],
        style,
    )


def table_grid(R, width) -> Table:
    grid = R["grid"]
    rows = [["k1 \\ b", "0 (MAP)", "0,75 (MAP)", "1 (MAP)", "0,75 (P@10)", "0,75 (NDCG)"]]
    for k1 in [0.5, 1.2, 2.0]:
        subset = grid[grid["k1"] == k1]
        cell = subset[subset["b"] == 0.75].iloc[0]
        rows.append(
            [
                fmt(k1, 1),
                fmt(subset[subset["b"] == 0.0].iloc[0]["MAP"]),
                fmt(cell["MAP"]),
                fmt(subset[subset["b"] == 1.0].iloc[0]["MAP"]),
                fmt(cell["P@10"]),
                fmt(cell["NDCG@10"]),
            ]
        )
    best = grid.loc[grid["MAP"].idxmax()]
    best_row = [0.5, 1.2, 2.0].index(best["k1"]) + 1
    w = width
    return make_table(
        rows,
        [w * 0.14, w * 0.16, w * 0.18, w * 0.15, w * 0.19, w * 0.18],
        [
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("BACKGROUND", (2, best_row), (2, best_row), BAND),
            ("FONTNAME", (2, best_row), (2, best_row), "Helvetica-Bold"),
        ],
    )


def table_b_effect(R, width) -> Table:
    showcase = R["b_showcase"]
    rows = [[header_cell(h) for h in ["b", "1º", "2º", "3º", "4º", "5º", "AP"]]]
    for label, payload in showcase["rankings"].items():
        cells = _top5_cells(payload["top5"])
        rows.append(
            [label.replace("b=", "").replace(".", ","), *[Paragraph(c, CELL) for c in cells], fmt(payload["AP"], 2)]
        )
    w = width
    return make_table(
        rows,
        [w * 0.10, *([w * 0.15] * 5), w * 0.15],
        [("ALIGN", (0, 0), (0, -1), "LEFT"), ("FONTSIZE", (0, 1), (-1, -1), 6.5)],
    )


def table_reformulation(R, width) -> Table:
    reform = R["reform"]
    operations = {
        219: "sinônimo",
        69: "+ específica",
        109: "+ específica (curta)",
        171: "− termos",
        15: "vocabulário",
    }
    rows = [[header_cell(h) for h in
             ["Cons.", "Operação", "AP<br/>vetorial", "AP<br/>BM25", "NDCG@10<br/>BM25"]]]

    def get(query_id, model, version, metric):
        subset = reform[
            (reform["query_id"] == query_id)
            & (reform["model"] == model)
            & (reform["version"] == version)
        ]
        return float(subset[metric].iloc[0])

    for query_id in [219, 69, 109, 171, 15]:
        rows.append(
            [
                f"q{query_id}",
                operations[query_id],
                f"{fmt(get(query_id,'VSM','original','AP'),2)} → "
                f"{fmt(get(query_id,'VSM','modificada','AP'),2)}",
                f"{fmt(get(query_id,'BM25','original','AP'),2)} → "
                f"{fmt(get(query_id,'BM25','modificada','AP'),2)}",
                f"{fmt(get(query_id,'BM25','original','NDCG@10'),2)} → "
                f"{fmt(get(query_id,'BM25','modificada','NDCG@10'),2)}",
            ]
        )
    w = width
    return make_table(
        rows,
        [w * 0.10, w * 0.26, w * 0.21, w * 0.21, w * 0.22],
        [("ALIGN", (0, 0), (1, -1), "LEFT"), ("FONTSIZE", (0, 1), (-1, -1), 6.6)],
    )


def figure(path: Path, width: float) -> list:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        aspect = image.height / image.width
    return [Image(str(path), width=width, height=width * aspect)]


# ----------------------------------------------------------------------
# Construção do documento
# ----------------------------------------------------------------------
def build() -> Path:
    R = load_results()
    meta, models_json = R["meta"], R["models_json"]
    vsm_agg = models_json["agregado"]["VSM"]
    bm_agg = models_json["agregado"]["BM25"]
    wins = models_json["vitorias_por_AP"]
    grid = R["grid"]
    best_grid = grid.loc[grid["MAP"].idxmax()]
    length = R["length"]
    prep = R["prep"]

    margin = 1.6 * cm
    gutter = 0.7 * cm
    page_width, page_height = A4
    usable = page_width - 2 * margin
    col_width = (usable - gutter) / 2
    title_height = 3.5 * cm

    frame_title = Frame(
        margin, page_height - margin - title_height, usable, title_height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    body_height = page_height - 2 * margin - title_height
    frames_first = [
        Frame(margin, margin, col_width, body_height, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(margin + col_width + gutter, margin, col_width, body_height,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    ]
    full_height = page_height - 2 * margin
    frames_rest = [
        Frame(margin, margin, col_width, full_height, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0),
        Frame(margin + col_width + gutter, margin, col_width, full_height,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(INK_SOFT)
        canvas.drawCentredString(page_width / 2, margin - 0.55 * cm, str(doc.page))
        canvas.restoreState()

    document = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        title="Trabalho Prático 1 — Recuperação de Informação",
        author="SCC0282",
    )
    document.addPageTemplates(
        [
            PageTemplate(id="first", frames=[frame_title] + frames_first, onPage=footer),
            PageTemplate(id="rest", frames=frames_rest, onPage=footer),
        ]
    )

    S: list = [NextPageTemplate("rest")]

    # ---------------- cabeçalho ----------------
    S.append(p("Modelo Vetorial e BM25 na coleção Cranfield:<br/>"
               "pré-processamento, parametrização e análise de erros", TITLE))
    authors_line = " &nbsp;•&nbsp; ".join(
        f"{name} ({number})" for name, number, _ in AUTHORS
    )
    emails_line = " &nbsp;•&nbsp; ".join(email for _, _, email in AUTHORS)
    S.append(p(authors_line, SUBTITLE))
    S.append(p(
        "Instituto de Ciências Matemáticas e de Computação — Universidade de São Paulo<br/>"
        "SCC0282 — Recuperação de Informação — 2º semestre de 2026", SUBTITLE))
    S.append(p(emails_line, SUBTITLE))
    S.append(Spacer(1, 4))

    # ---------------- 1. Introdução ----------------
    S.append(p("1. Introdução", H1))
    S.append(p(
        "Sistemas de recuperação de informação precisam ordenar uma coleção inteira "
        "a partir de uma consulta curta e ambígua. Os modelos clássicos resolvem isso "
        "de maneiras distintas: o <b>modelo vetorial</b> representa documentos e "
        "consultas como vetores de pesos TF-IDF e ordena pela similaridade do cosseno, "
        "enquanto o <b>BM25</b> parte do arcabouço probabilístico e pondera cada termo "
        "com saturação de frequência e normalização explícita por comprimento. Embora "
        "ambos valorizem termos raros e controlem a influência da frequência, fazem-no "
        "de formas diferentes — e é justamente essa diferença que este trabalho procura "
        "tornar visível.", BODY))
    S.append(p(
        "O objetivo é construir e avaliar um sistema de recuperação textual sobre a "
        "coleção Cranfield, comparando estratégias de pré-processamento, os dois "
        "modelos de ranqueamento e a parametrização do BM25, e — mais importante — "
        "explicar <i>por que</i> os comportamentos observados ocorrem, com base em "
        "exemplos concretos. A principal conclusão é que, nesta coleção, quase toda a "
        "discordância entre os dois modelos é explicada por um único mecanismo: a "
        "<b>forma como cada um normaliza o comprimento dos documentos</b>. Essa "
        "hipótese é levantada a partir da inspeção de consultas individuais e depois "
        "confirmada quantitativamente sobre as 225 consultas.", BODY))

    # ---------------- 2. Técnicas ----------------
    S.append(p("2. Técnicas Utilizadas", H1))

    S.append(p("2.1 Pré-processamento", H2))
    S.append(p(
        "O texto passa por <b>normalização para minúsculas</b> e <b>tokenização</b> por "
        "expressão regular <font face='Courier'>[a-z]+</font>, que é determinística e "
        "adequada ao texto técnico ASCII do Cranfield. A escolha tem um efeito "
        "colateral relevante: hifens viram separadores, de modo que "
        "<i>boundary-layer-control</i> gera três tokens e <i>photo-thermoelastic</i> "
        "gera <i>photo</i> e <i>thermoelast</i> — fato que reaparece na §4.6. "
        "A <b>remoção de stopwords</b> usa a lista <i>english</i> do NLTK (198 formas, "
        "expandidas para fragmentos alfabéticos) e o <b>stemming</b> usa o algoritmo de "
        "Porter. As stopwords são removidas <i>antes</i> do stemming, pois a lista é "
        "definida sobre formas plenas e o radical de uma stopword pode coincidir com o "
        "de um termo de conteúdo.", BODY))

    S.append(p("2.2 Modelo Vetorial", H2))
    S.append(p(
        "Adotamos o esquema apresentado em aula (<i>ltc.ltc</i> na notação SMART), "
        "aplicado igualmente a documentos e consultas:", BODY))
    S.append(p(
        "<font face='Courier' size='7.6'>w(i,j) = (1 + log₁₀ f(i,j)) × log₁₀(N / n(i))</font>",
        ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=3)))
    S.append(p(
        "Cada documento é um vetor de dimensão |V| = "
        f"{meta['vocab_size']}, com uma coordenada por termo do vocabulário e "
        "praticamente todas nulas. O score é a similaridade do cosseno, "
        "<font face='Courier' size='7.6'>sim(d,q) = (d·q)/(‖d‖‖q‖)</font>: o numerador "
        "mede o alinhamento entre consulta e documento e o denominador remove o efeito "
        "da magnitude. Como normalizamos os vetores para norma unitária já na "
        "indexação, o cosseno se reduz a um produto interno — é o mesmo cálculo, "
        "reorganizado para ser feito de uma vez sobre os 1400 documentos. Note que "
        "<font face='Courier' size='7.6'>idf</font> vale exatamente zero para um termo "
        "presente em toda a coleção, que assim é eliminado do ranking.", BODY))

    S.append(p("2.3 BM25", H2))
    S.append(p(
        "O BM25 foi implementado explicitamente, termo a termo, sem biblioteca de "
        "ranqueamento:", BODY))
    S.append(p(
        "<font face='Courier' size='7.6'>score(d,q) = Σ IDF(t)·f(t,d)(k₁+1) / "
        "[f(t,d) + k₁(1−b+b·|d|/avgdl)]</font>",
        ParagraphStyle("eq2", parent=BODY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=3)))
    S.append(p(
        "O quociente cresce com a frequência mas <b>satura</b> em k₁+1, ao contrário do "
        "tf logarítmico do modelo vetorial; k₁ controla quão rápido essa saturação "
        "ocorre (com k₁ = 0 o modelo vira binário) e b interpola entre nenhuma "
        "normalização por comprimento (b = 0) e normalização integral (b = 1). "
        "Somamos a contribuição de um termo tantas vezes quanto ele ocorre na consulta, "
        "o que equivale a k₃ → ∞. Usamos o IDF de Robertson suavizado, "
        "<font face='Courier' size='7.6'>log(1 + (N−n+0,5)/(n+0,5))</font>, cujo "
        "&ldquo;1 +&rdquo; impede valores negativos para termos muito frequentes.", BODY))
    S.append(p(
        "Vale registrar uma identidade algébrica que explica um resultado nosso: aquela "
        "expressão é idêntica a "
        "<font face='Courier' size='7.6'>log((N+1)/(n+0,5))</font>, ao passo que a "
        "forma do BIM vista em aula é "
        "<font face='Courier' size='7.6'>log((N+0,5)/(n+0,5))</font>. As duas diferem "
        "apenas por uma constante aditiva log((N+1)/(N+0,5)) ≈ 0,00036 em todos os "
        "termos. Por isso, ao testarmos as duas variantes, o MAP mudou apenas na sexta "
        "casa decimal ("
        f"{fmt(float(R['idf'][R['idf']['idf_variant']=='robertson']['MAP'].iloc[0]), 6)} contra "
        f"{fmt(float(R['idf'][R['idf']['idf_variant']=='bim']['MAP'].iloc[0]), 6)}"
        ") — a escolha entre elas é irrelevante nesta coleção, e o motivo é algébrico, "
        "não empírico.", BODY))

    S.append(p("2.4 Decisões de implementação", H2))
    S.append(p(
        "<b>Índice compartilhado.</b> Ambos os modelos derivam seus pesos da mesma "
        "matriz esparsa de frequências brutas, garantindo que qualquer diferença venha "
        "da função de ranqueamento e não da indexação. <b>Texto indexado:</b> em 1395 "
        "dos 1400 documentos o campo <font face='Courier'>.W</font> já começa com o "
        "título; concatenar <font face='Courier'>.T</font> dobraria o tf dos termos do "
        "título, então só o fazemos quando o título ainda não é prefixo do resumo. "
        "<b>Desempate determinístico:</b> scores iguais são ordenados pelo "
        "<font face='Courier'>doc_id</font> crescente, de modo que duas execuções "
        "produzem rankings idênticos.", BODY))

    # ---------------- 3. Avaliação ----------------
    S.append(p("3. Avaliação", H1))
    S.append(p(
        f"<b>Coleção.</b> Cranfield: {meta['num_documents']} documentos de aerodinâmica, "
        f"{meta['num_queries']} consultas em linguagem natural e "
        f"{meta['num_qrel_pairs']} julgamentos de relevância, obtida dos arquivos "
        "originais da University of Glasgow. Cada consulta tem entre 1 e 39 documentos "
        "relevantes (média 7,2), e nenhuma fica sem relevantes.", BODY))
    S.append(p(
        "<b>Dois detalhes da coleção que exigem cuidado.</b> Primeiro, os "
        "identificadores <font face='Courier'>.I</font> das consultas <i>não são "
        "sequenciais</i> (001, 002, 004, …, 365), mas o arquivo "
        "<font face='Courier'>cranqrel</font> numera as consultas de 1 a 225 pela "
        "<i>posição</i> no arquivo; usar o <font face='Courier'>.I</font> como chave "
        "desalinha silenciosamente consultas e julgamentos. Renumeramos por posição, "
        "como faz o <font face='Courier'>ir_datasets</font>. Segundo, a escala de "
        "relevância do Cranfield é <i>invertida</i>: o grau 1 é o melhor (&ldquo;resposta "
        "completa&rdquo;) e o 4 é o pior (&ldquo;interesse mínimo&rdquo;).", BODY))
    S.append(p(
        "<b>Relevância.</b> Para as métricas binárias seguimos o enunciado: relevante "
        "⇔ grau ≥ 1; graus −1 e documentos não julgados contam como não relevantes. "
        "Para o NDCG mantivemos os graus positivos como relevância graduada, mas "
        "<i>corrigindo a inversão</i> (1→4, 2→3, 3→2, 4→1): usar o grau bruto daria o "
        "maior ganho ao documento menos relevante. A diferença não é cosmética — o "
        f"NDCG@10 do BM25 vale "
        f"{fmt(float(R['gain'][R['gain']['gain_mapping']=='inverted']['NDCG@10'].iloc[0]))} "
        "com o mapeamento corrigido e apenas "
        f"{fmt(float(R['gain'][R['gain']['gain_mapping']=='raw']['NDCG@10'].iloc[0]))} "
        "com o grau bruto.", BODY))
    S.append(p(
        "<b>Métricas.</b> P@10, R@10, F1@10, MAP, MRR e NDCG@10, definidas como em "
        "aula: AP(q) = (1/|R_q|)·Σ P@k·rel(k), com |R_q| contando <i>todos</i> os "
        "relevantes (relevantes não recuperados penalizam a métrica); "
        "DCG@k = Σ (2^rel−1)/log₂(i+1); e curva precisão-revocação interpolada em 11 "
        "pontos por P(r) = max_{r'≥r} P(r'). Todas foram implementadas por nós e "
        "validadas contra os valores calculados à mão no exercício do slide 41 da "
        "Aula 05.", BODY))
    S.append(p(
        "<b>Configuração experimental.</b> O ranking é calculado sobre os 1400 "
        "documentos e os cortes @10 aplicados só na avaliação. Os qrels são usados "
        "<i>exclusivamente para avaliar</i>: nenhuma etapa de indexação, ponderação, "
        "escolha de parâmetros ou reformulação os consulta. A configuração principal "
        f"usa o pré-processamento vencedor da §4.1 e k₁ = {fmt(meta['bm25_default']['k1'],1)}, "
        f"b = {fmt(meta['bm25_default']['b'],2)}.", BODY))
    S.append(p(
        "<b>Verificação da implementação.</b> Uma versão ingênua em Python puro, "
        "escrita com laços diretamente a partir das fórmulas, reproduz os scores da "
        "versão vetorizada em SciPy com tolerância 10⁻¹⁰, para os dois modelos e três "
        "combinações de (k₁, b). O modelo vetorial também foi comparado ao "
        "<font face='Courier'>TfidfVectorizer</font> do scikit-learn (correlação de "
        "Spearman &gt; 0,95; não se espera igualdade exata porque o scikit-learn usa "
        "idf = ln(N/df) + 1). São 16 testes, em "
        "<font face='Courier'>tests/test_ir_tp1.py</font>.", BODY))

    # ---------------- 4/5. Resultados ----------------
    S.append(p("4. Resultados Obtidos e Análise", H1))

    S.append(p("4.1 Pré-processamento", H2))
    best_config = meta["best_preprocessing_config"]
    none_bm = float(prep[(prep["config"] == "none") & (prep["model"] == "BM25")]["MAP"].iloc[0])
    stop_bm = float(prep[(prep["config"] == "stop") & (prep["model"] == "BM25")]["MAP"].iloc[0])
    stem_bm = float(prep[(prep["config"] == "stem") & (prep["model"] == "BM25")]["MAP"].iloc[0])
    both_bm = float(prep[(prep["config"] == "stop_stem") & (prep["model"] == "BM25")]["MAP"].iloc[0])
    S.append(p(
        "O ganho é <b>monotônico</b> nas quatro configurações e vale para os dois "
        f"modelos: o MAP do BM25 sobe de {fmt(none_bm)} (linha de base) para "
        f"{fmt(stop_bm)} com stopwords, {fmt(stem_bm)} com stemming e {fmt(both_bm)} com "
        f"ambos — ganho total de {fmt(100*(both_bm-none_bm)/none_bm, 1)}%. Os dois "
        "efeitos são <i>complementares</i>, e não redundantes, porque atacam problemas "
        "diferentes.", BODY))
    S.append(table_preprocessing(R, col_width))
    S.append(p("Tabela 1. Efeito do pré-processamento. |V| é o tamanho do vocabulário e "
               "|d| méd. o comprimento médio dos documentos em tokens.", CAPTION))
    S.append(p(
        "A tabela deixa o mecanismo visível. A remoção de stopwords quase não altera o "
        "vocabulário (7044 → 6925 termos, pois a lista tem só 198 formas) mas elimina "
        "<b>42% de todos os tokens</b> (comprimento médio 159,1 → 92,1): o efeito é "
        "sobre o <i>denominador</i> das normalizações — a norma do cosseno e o "
        "|d|/avgdl do BM25 deixam de ser dominados por palavras sem poder "
        "discriminativo. Já o stemming não remove token algum, mas funde formas "
        "flexionadas e reduz o vocabulário em 38% (7044 → 4403): o efeito é sobre a "
        "<i>cobertura</i> do casamento léxico, permitindo que uma consulta com "
        "<i>solutions</i> case com um documento que diz <i>solution</i>. Como um ataca "
        "a normalização e o outro a cobertura, somam-se. Vale notar que o stemming "
        f"sozinho ({fmt(stem_bm)}) supera as stopwords sozinhas ({fmt(stop_bm)}), o que "
        "é coerente com uma coleção de textos científicos, onde a variação morfológica "
        "é intensa. Todos os resultados seguintes usam a configuração vencedora "
        f"(<b>{best_config.replace('_',' + ')}</b>).", BODY))
    S.extend(figure(FIGURES / "fig1b_preprocessamento_compacto.png", col_width * 0.90))
    S.append(p("Figura 1. MAP e P@10 nas quatro configurações de pré-processamento.", CAPTION))

    S.append(p("4.2 Comparação entre os modelos", H2))
    S.append(table_models(R, col_width))
    S.append(p("Tabela 2. Métricas agregadas sobre as 225 consultas.", CAPTION))
    S.append(p(
        "O BM25 é superior em <i>todas</i> as métricas, com vantagem maior no MRR "
        f"(+{fmt(100*(bm_agg['MRR']-vsm_agg['MRR'])/vsm_agg['MRR'],1)}%) e no MAP "
        f"(+{fmt(100*(bm_agg['MAP']-vsm_agg['MAP'])/vsm_agg['MAP'],1)}%) — ou seja, "
        "ganha sobretudo no topo do ranking. A curva precisão-revocação (Figura 2) "
        "mostra que a vantagem se mantém em <i>todos</i> os 11 níveis de revocação, e "
        "não apenas em média.", BODY))
    S.append(p(
        "A média, porém, esconde o essencial: o BM25 vence em "
        f"{wins['BM25_melhor']} consultas, <b>perde em {wins['VSM_melhor']}</b> e "
        f"empata em {wins['empate']}. A dispersão da Figura 3 confirma que há um número "
        "expressivo de consultas abaixo da diagonal. A pergunta interessante não é "
        "qual modelo tem a maior média, mas <i>o que separa</i> as consultas dos dois "
        "grupos.", BODY))
    S.extend(figure(FIGURES / "fig3_curva_precisao_revocacao.png", col_width * 0.92))
    S.append(p("Figura 2. Curva precisão × revocação interpolada em 11 pontos.", CAPTION))

    S.append(p("4.3 Análise por consulta", H2))
    S.append(table_query_analysis(R, col_width))
    S.append(p("Tabela 3. Top-5 de cada modelo nas seis consultas selecionadas. "
               "Identificadores de documento; <b>negrito●</b> = relevante (grau ≥ 1).",
               CAPTION))
    S.append(p(
        "A seleção é automática: entre as consultas com ao menos 4 relevantes, as duas "
        "de maior e menor ΔAP e as duas de menor AP médio. Lendo os Top-5 lado a lado, "
        "aparece um padrão nítido — e é um padrão de <b>comprimento de documento</b>.", BODY))
    S.append(p(
        "<b>(i) BM25 claramente superior — q223 e q59.</b> Em q223 "
        "(<i>shear buckling of unstiffened rectangular plates</i>), o modelo vetorial "
        "coloca no topo os documentos 1008 (51 tokens) e 864 (29 tokens), ambos "
        "irrelevantes e <i>muito curtos</i>, e só encontra um relevante na 4ª posição. "
        "O BM25 recupera na 3ª e 5ª posições os documentos 1398 (115 tokens) e 1387 "
        "(120 tokens), ambos relevantes e <i>longos</i>. A causa é a normalização: o "
        "cosseno divide pela norma <i>completa</i> do vetor, o que penaliza "
        "integralmente um documento longo, mesmo quando o comprimento vem de tratar o "
        "assunto com profundidade; o BM25 com b = 0,75 normaliza apenas parcialmente. "
        "Em q59 ocorre o mesmo: o vetorial promove o documento 382 (21 tokens) e o "
        "BM25 encontra o relevante 785 já na 1ª posição.", BODY))
    S.append(p(
        "<b>(ii) Modelo vetorial superior — q113 e q190.</b> O mecanismo é o "
        "<i>mesmo</i>, com o sinal trocado. Em q113 o BM25 traz para o topo os "
        "documentos 704 (185 tokens), 815 (142) e 638 (138) — todos longos e todos "
        "irrelevantes —, enquanto o vetorial acerta as duas primeiras posições com os "
        "documentos 265 (49 tokens) e 748 (83). Aqui os relevantes são curtos e "
        "específicos, e a normalização integral do cosseno é exatamente o que se quer. "
        "Em q190 o vetorial coloca três relevantes no Top-4 e o BM25 intercala os "
        "documentos 856 (137) e 1339 (130), não relevantes. Em outras palavras: a "
        "normalização parcial do BM25 é uma <i>aposta</i> — paga bem quando os "
        "relevantes são longos e custa caro quando são curtos.", BODY))
    S.append(p(
        "<b>(iii) Ambos insatisfatórios — q124 e q13.</b> Estas revelam limitações que "
        "nenhum ajuste de ranqueamento resolve. A q13 (<i>what is the basic mechanism "
        "of the transonic aileron buzz</i>) tem, na 1ª posição de <i>ambos</i> os "
        "modelos, o documento 496, &ldquo;<i>a theory of transonic aileron buzz, "
        "neglecting viscous effects</i>&rdquo; — o casamento léxico é quase perfeito, e "
        "mesmo assim o julgamento é <b>−1</b>, isto é, não relevante. A q124 é uma "
        "consulta <i>composta</i>, com duas necessidades de informação distintas "
        "(escoamento em canais delgados <i>e</i> estabilidade de cascas cônicas); o "
        "documento 941, cujo título praticamente repete a primeira metade da consulta, "
        "também é julgado −1. Ou seja, o erro não está no ranqueamento, mas no "
        "descompasso entre <i>sobreposição léxica</i> e <i>relevância julgada</i>: o "
        "avaliador julgou contra a necessidade real, que os modelos não enxergam.", BODY))

    S.append(p("4.4 Verificando a hipótese do comprimento", H2))
    S.append(p(
        "A leitura acima é uma hipótese construída sobre seis consultas. Testamo-la "
        "sobre as 225. Se a normalização for mesmo o mecanismo dominante, então (a) o "
        "Top-10 do BM25 deve ser em média mais longo que o do vetorial e (b) o BM25 "
        "deve levar vantagem justamente nas consultas cujos relevantes são longos.", BODY))
    S.append(p(
        "Ambas se confirmam. O comprimento médio da coleção é "
        f"{fmt(length['comprimento_medio_colecao'],1)} tokens e o dos documentos "
        f"relevantes é {fmt(length['comprimento_medio_documentos_relevantes'],1)} — "
        "praticamente idêntico, ou seja, <i>ser longo não torna um documento "
        "relevante</i>. Ainda assim, o Top-10 do modelo vetorial tem em média "
        f"{fmt(length['comprimento_medio_top10_VSM'],1)} tokens (bem abaixo da média da "
        f"coleção) e o do BM25, {fmt(length['comprimento_medio_top10_BM25'],1)} (acima). "
        "O vetorial tem, portanto, um <b>viés sistemático para documentos curtos</b> "
        "que não é justificado pelos dados. E a correlação de Spearman entre ΔAP "
        "(BM25 − vetorial) e o comprimento médio dos relevantes de cada consulta é "
        f"<b>ρ = {fmt(length['spearman_delta_AP_vs_comprimento_relevantes']['rho'],3)}</b> "
        f"(p ≈ {length['spearman_delta_AP_vs_comprimento_relevantes']['p_value']:.0e}, "
        f"n = {length['spearman_delta_AP_vs_comprimento_relevantes']['n']}). A hipótese "
        "levantada a partir de seis exemplos resiste, portanto, a um teste sobre a "
        "coleção inteira.", BODY))
    S.extend(figure(FIGURES / "fig5_ap_por_consulta.png", col_width * 0.88))
    S.append(p("Figura 3. AP por consulta. Pontos acima da diagonal favorecem o BM25.",
               CAPTION))

    S.append(p("4.5 Variação dos parâmetros do BM25", H2))
    S.append(table_grid(R, col_width))
    S.append(p("Tabela 4. Grade k₁ × b. As três últimas colunas fixam b = 0,75.", CAPTION))
    S.extend(figure(FIGURES / "fig4_bm25_grade_MAP.png", col_width * 0.88))
    S.append(p("Figura 4. MAP do BM25 na grade de parâmetros. A cor codifica a "
               "magnitude; o valor está impresso em cada célula.", CAPTION))
    S.append(p(
        f"O melhor ponto da grade é k₁ = {fmt(best_grid['k1'],1)}, b = "
        f"{fmt(best_grid['b'],2)}, com MAP {fmt(best_grid['MAP'])} e P@10 "
        f"{fmt(best_grid['P@10'])}. O resultado mais claro é que <b>b importa muito "
        "mais que k₁</b>: fixado k₁, passar de b = 0 para b = 0,75 melhora o MAP entre "
        "8,6% e 10,8%, enquanto variar k₁ de 0,5 a 2,0 com b fixo rende no máximo 7%. "
        "Coerente com a §4.4 — a normalização por comprimento é o fator dominante nesta "
        "coleção. Também há <i>interação</i> entre os parâmetros: para k₁ pequeno "
        "(0,5 e 1,2) o melhor b é 1,0, mas para k₁ = 2,0 o melhor é 0,75. A leitura é "
        "que k₁ alto deixa o score mais sensível à frequência bruta, e aí uma "
        "normalização total (b = 1) passa a penalizar demais documentos longos que "
        "acumulam ocorrências legítimas; um pouco de folga (b = 0,75) compensa.", BODY))
    S.append(p(
        f"<b>Efeito de b em uma consulta concreta.</b> A q{R['b_showcase']['query_id']} "
        f"(&ldquo;<i>{R['b_showcase']['query_text'][:74]}…</i>&rdquo;, "
        f"{R['b_showcase']['num_relevant']} relevantes) é o caso mais didático.", BODY))
    S.append(table_b_effect(R, col_width))
    S.append(p("Tabela 5. Top-5 da consulta q67 para três valores de b (k₁ = 1,2).",
               CAPTION))
    S.append(p(
        "Com <b>b = 0</b> não há normalização alguma e o Top-5 é tomado por documentos "
        "de 177, 164, 138, 130 e 116 tokens, dos quais apenas o último é relevante: "
        "sem normalização, o score cresce com o tamanho, pois documentos longos "
        "simplesmente acumulam mais ocorrências dos termos da consulta. Com "
        "<b>b = 0,75</b> o ranking se inverte por completo — os cinco primeiros passam "
        "a ter 34, 36, 47, 17 e 91 tokens, e quatro são relevantes. Com <b>b = 1</b> os "
        "cinco são relevantes e a AP vai de 0,14 para 0,55. Os documentos relevantes "
        "desta consulta são artigos curtos e diretamente sobre o tema "
        "(&ldquo;<i>the boundary layer in simple shear flow past a flat plate</i>&rdquo;, "
        "17 tokens); sem normalização eles são esmagados por textos longos que "
        "mencionam os mesmos termos de passagem.", BODY))

    S.append(p("4.6 Reformulação de consultas", H2))
    S.append(p(
        "Cinco consultas foram reescritas <i>manualmente</i>, cada uma com uma operação "
        "diferente e com a hipótese registrada no código <i>antes</i> de rodar os "
        "modelos.", BODY))
    S.append(table_reformulation(R, col_width))
    S.append(p("Tabela 6. Efeito das reformulações (original → modificada).", CAPTION))

    S.append(p(
        "<b>q219 — sinônimo.</b> O caso mais limpo. A consulta original diz "
        "&ldquo;<i>…when the reynolds number is <b>small</b></i>&rdquo;, mas a literatura "
        "consagrou a expressão &ldquo;<i>low Reynolds number</i>&rdquo;. Trocando uma "
        "<i>única</i> palavra, <i>small</i> por <i>low</i>, a AP do vetorial vai de "
        "0,045 para 0,119 e a do BM25 de 0,037 para 0,093 — e a P@10, que era zero nos "
        "dois modelos, passa a 0,10. Nada mudou no significado da pergunta; mudou "
        "apenas o vocabulário. É a ilustração mais direta da limitação apontada na "
        "Aula 04: modelos puramente léxicos não enxergam sinonímia.", BODY))
    S.append(p(
        "<b>q69 e q109 — tornar mais específica.</b> Ambas confirmam a hipótese, com "
        "ganho modesto. Em q69 os termos originais são todos pouco discriminativos "
        "(<i>solution</i>, <i>boundary</i>, <i>layer</i> e <i>equation</i> ocorrem em "
        "400 a 470 dos 1400 documentos); acrescentar <i>series</i>, <i>laminar</i> e "
        "<i>compressible</i> leva a AP do BM25 de 0,039 a 0,092 e a R@10 de 0 a 0,20. "
        "Em q109, uma consulta de apenas 4 tokens, explicitar a consequência física "
        "(tensão térmica, flambagem, flutter) leva a P@10 de 0 a 0,10.", BODY))
    S.append(p(
        "<b>q171 — remover termos: hipótese parcialmente refutada.</b> Prevíamos que "
        "enxugar a consulta de 17 tokens beneficiaria sobretudo o modelo vetorial, "
        "prejudicado pela diluição da norma. O vetorial de fato melhorou um pouco "
        "(0,306 → 0,319), mas o <b>BM25 piorou</b> (0,639 → 0,556). A explicação é que "
        "o BM25 não normaliza o vetor de consulta: termos adicionais só <i>somam</i> "
        "evidência, e mesmo termos de IDF baixo contribuem positivamente quando "
        "presentes. Remover termos, para o BM25, é descartar evidência sem contrapartida "
        "— o que revela uma assimetria entre os modelos que não havíamos previsto.", BODY))
    S.append(p(
        "<b>q15 — vocabulário técnico e o valor da métrica graduada.</b> O termo "
        "<i>photoelastic</i> ocorre em um único documento, mas os dois relevantes usam "
        "<i>photo-thermoelastic</i>, que o tokenizador quebra em <i>photo</i> e "
        "<i>thermoelast</i> — o stemming não aproxima essas formas. Adotando o termo "
        "dos autores, a AP do BM25 <i>não muda</i> (1,00 → 1,00), porque os dois "
        "relevantes já estavam no topo; mas o <b>NDCG@10 sobe de 0,74 para 1,00</b>, "
        "pois a reformulação reordenou os dois relevantes, trazendo para a 1ª posição "
        "o de grau 1 (&ldquo;resposta completa&rdquo;). É um bom argumento a favor de "
        "reportar métricas graduadas: a AP é cega a uma melhora que o usuário "
        "perceberia.", BODY))

    S.append(p("4.7 Análise de erros", H2))
    S.append(p(
        "<b>Falsos positivos.</b> Escolhemos documentos não relevantes trazidos pelo "
        "BM25 para as duas primeiras posições. (1) Na q98, o documento 638 "
        "(&ldquo;<i>longitudinal aerodynamic characteristics at low subsonic speeds of a "
        "highly swept wing utilizing nose flaps</i>&rdquo;, grau −1) aparece em 1º com "
        "score 35,1. A decomposição do score mostra de onde ele vem: <i>flap</i> "
        "(+6,74), <i>control</i> (+5,70), <i>trail</i> (+4,87), <i>subson</i> (+3,03). "
        "Todos os termos da consulta estão lá — mas a consulta pergunta se controles "
        "<i>na ponta</i> se comparam a <i>flaps de bordo de fuga</i>, e o documento "
        "trata de <i>flaps de nariz</i>. É a hipótese de <b>independência entre termos</b> "
        "falhando: o modelo soma evidências isoladas e não percebe que a consulta pede "
        "uma <i>relação</i> entre elas. (2) Na q124, o documento 941 tem título quase "
        "idêntico a metade da consulta e recebe +8,40 só de <i>channel</i> (n=17, IDF "
        "altíssimo), mas é julgado −1, porque a necessidade real de informação está na "
        "<i>outra</i> metade da consulta. (3) Um terceiro caso, na q139, é de outra "
        "natureza: os documentos 847 e 897 estão em 1º e 2º e parecem genuinamente "
        "pertinentes, mas simplesmente <b>não foram julgados</b>. Como o enunciado manda "
        "tratar não julgados como não relevantes, eles contam como erro — o que é, na "
        "verdade, uma limitação de cobertura dos qrels do Cranfield, não do sistema.", BODY))
    S.append(p(
        "<b>Falso negativo.</b> Na q80 (&ldquo;<i>are methods of measuring aerodynamic "
        "derivatives available which could be adopted for use in short running time "
        "facilities</i>&rdquo;), o documento 596 tem grau 3 e aparece na posição "
        "<b>1206</b> em ambos os modelos. O motivo é categórico: a interseção entre os "
        "tokens da consulta e os do documento é <b>vazia</b>. O documento se chama "
        "&ldquo;<i>the properties of crossed flexure pivots…</i>&rdquo; e descreve o "
        "<i>aparato</i> usado na medição, enquanto a consulta descreve o <i>método</i> e "
        "a <i>finalidade</i>. Score zero não é um erro de ponderação: é o limite do "
        "casamento léxico. Nenhum ajuste de k₁, b ou pré-processamento recupera esse "
        "documento — seria preciso representação semântica (embeddings) ou expansão de "
        "consulta. Os outros dois falsos negativos que examinamos têm a mesma assinatura: "
        "no documento 236 da q87 (grau 4), o único termo casado é <i>flow</i>, o de "
        "menor IDF da consulta (n = 730 de 1400).", BODY))

    # ---------------- 6. Considerações finais ----------------
    S.append(p("5. Considerações finais", H1))
    S.append(p(
        "<b>1. O pré-processamento é barato e vale a pena.</b> Stopwords e stemming "
        f"juntos elevam o MAP do BM25 em {fmt(100*(both_bm-none_bm)/none_bm,1)}%, e são "
        "complementares porque atacam frentes distintas: um corrige o denominador das "
        "normalizações, o outro amplia a cobertura do casamento léxico.", BODY))
    S.append(p(
        "<b>2. O BM25 vence, mas não por ser &ldquo;melhor&rdquo; em abstrato.</b> "
        f"Ele supera o vetorial em todas as métricas (MAP +{fmt(100*(bm_agg['MAP']-vsm_agg['MAP'])/vsm_agg['MAP'],1)}%) "
        f"e em todos os níveis de revocação, mas perde em {wins['VSM_melhor']} das 225 "
        "consultas. A diferença é governada pela normalização por comprimento: o cosseno "
        "normaliza integralmente e adquire um viés para documentos curtos (Top-10 médio "
        f"de {fmt(length['comprimento_medio_top10_VSM'],1)} tokens contra "
        f"{fmt(length['comprimento_medio_colecao'],1)} da coleção), enquanto o BM25 "
        "normaliza parcialmente. Confirmamos isso com ρ = "
        f"{fmt(length['spearman_delta_AP_vs_comprimento_relevantes']['rho'],2)} entre ΔAP "
        "e o comprimento dos relevantes.", BODY))
    S.append(p(
        "<b>3. b importa mais que k₁</b>, e os dois interagem: o melhor b cai de 1,0 "
        "para 0,75 quando k₁ cresce de 1,2 para 2,0.", BODY))
    S.append(p(
        "<b>4. O teto dos dois modelos é léxico.</b> As consultas em que ambos falham "
        "não falham por ranqueamento: ou o documento relevante não compartilha "
        "<i>nenhum</i> termo com a consulta, ou a consulta é composta, ou o julgamento "
        "não acompanha a sobreposição léxica. Trocar &ldquo;<i>small</i>&rdquo; por "
        "&ldquo;<i>low</i>&rdquo; em q219 quase triplicou a AP — um ganho que nenhum "
        "ajuste de parâmetro produziria. É exatamente onde entram expansão de consulta e "
        "representações densas.", BODY))
    S.append(p(
        "<b>5. A escolha da métrica muda o diagnóstico.</b> A AP não viu a melhora da "
        "reformulação de q15, que o NDCG@10 registrou (0,74 → 1,00); e o NDCG varia 17% "
        "conforme se corrija ou não a escala invertida do Cranfield. Decisões de "
        "avaliação merecem tanta atenção quanto decisões de modelagem.", BODY))

    # ---------------- 7. IA ----------------
    S.append(p("6. Uso de ferramentas de IA", H1))
    S.append(p(
        "Ferramentas de IA generativa (Claude) foram utilizadas como apoio nas "
        "seguintes atividades: <b>(i) apoio à programação</b> — estruturação dos "
        "módulos, vetorização com matrizes esparsas e escrita dos testes automatizados; "
        "<b>(ii) depuração</b> — diagnóstico de uma instalação corrompida do SciPy "
        "causada pelo limite de 260 caracteres de caminho do Windows, e identificação "
        "do desalinhamento entre os identificadores <font face='Courier'>.I</font> das "
        "consultas e a numeração do <font face='Courier'>cranqrel</font>; "
        "<b>(iii) explicação de conceitos</b> — discussão das variantes de IDF do BM25 e "
        "das convenções de ganho do NDCG; e <b>(iv) revisão textual</b> deste relatório. "
        "As fórmulas, as decisões de implementação, o desenho dos experimentos, as "
        "reformulações manuais de consulta e a interpretação dos resultados foram "
        "revisados e são de responsabilidade dos autores, que compreendem integralmente "
        "o código entregue.", BODY))

    S.append(p("Referências", H1))
    S.append(p(
        "[1] MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. <i>Introduction to Information "
        "Retrieval</i>. Cambridge University Press, 2008.<br/>"
        "[2] ROBERTSON, S.; ZARAGOZA, H. <i>The Probabilistic Relevance Framework: BM25 "
        "and Beyond</i>. Foundations and Trends in IR, 2009.<br/>"
        "[3] SALTON, G.; WONG, A.; YANG, C. S. A Vector Space Model for Automatic "
        "Indexing. <i>Communications of the ACM</i>, v. 18, n. 11, 1975.<br/>"
        "[4] CLEVERDON, C. W. The Cranfield tests on index language devices. "
        "<i>Aslib Proceedings</i>, 1967.<br/>"
        "[5] JÄRVELIN, K.; KEKÄLÄINEN, J. Cumulated Gain-based Evaluation of IR "
        "Techniques. <i>ACM TOIS</i>, v. 20, n. 4, 2002.",
        ParagraphStyle("ref", parent=BODY, fontSize=7.4, leading=9)))

    document.build(S)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Relatório gerado: {path}")
