"""Geração dos gráficos do relatório.

Paleta
------
Usamos a paleta categórica de referência sem alterações: slot 1 (azul
``#2a78d6``) para o Modelo Vetorial e slot 2 (laranja ``#eb6834``) para o
BM25. São slots adjacentes, cuja separação para daltonismo já é validada.
A cor identifica sempre a MESMA entidade em todas as figuras, e nunca a
posição/ranking. Os mapas de calor usam uma rampa sequencial de um único
matiz (azul, claro -> escuro), pois codificam magnitude contínua.

Como o destino é um PDF impresso, geramos apenas a versão clara. Todos os
gráficos com duas séries trazem legenda E rótulos diretos, de modo que a
identidade nunca dependa exclusivamente da cor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- tokens de cor -----------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e6e5e1"

SERIES = {"VSM": "#2a78d6", "BM25": "#eb6834"}

#: Rampa sequencial de um matiz (azul), para magnitude contínua.
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"],
)

#: Nomes legíveis das configurações de pré-processamento.
CONFIG_LABELS = {
    "none": "sem stopwords\nsem stemming",
    "stop": "stopwords",
    "stem": "stemming",
    "stop_stem": "stopwords\n+ stemming",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
        }
    )


def _recede_axes(ax) -> None:
    """Deixa eixos e grade recessivos: só a grade horizontal, sem molduras."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def _label_bars(ax, bars, fmt: str = "{:.3f}", fontsize: int = 7) -> None:
    """Rótulo direto em cada barra, em tinta de texto (nunca na cor da série)."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=TEXT_SECONDARY,
        )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
# Figura 1 -- efeito do pré-processamento
# ======================================================================
def plot_preprocessing_compact(summary, figures_dir: Path) -> Path:
    """Versão de painel único (só MAP) para caber legível em uma coluna do PDF."""
    _apply_style()
    plt.rcParams.update({"font.size": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    configs = ["none", "stop", "stem", "stop_stem"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))

    x = np.arange(len(configs))
    width = 0.36
    for offset, model in zip((-width / 2, width / 2), ("VSM", "BM25")):
        values = [
            float(summary[(summary["config"] == c) & (summary["model"] == model)]["MAP"].iloc[0])
            for c in configs
        ]
        bars = ax.bar(
            x + offset,
            values,
            width * 0.94,
            label=model,
            color=SERIES[model],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        _label_bars(ax, bars, fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=9)
    ax.set_ylabel("MAP")
    ax.set_title("MAP por pré-processamento")
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.1) * 1.18)
    _recede_axes(ax)
    ax.legend(loc="upper left", ncols=2, fontsize=9.5)
    fig.tight_layout()
    path = _save(fig, figures_dir / "fig1b_preprocessamento_compacto.png")
    _apply_style()
    return path


def plot_preprocessing(summary, figures_dir: Path) -> Path:
    """Barras agrupadas: MAP e P@10 por configuração de pré-processamento."""
    _apply_style()
    configs = ["none", "stop", "stem", "stop_stem"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    for ax, metric in zip(axes, ["MAP", "P@10"]):
        x = np.arange(len(configs))
        width = 0.36
        for offset, model in zip((-width / 2, width / 2), ("VSM", "BM25")):
            values = [
                float(
                    summary[(summary["config"] == c) & (summary["model"] == model)][
                        metric
                    ].iloc[0]
                )
                for c in configs
            ]
            bars = ax.bar(
                x + offset,
                values,
                width * 0.94,  # folga de ~2px entre barras vizinhas
                label=model,
                color=SERIES[model],
                edgecolor=SURFACE,
                linewidth=0.8,
            )
            _label_bars(ax, bars)

        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=7.5)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} por pré-processamento")
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.1) * 1.16)
        _recede_axes(ax)

    axes[0].legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return _save(fig, figures_dir / "fig1_preprocessamento.png")


# ======================================================================
# Figura 2 -- comparação entre modelos
# ======================================================================
def plot_model_comparison(summary, figures_dir: Path) -> Path:
    """Barras agrupadas com todas as métricas agregadas dos dois modelos."""
    _apply_style()
    metrics = ["P@10", "R@10", "F1@10", "MAP", "MRR", "NDCG@10"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    x = np.arange(len(metrics))
    width = 0.36
    for offset, model in zip((-width / 2, width / 2), ("VSM", "BM25")):
        row = summary[summary["model"] == model].iloc[0]
        values = [float(row[m]) for m in metrics]
        bars = ax.bar(
            x + offset,
            values,
            width * 0.94,
            label=model,
            color=SERIES[model],
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        _label_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("valor médio sobre as 225 consultas")
    ax.set_title("Modelo Vetorial vs. BM25 (k1=1,2  b=0,75)")
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.1) * 1.16)
    _recede_axes(ax)
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    return _save(fig, figures_dir / "fig2_comparacao_modelos.png")


# ======================================================================
# Figura 3 -- curva precisão x revocação em 11 pontos
# ======================================================================
def plot_precision_recall(curves, figures_dir: Path) -> Path:
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    levels = [i / 10 for i in range(11)]

    for model in ("VSM", "BM25"):
        ax.plot(
            levels,
            curves[model],
            label=model,
            color=SERIES[model],
            marker="o",
            markersize=3.5,
        )
    # Rótulo direto no início de cada curva, além da legenda.
    for model, dy in (("VSM", -16), ("BM25", 7)):
        ax.annotate(
            model,
            (levels[3], curves[model][3]),
            textcoords="offset points",
            xytext=(4, dy),
            fontsize=8,
            color=TEXT_SECONDARY,
            fontweight="bold",
        )

    ax.set_xlabel("revocação")
    ax.set_ylabel("precisão interpolada")
    ax.set_title("Curva precisão x revocação (11 pontos)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, None)
    _recede_axes(ax)
    ax.grid(axis="x", visible=True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, figures_dir / "fig3_curva_precisao_revocacao.png")


# ======================================================================
# Figura 4 -- grade de parâmetros do BM25
# ======================================================================
def plot_bm25_grid(grid, figures_dir: Path, metric: str = "MAP") -> Path:
    """Mapa de calor ``k1 x b`` com rampa sequencial de um único matiz."""
    _apply_style()
    k1_values = sorted(grid["k1"].unique())
    b_values = sorted(grid["b"].unique())

    matrix = np.array(
        [
            [
                float(grid[(grid["k1"] == k1) & (grid["b"] == b)][metric].iloc[0])
                for b in b_values
            ]
            for k1 in k1_values
        ]
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    image = ax.imshow(matrix, cmap=BLUE_RAMP, aspect="auto")

    ax.set_xticks(range(len(b_values)), [f"{b:g}" for b in b_values])
    ax.set_yticks(range(len(k1_values)), [f"{k:g}" for k in k1_values])
    ax.set_xlabel("b  (normalização por comprimento)")
    ax.set_ylabel("k1  (saturação da frequência)")
    ax.set_title(f"{metric} do BM25 na grade de parâmetros")
    ax.grid(visible=False)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Valor em cada célula: a cor codifica a magnitude, o número a lê com
    # precisão (e garante legibilidade independente da cor).
    threshold = matrix.min() + 0.62 * (matrix.max() - matrix.min())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#ffffff" if matrix[i, j] > threshold else TEXT_PRIMARY,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=7)
    fig.tight_layout()
    return _save(fig, figures_dir / f"fig4_bm25_grade_{metric.replace('@','')}.png")


# ======================================================================
# Figura 5 -- dispersão da AP por consulta
# ======================================================================
def plot_per_query_scatter(comparison, figures_dir: Path) -> Path:
    """AP do BM25 contra AP do vetorial, uma marca por consulta."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.6))

    ax.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(
        comparison["AP_VSM"],
        comparison["AP_BM25"],
        s=22,
        color=SERIES["BM25"],
        edgecolor=SURFACE,
        linewidth=0.8,
        alpha=0.85,
        zorder=2,
    )

    ax.annotate(
        "BM25 melhor",
        (0.06, 0.92),
        fontsize=8,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.annotate(
        "vetorial melhor",
        (0.52, 0.06),
        fontsize=8,
        color=TEXT_SECONDARY,
        fontweight="bold",
    )
    ax.set_xlabel("AP -- Modelo Vetorial")
    ax.set_ylabel("AP -- BM25")
    ax.set_title("Average Precision por consulta")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    _recede_axes(ax)
    ax.grid(axis="x", visible=True)
    fig.tight_layout()
    return _save(fig, figures_dir / "fig5_ap_por_consulta.png")


# ======================================================================
# Figura 6 -- reformulação de consultas
# ======================================================================
def plot_reformulation(frame, figures_dir: Path) -> Path:
    """Um painel por modelo; em cada um, AP antes e depois da reformulação."""
    _apply_style()
    query_ids = sorted(frame["query_id"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)

    for ax, model in zip(axes, ("VSM", "BM25")):
        x = np.arange(len(query_ids))
        width = 0.36
        for offset, version, color in (
            (-width / 2, "original", TEXT_MUTED),
            (width / 2, "modificada", SERIES[model]),
        ):
            values = [
                float(
                    frame[
                        (frame["query_id"] == q)
                        & (frame["model"] == model)
                        & (frame["version"] == version)
                    ]["AP"].iloc[0]
                )
                for q in query_ids
            ]
            bars = ax.bar(
                x + offset,
                values,
                width * 0.94,
                label=version,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.8,
            )
            _label_bars(ax, bars, fmt="{:.2f}")

        ax.set_xticks(x, [f"q{q}" for q in query_ids])
        ax.set_title(model)
        _recede_axes(ax)
        ax.legend(loc="upper left", ncols=2)

    axes[0].set_ylabel("Average Precision")
    axes[0].set_ylim(0, max(axes[0].get_ylim()[1], 0.1) * 1.2)
    fig.suptitle(
        "Efeito da reformulação manual das consultas",
        fontsize=10,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    return _save(fig, figures_dir / "fig6_reformulacao.png")


# ======================================================================
def generate_all(results: dict, figures_dir: Path) -> list[Path]:
    figures_dir = Path(figures_dir)
    paths = [
        plot_preprocessing(results["preprocessing"]["summary"], figures_dir),
        plot_preprocessing_compact(results["preprocessing"]["summary"], figures_dir),
        plot_model_comparison(results["models"]["summary"], figures_dir),
        plot_precision_recall(results["models"]["curves"], figures_dir),
        plot_bm25_grid(results["parameters"]["grid"], figures_dir, "MAP"),
        plot_bm25_grid(results["parameters"]["grid"], figures_dir, "P@10"),
        plot_per_query_scatter(results["models"]["comparison"], figures_dir),
        plot_reformulation(results["reformulation"]["frame"], figures_dir),
    ]
    return paths
