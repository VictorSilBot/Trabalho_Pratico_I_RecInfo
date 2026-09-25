"""Execução de todos os experimentos exigidos pelo enunciado (requisitos 1-9).

Cada função 'experiment_*' corresponde a um requisito, grava seus artefatos
em 'results/' e devolve um dicionário com o que o relatório precisa.

Convenções globais:
- Todos os experimentos usam a MESMA coleção, as mesmas 225 consultas e o
  mesmo qrel.
- O ranking é calculado sobre os 1400 documentos; os cortes (@10) são
  aplicados apenas no cálculo das métricas.
- Os qrels são usados exclusivamente para AVALIAR. Nenhuma etapa de
  indexação, ponderação ou reformulação consulta os julgamentos.
- A configuração principal ("main") usa o pré-processamento vencedor do
  requisito 1, escolhido por MAP do BM25, e os parâmetros padrão
  'k1 = 1.2', 'b = 0.75'.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .bm25 import BM25, IDFVariant
from .dataset import Cranfield
from .indexing import InvertedIndex, build_index
from .metrics import (
    GainMapping,
    aggregate,
    average_precision,
    binary_relevant,
    evaluate_run,
    mean_interpolated_curve,
    ndcg_at_k,
)
from .preprocess import CONFIG_ORDER, build_configs
from .query_reformulations import REFORMULATIONS
from .vsm import VectorSpaceModel

#: Corte usado em todas as métricas @k.
K = 10

#: Parâmetros padrão do BM25 (valores clássicos de Robertson).
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

#: Grade exigida no requisito 7.
K1_GRID = [0.5, 1.2, 2.0]
B_GRID = [0.0, 0.75, 1.0]

#: Nas análises por consulta, só consideramos consultas com pelo menos este
#: número de documentos relevantes, para que o Top-5 seja informativo.
MIN_RELEVANT_FOR_ANALYSIS = 4


# ======================================================================
# Utilitários
# ======================================================================
def run_model(model, queries) -> dict[int, list[int]]:
    """Executa um modelo sobre todas as consultas, devolvendo os rankings."""
    return {q.query_id: [doc_id for doc_id, _ in model.rank(q.text)] for q in queries}


def run_model_with_scores(model, queries) -> dict[int, list[tuple[int, float]]]:
    return {q.query_id: model.rank(q.text, top_k=50) for q in queries}


def build_models(
    collection: Cranfield, config_name: str, data_dir: Path
) -> tuple[InvertedIndex, VectorSpaceModel, BM25]:
    """Constrói índice, modelo vetorial e BM25 para um pré-processamento."""
    config = build_configs(Path(data_dir) / "stopwords_en.txt")[config_name]
    index = build_index(
        [d.text for d in collection.documents], collection.doc_ids, config
    )
    return index, VectorSpaceModel(index), BM25(index, k1=DEFAULT_K1, b=DEFAULT_B)


def _relevance_flags(
    ranked: Sequence[int], qrels_for_query: dict[int, int], n: int
) -> list[dict]:
    """Anota os 'n' primeiros documentos com seu julgamento de relevância."""
    relevant = binary_relevant(qrels_for_query)
    out = []
    for position, doc_id in enumerate(ranked[:n], start=1):
        grade = qrels_for_query.get(doc_id)
        out.append(
            {
                "position": position,
                "doc_id": doc_id,
                "grade": grade if grade is not None else "não julgado",
                "relevant": doc_id in relevant,
            }
        )
    return out


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")


def _write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


# ======================================================================
# Requisito 1: Pré-processamento
# ======================================================================
def experiment_preprocessing(collection: Cranfield, data_dir: Path, results_dir: Path):
    """Compara as quatro configurações de pré-processamento nos dois modelos."""
    rows = []
    per_query_rows = []
    vocab_stats = []

    for config_name in CONFIG_ORDER:
        index, vsm, bm25 = build_models(collection, config_name, data_dir)
        vocab_stats.append(
            {
                "config": config_name,
                "vocab_size": index.vocab_size,
                "avg_doc_length": index.avg_doc_length,
                "total_tokens": float(index.doc_lengths.sum()),
            }
        )

        for model_name, model in [("VSM", vsm), ("BM25", bm25)]:
            run = run_model(model, collection.queries)
            per_query, agg = evaluate_run(run, collection.qrels, K)
            rows.append({"config": config_name, "model": model_name, **agg})
            for query_id, metrics in per_query.items():
                per_query_rows.append(
                    {
                        "config": config_name,
                        "model": model_name,
                        "query_id": query_id,
                        **metrics,
                    }
                )

    summary = pd.DataFrame(rows)
    _write_csv(summary, results_dir / "01_preprocessing_summary.csv")
    _write_csv(
        pd.DataFrame(per_query_rows), results_dir / "01_preprocessing_per_query.csv"
    )
    _write_csv(pd.DataFrame(vocab_stats), results_dir / "01_preprocessing_vocab.csv")

    # O pré-processamento principal é o que maximiza o MAP do BM25.
    bm25_rows = summary[summary["model"] == "BM25"]
    best_config = str(bm25_rows.loc[bm25_rows["MAP"].idxmax(), "config"])

    return {
        "summary": summary,
        "vocab": pd.DataFrame(vocab_stats),
        "best_config": best_config,
    }


# ======================================================================
# Requisitos 2, 3, 4 e 5: Modelos, métricas e comparação
# ======================================================================
def experiment_models(
    collection: Cranfield, index, vsm, bm25, results_dir: Path
):
    """Avalia os dois modelos e compara-os consulta a consulta."""
    runs = {
        "VSM": run_model(vsm, collection.queries),
        "BM25": run_model(bm25, collection.queries),
    }

    per_query = {}
    aggregates = {}
    for name, run in runs.items():
        per_query[name], aggregates[name] = evaluate_run(run, collection.qrels, K)

    # --- resultados por consulta (requisito 4) ---
    rows = []
    for name in runs:
        for query_id, metrics in per_query[name].items():
            rows.append({"model": name, "query_id": query_id, **metrics})
    per_query_frame = pd.DataFrame(rows)
    _write_csv(per_query_frame, results_dir / "02_per_query_metrics.csv")

    summary = pd.DataFrame(
        [{"model": name, **agg} for name, agg in aggregates.items()]
    )
    _write_csv(summary, results_dir / "02_model_summary.csv")

    # --- comparação lado a lado (requisito 5) ---
    comparison_rows = []
    for query in collection.queries:
        query_id = query.query_id
        vsm_metrics = per_query["VSM"][query_id]
        bm25_metrics = per_query["BM25"][query_id]
        comparison_rows.append(
            {
                "query_id": query_id,
                "original_id": query.original_id,
                "num_relevant": int(vsm_metrics["num_relevant"]),
                "query_length_tokens": len(index.preprocessor(query.text)),
                "AP_VSM": vsm_metrics["AP"],
                "AP_BM25": bm25_metrics["AP"],
                "delta_AP": bm25_metrics["AP"] - vsm_metrics["AP"],
                f"P@{K}_VSM": vsm_metrics[f"P@{K}"],
                f"P@{K}_BM25": bm25_metrics[f"P@{K}"],
                "query_text": " ".join(query.text.split()),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    _write_csv(comparison, results_dir / "02_model_comparison_per_query.csv")

    # Curvas precisão-revocação interpoladas em 11 pontos.
    curves = {
        name: mean_interpolated_curve(run, collection.qrels) for name, run in runs.items()
    }
    _write_json(curves, results_dir / "02_precision_recall_curves.json")

    wins = {
        "BM25_melhor": int((comparison["delta_AP"] > 1e-9).sum()),
        "VSM_melhor": int((comparison["delta_AP"] < -1e-9).sum()),
        "empate": int((comparison["delta_AP"].abs() <= 1e-9).sum()),
    }
    _write_json(
        {"agregado": aggregates, "vitorias_por_AP": wins},
        results_dir / "02_model_summary.json",
    )

    return {
        "runs": runs,
        "per_query": per_query,
        "aggregates": aggregates,
        "summary": summary,
        "comparison": comparison,
        "curves": curves,
        "wins": wins,
    }


# ======================================================================
# Requisito 6: Análise por consulta
# ======================================================================
def experiment_query_analysis(
    collection: Cranfield, index, vsm, bm25, comparison: pd.DataFrame, results_dir: Path
):
    """Seleciona e detalha as seis consultas exigidas pelo requisito 6.

    A seleção é automática e determinística: entre as consultas com pelo
    menos 'MIN_RELEVANT_FOR_ANALYSIS' documentos relevantes, tomamos as
    duas com maior 'delta_AP' (BM25 superior), as duas com menor
    'delta_AP' (vetorial superior) e as duas com menor AP médio (ambos
    insatisfatórios).
    """
    eligible = comparison[comparison["num_relevant"] >= MIN_RELEVANT_FOR_ANALYSIS].copy()
    eligible["mean_AP"] = (eligible["AP_VSM"] + eligible["AP_BM25"]) / 2

    selection = {
        "bm25_superior": eligible.nlargest(2, "delta_AP")["query_id"].tolist(),
        "vsm_superior": eligible.nsmallest(2, "delta_AP")["query_id"].tolist(),
        "ambos_insatisfatorios": eligible.nsmallest(2, "mean_AP")["query_id"].tolist(),
    }

    detail = {}
    for category, query_ids in selection.items():
        detail[category] = []
        for query_id in query_ids:
            query = collection.query_by_id(query_id)
            qrels_for_query = collection.qrels.get(query_id, {})
            row = comparison[comparison["query_id"] == query_id].iloc[0]

            entry = {
                "query_id": query_id,
                "original_id": query.original_id,
                "query_text": " ".join(query.text.split()),
                "query_tokens": index.preprocessor(query.text),
                "num_relevant": int(row["num_relevant"]),
                "AP_VSM": float(row["AP_VSM"]),
                "AP_BM25": float(row["AP_BM25"]),
                "delta_AP": float(row["delta_AP"]),
                "top5": {},
            }
            for model_name, model in [("VSM", vsm), ("BM25", bm25)]:
                ranked = [doc_id for doc_id, _ in model.rank(query.text, top_k=5)]
                flags = _relevance_flags(ranked, qrels_for_query, 5)
                for flag in flags:
                    document = collection.document_by_id(flag["doc_id"])
                    flag["title"] = document.title
                    flag["length_tokens"] = len(index.preprocessor(document.text))
                entry["top5"][model_name] = flags
            detail[category].append(entry)

    _write_json({"selecao": selection, "detalhe": detail}, results_dir / "03_query_analysis.json")
    _write_csv(
        pd.DataFrame(
            [
                {
                    "categoria": category,
                    "query_id": entry["query_id"],
                    "modelo": model_name,
                    **flag,
                }
                for category, entries in detail.items()
                for entry in entries
                for model_name, flags in entry["top5"].items()
                for flag in flags
            ]
        ),
        results_dir / "03_query_analysis_top5.csv",
    )
    return {"selection": selection, "detail": detail}


# ======================================================================
# Requisito 7: Variação dos parâmetros do BM25
# ======================================================================
def experiment_bm25_parameters(
    collection: Cranfield, index, results_dir: Path
):
    """Varre a grade 'k1 x b' e isola o efeito de 'b' em uma consulta."""
    rows = []
    per_query_rows = []
    runs_by_params: dict[tuple[float, float], dict[int, list[int]]] = {}

    for k1 in K1_GRID:
        for b in B_GRID:
            model = BM25(index, k1=k1, b=b)
            run = run_model(model, collection.queries)
            runs_by_params[(k1, b)] = run
            per_query, agg = evaluate_run(run, collection.qrels, K)
            rows.append({"k1": k1, "b": b, **agg})
            for query_id, metrics in per_query.items():
                per_query_rows.append({"k1": k1, "b": b, "query_id": query_id, **metrics})

    grid = pd.DataFrame(rows)
    _write_csv(grid, results_dir / "04_bm25_parameter_grid.csv")
    _write_csv(pd.DataFrame(per_query_rows), results_dir / "04_bm25_parameter_per_query.csv")

    best = grid.loc[grid["MAP"].idxmax()]

    # --- efeito de b em uma consulta concreta ---
    # Fixamos k1 = 1.2 e procuramos a consulta cujo Top-10 mais muda entre
    # b = 0 e b = 1, medindo (a) a variação de AP e (b) quantos documentos
    # do Top-10 são trocados.
    run_b0 = runs_by_params[(DEFAULT_K1, 0.0)]
    run_b1 = runs_by_params[(DEFAULT_K1, 1.0)]

    effect_rows = []
    for query in collection.queries:
        query_id = query.query_id
        qrels_for_query = collection.qrels.get(query_id, {})
        relevant = binary_relevant(qrels_for_query)
        top_b0, top_b1 = run_b0[query_id][:K], run_b1[query_id][:K]
        effect_rows.append(
            {
                "query_id": query_id,
                "num_relevant": len(relevant),
                "AP_b0": average_precision(run_b0[query_id], relevant),
                "AP_b1": average_precision(run_b1[query_id], relevant),
                "top10_overlap": len(set(top_b0) & set(top_b1)),
                "top10_changed": K - len(set(top_b0) & set(top_b1)),
            }
        )
    effect = pd.DataFrame(effect_rows)
    effect["delta_AP"] = effect["AP_b1"] - effect["AP_b0"]
    effect["abs_delta_AP"] = effect["delta_AP"].abs()
    _write_csv(effect, results_dir / "04_effect_of_b_per_query.csv")

    eligible = effect[effect["num_relevant"] >= MIN_RELEVANT_FOR_ANALYSIS]
    # Consulta didática: grande variação de AP E boa troca de documentos.
    showcase_id = int(
        eligible.sort_values(["abs_delta_AP", "top10_changed"], ascending=False)
        .iloc[0]["query_id"]
    )

    showcase_query = collection.query_by_id(showcase_id)
    showcase_qrels = collection.qrels.get(showcase_id, {})
    showcase = {
        "query_id": showcase_id,
        "query_text": " ".join(showcase_query.text.split()),
        "num_relevant": len(binary_relevant(showcase_qrels)),
        "k1": DEFAULT_K1,
        "rankings": {},
    }
    for b in B_GRID:
        model = BM25(index, k1=DEFAULT_K1, b=b)
        ranked = [doc_id for doc_id, _ in model.rank(showcase_query.text, top_k=5)]
        flags = _relevance_flags(ranked, showcase_qrels, 5)
        for flag in flags:
            document = collection.document_by_id(flag["doc_id"])
            flag["title"] = document.title
            flag["length_tokens"] = len(index.preprocessor(document.text))
        showcase["rankings"][f"b={b}"] = {
            "top5": flags,
            "AP": average_precision(
                [d for d, _ in model.rank(showcase_query.text)],
                binary_relevant(showcase_qrels),
            ),
        }
    _write_json(showcase, results_dir / "04_effect_of_b_showcase.json")

    return {
        "grid": grid,
        "best": {"k1": float(best["k1"]), "b": float(best["b"]), "MAP": float(best["MAP"])},
        "effect": effect,
        "showcase": showcase,
    }


# ======================================================================
# Requisito 8: Modificação de consultas
# ======================================================================
def experiment_query_reformulation(
    collection: Cranfield, index, vsm, bm25, results_dir: Path
):
    """Executa as versões original e reformulada de cinco consultas."""
    rows = []
    detail = []

    for reformulation in REFORMULATIONS:
        query_id = reformulation.query_id
        qrels_for_query = collection.qrels.get(query_id, {})
        relevant = binary_relevant(qrels_for_query)

        entry = {**asdict(reformulation), "num_relevant": len(relevant), "versions": {}}

        for version, text in [
            ("original", reformulation.original),
            ("modificada", reformulation.modified),
        ]:
            for model_name, model in [("VSM", vsm), ("BM25", bm25)]:
                full_ranking = [doc_id for doc_id, _ in model.rank(text)]
                top10 = full_ranking[:K]
                metrics = {
                    f"P@{K}": sum(1 for d in top10 if d in relevant) / K,
                    f"R@{K}": (
                        sum(1 for d in top10 if d in relevant) / len(relevant)
                        if relevant
                        else 0.0
                    ),
                    "AP": average_precision(full_ranking, relevant),
                    f"NDCG@{K}": ndcg_at_k(full_ranking, qrels_for_query, K),
                }
                rows.append(
                    {
                        "query_id": query_id,
                        "operation": reformulation.operation,
                        "version": version,
                        "model": model_name,
                        "query_tokens": len(index.preprocessor(text)),
                        **metrics,
                    }
                )
                entry["versions"].setdefault(version, {})[model_name] = {
                    "tokens": index.preprocessor(text),
                    "metrics": metrics,
                    "top10": _relevance_flags(top10, qrels_for_query, K),
                }
        detail.append(entry)

    frame = pd.DataFrame(rows)
    _write_csv(frame, results_dir / "05_query_reformulation.csv")
    _write_json(detail, results_dir / "05_query_reformulation_detail.json")

    # Tabela larga: uma linha por (consulta, modelo) com antes/depois.
    pivot = frame.pivot_table(
        index=["query_id", "operation", "model"],
        columns="version",
        values=["AP", f"P@{K}", f"NDCG@{K}"],
    ).reset_index()
    pivot.columns = [
        "_".join(str(part) for part in col if part) for col in pivot.columns
    ]
    _write_csv(pivot, results_dir / "05_query_reformulation_wide.csv")

    return {"frame": frame, "detail": detail, "wide": pivot}


# ======================================================================
# Requisito 9: Análise de erros
# ======================================================================
def experiment_error_analysis(
    collection: Cranfield, index, vsm, bm25, comparison: pd.DataFrame, results_dir: Path
):
    """Seleciona falsos positivos no topo e um relevante fora do Top-10.

    A seleção é automática:

    - Falsos positivos: entre as consultas analisadas no requisito 6,
      documentos NÃO relevantes que o BM25 coloca nas duas primeiras
      posições, escolhidos pelo maior score.
    - Falso negativo: documento relevante com o PIOR posto entre os
      documentos relevantes das mesmas consultas, ou seja, o caso mais
      extremo de relevante perdido.

    Para cada caso guardamos a decomposição do score ('explain'), que é o
    que permite argumentar sobre a causa.
    """
    eligible = comparison[comparison["num_relevant"] >= MIN_RELEVANT_FOR_ANALYSIS]
    # Consultas com desempenho fraco em ambos os modelos: é onde os erros são
    # mais informativos.
    candidates = eligible.assign(
        mean_AP=(eligible["AP_VSM"] + eligible["AP_BM25"]) / 2
    ).nsmallest(6, "mean_AP")["query_id"].tolist()

    false_positives = []
    false_negatives = []

    for query_id in candidates:
        query = collection.query_by_id(query_id)
        qrels_for_query = collection.qrels.get(query_id, {})
        relevant = binary_relevant(qrels_for_query)
        ranking = bm25.rank(query.text)
        ranked_ids = [doc_id for doc_id, _ in ranking]

        # Falsos positivos: não relevantes nas duas primeiras posições.
        for position, (doc_id, score) in enumerate(ranking[:2], start=1):
            if doc_id in relevant:
                continue
            document = collection.document_by_id(doc_id)
            false_positives.append(
                {
                    "query_id": query_id,
                    "query_text": " ".join(query.text.split()),
                    "position": position,
                    "doc_id": doc_id,
                    "title": document.title,
                    "grade": qrels_for_query.get(doc_id, "não julgado"),
                    "score": score,
                    "doc_length_tokens": len(index.preprocessor(document.text)),
                    "explain_bm25": bm25.explain(query.text, doc_id),
                }
            )

        # Falso negativo: o relevante pior posicionado desta consulta.
        positions = {doc_id: i for i, doc_id in enumerate(ranked_ids, start=1)}
        missed = [(positions[d], d) for d in relevant if positions.get(d, 10**9) > K]
        if missed:
            worst_position, doc_id = max(missed)
            document = collection.document_by_id(doc_id)
            false_negatives.append(
                {
                    "query_id": query_id,
                    "query_text": " ".join(query.text.split()),
                    "doc_id": doc_id,
                    "title": document.title,
                    "grade": qrels_for_query.get(doc_id),
                    "rank_bm25": worst_position,
                    "rank_vsm": (
                        [d for d, _ in vsm.rank(query.text)].index(doc_id) + 1
                    ),
                    "doc_length_tokens": len(index.preprocessor(document.text)),
                    "query_tokens": index.preprocessor(query.text),
                    "doc_tokens_matched": sorted(
                        set(index.preprocessor(query.text))
                        & set(index.preprocessor(document.text))
                    ),
                    "explain_bm25": bm25.explain(query.text, doc_id),
                }
            )

    # Ordena os falsos positivos pelo score (os mais "convincentes" para o
    # modelo são os mais interessantes de explicar) e os falsos negativos
    # pelo posto (o mais perdido primeiro).
    false_positives.sort(key=lambda e: -e["score"])
    false_negatives.sort(key=lambda e: -e["rank_bm25"])

    payload = {
        "consultas_candidatas": candidates,
        "falsos_positivos": false_positives[:4],
        "falsos_negativos": false_negatives[:3],
    }
    _write_json(payload, results_dir / "06_error_analysis.json")
    _write_csv(
        pd.DataFrame(
            [
                {k: v for k, v in entry.items() if k != "explain_bm25"}
                for entry in false_positives[:4]
            ]
        ),
        results_dir / "06_error_analysis_false_positives.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    k: (v if not isinstance(v, list) else " ".join(map(str, v)))
                    for k, v in entry.items()
                    if k != "explain_bm25"
                }
                for entry in false_negatives[:3]
            ]
        ),
        results_dir / "06_error_analysis_false_negatives.csv",
    )
    return payload


# ======================================================================
# Apoio ao requisito 5: por que os modelos discordam
# ======================================================================
def experiment_length_analysis(
    collection: Cranfield, index, vsm, bm25, comparison: pd.DataFrame, results_dir: Path
):
    """Testa quantitativamente a hipótese da normalização por comprimento.

    A inspeção das consultas do requisito 6 sugere que a discordância entre
    os dois modelos é governada pelo COMPRIMENTO dos documentos: a
    normalização pelo cosseno é integral (divide pela norma completa do
    vetor), enquanto o BM25 com 'b = 0.75' normaliza apenas parcialmente.
    Se isso for verdade, então:

    a) os documentos do Top-10 do BM25 devem ser em média mais longos que os
       do modelo vetorial; e
    b) o BM25 deve levar vantagem justamente nas consultas cujos documentos
       relevantes são longos.

    Medimos as duas coisas. O item (b) é avaliado pela correlação de
    Spearman entre 'delta_AP' e o comprimento médio dos documentos
    relevantes da consulta.
    """
    from scipy.stats import spearmanr

    lengths = {
        doc_id: float(index.doc_lengths[row])
        for row, doc_id in enumerate(index.doc_ids)
    }

    rows = []
    for query in collection.queries:
        query_id = query.query_id
        relevant = binary_relevant(collection.qrels.get(query_id, {}))
        entry = {
            "query_id": query_id,
            "mean_length_relevant": float(np.mean([lengths[d] for d in relevant])),
        }
        for model_name, model in [("VSM", vsm), ("BM25", bm25)]:
            top = [doc_id for doc_id, _ in model.rank(query.text, top_k=K)]
            entry[f"mean_length_top{K}_{model_name}"] = float(
                np.mean([lengths[d] for d in top])
            )
        rows.append(entry)

    frame = pd.DataFrame(rows).merge(
        comparison[["query_id", "delta_AP", "num_relevant"]], on="query_id"
    )
    _write_csv(frame, results_dir / "08_length_analysis_per_query.csv")

    correlation = spearmanr(frame["delta_AP"], frame["mean_length_relevant"])
    summary = {
        "comprimento_medio_colecao": float(np.mean(list(lengths.values()))),
        "comprimento_medio_documentos_relevantes": float(
            frame["mean_length_relevant"].mean()
        ),
        f"comprimento_medio_top{K}_VSM": float(frame[f"mean_length_top{K}_VSM"].mean()),
        f"comprimento_medio_top{K}_BM25": float(frame[f"mean_length_top{K}_BM25"].mean()),
        "spearman_delta_AP_vs_comprimento_relevantes": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "n": int(len(frame)),
        },
    }
    _write_json(summary, results_dir / "08_length_analysis_summary.json")
    return {"frame": frame, "summary": summary}


# ======================================================================
# Experimentos complementares (decisões de implementação)
# ======================================================================
def experiment_design_choices(collection: Cranfield, index, results_dir: Path):
    """Quantifica duas decisões de implementação discutidas no relatório.

    1. Variante do IDF do BM25: Robertson com suavização (não negativo)
       contra a forma 'log((N+0.5)/(n+0.5))' mostrada na Aula 03.
    2. Mapeamento de ganho do NDCG: corrigir a escala invertida do
       Cranfield contra usar o grau bruto.
    """
    rows = []
    for variant in IDFVariant:
        model = BM25(index, k1=DEFAULT_K1, b=DEFAULT_B, idf_variant=variant)
        run = run_model(model, collection.queries)
        _, agg = evaluate_run(run, collection.qrels, K)
        rows.append({"idf_variant": variant.value, **agg})
    idf_frame = pd.DataFrame(rows)
    _write_csv(idf_frame, results_dir / "07_idf_variants.csv")

    model = BM25(index, k1=DEFAULT_K1, b=DEFAULT_B)
    run = run_model(model, collection.queries)
    gain_rows = []
    for mapping in GainMapping:
        per_query = {
            query_id: {
                f"NDCG@{K}": ndcg_at_k(ranked, collection.qrels.get(query_id, {}), K, mapping)
            }
            for query_id, ranked in run.items()
        }
        gain_rows.append({"gain_mapping": mapping.value, **aggregate(per_query)})
    gain_frame = pd.DataFrame(gain_rows)
    _write_csv(gain_frame, results_dir / "07_ndcg_gain_mappings.csv")

    return {"idf": idf_frame, "gain": gain_frame}


# ======================================================================
# Orquestração
# ======================================================================
def run_all_experiments(collection: Cranfield, data_dir: Path, results_dir: Path) -> dict:
    """Executa os nove requisitos em sequência e devolve tudo o que produzem."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Pré-processamento (4 configurações x 2 modelos)...")
    preprocessing = experiment_preprocessing(collection, data_dir, results_dir)
    best_config = preprocessing["best_config"]
    print(f"      melhor configuração por MAP do BM25: {best_config}")

    index, vsm, bm25 = build_models(collection, best_config, data_dir)

    print("[2/8] Modelos, métricas por consulta e comparação...")
    models = experiment_models(collection, index, vsm, bm25, results_dir)

    print("[3/8] Análise por consulta...")
    query_analysis = experiment_query_analysis(
        collection, index, vsm, bm25, models["comparison"], results_dir
    )

    print("[4/8] Variação dos parâmetros do BM25 (grade 3x3)...")
    parameters = experiment_bm25_parameters(collection, index, results_dir)

    print("[5/8] Reformulação manual de consultas...")
    reformulation = experiment_query_reformulation(
        collection, index, vsm, bm25, results_dir
    )

    print("[6/8] Análise de erros...")
    errors = experiment_error_analysis(
        collection, index, vsm, bm25, models["comparison"], results_dir
    )

    print("[7/8] Hipótese do comprimento dos documentos...")
    lengths = experiment_length_analysis(
        collection, index, vsm, bm25, models["comparison"], results_dir
    )

    print("[8/8] Decisões de implementação (IDF e ganho do NDCG)...")
    design = experiment_design_choices(collection, index, results_dir)

    # Metadados da execução, para reprodutibilidade.
    metadata = {
        "num_documents": len(collection.documents),
        "num_queries": len(collection.queries),
        "num_qrel_pairs": sum(len(v) for v in collection.qrels.values()),
        "best_preprocessing_config": best_config,
        "vocab_size": index.vocab_size,
        "avg_doc_length": index.avg_doc_length,
        "k": K,
        "bm25_default": {"k1": DEFAULT_K1, "b": DEFAULT_B},
        "bm25_best": parameters["best"],
        "aggregates": models["aggregates"],
        "wins": models["wins"],
    }
    _write_json(metadata, results_dir / "00_run_metadata.json")

    return {
        "metadata": metadata,
        "preprocessing": preprocessing,
        "models": models,
        "query_analysis": query_analysis,
        "parameters": parameters,
        "reformulation": reformulation,
        "errors": errors,
        "lengths": lengths,
        "design": design,
        "index": index,
        "vsm": vsm,
        "bm25": bm25,
    }
