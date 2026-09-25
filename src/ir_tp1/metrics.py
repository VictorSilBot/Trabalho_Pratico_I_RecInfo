"""Métricas de avaliação (requisito 4), conforme as definições da Aula 05.

Política de relevância (definida pelo enunciado):
Para as métricas BINÁRIAS (Precision, Recall, F1, MAP, MRR) é relevante todo
documento com grau >= 1. O grau -1 e os documentos não julgados contam como
não relevantes.

Para o NDCG, mantemos os graus positivos como relevância graduada. Aqui há
uma armadilha específica do Cranfield: sua escala é invertida: o grau 1
é o melhor ("resposta completa") e o 4 é o pior ("interesse mínimo"), como
documentado em 'cranqrel.readme'. Usar o grau bruto como ganho daria ao
documento MENOS relevante o MAIOR ganho, invertendo o que o NDCG deveria
medir. Por isso o mapeamento padrão é 'GainMapping.INVERTED':

    grau  1 -> rel 4      (melhor)
    grau  2 -> rel 3
    grau  3 -> rel 2
    grau  4 -> rel 1
    grau -1, não julgado -> rel 0

O mapeamento 'RAW' (ganho = grau bruto) está implementado para permitir a
comparação numérica entre as duas convenções, reportada no relatório.

O ganho segue a forma comum da Aula 05, '2^rel - 1', e o desconto é
'log2(i + 1)' com 'i' começando em 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from enum import Enum

import numpy as np

#: Níveis padrão de revocação para a curva de 11 pontos (Aula 05, slide 20).
STANDARD_RECALL_LEVELS = [i / 10 for i in range(11)]


class GainMapping(str, Enum):
    """Como converter o grau do Cranfield em ganho de relevância para o NDCG."""

    #: Corrige a escala invertida do Cranfield: grau 1 (melhor) -> rel 4.
    INVERTED = "inverted"
    #: Usa o grau bruto como relevância (apenas para comparação).
    RAW = "raw"


def grade_to_gain_level(grade: int, mapping: GainMapping = GainMapping.INVERTED) -> int:
    """Converte um grau do Cranfield no nível de relevância usado pelo NDCG."""
    if grade < 1:
        return 0
    if mapping is GainMapping.INVERTED:
        return max(0, 5 - grade)
    return grade


def binary_relevant(qrels_for_query: dict[int, int]) -> set[int]:
    """Conjunto 'R_q' de documentos relevantes (grau >= 1)."""
    return {doc_id for doc_id, grade in qrels_for_query.items() if grade >= 1}


# ----------------------------------------------------------------------
# Métricas binárias
# ----------------------------------------------------------------------
def precision_at_k(ranked_ids: Sequence[int], relevant: set[int], k: int) -> float:
    """Fração dos 'k' primeiros documentos que é relevante.

    O denominador é 'k' fixo (e não 'min(k, len(ranking))'): ranqueamos
    toda a coleção, então o ranking sempre tem pelo menos 'k' documentos.
    """
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / k


def recall_at_k(ranked_ids: Sequence[int], relevant: set[int], k: int) -> float:
    """Fração dos documentos relevantes que aparece nos 'k' primeiros."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def f1_at_k(ranked_ids: Sequence[int], relevant: set[int], k: int) -> float:
    """Média harmônica entre P@k e R@k."""
    precision = precision_at_k(ranked_ids, relevant, k)
    recall = recall_at_k(ranked_ids, relevant, k)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def average_precision(
    ranked_ids: Sequence[int], relevant: set[int], cutoff: int | None = None
) -> float:
    """Average Precision (Aula 05, slide 29).

        AP(q) = (1 / |R_q|) * sum_k  P@k * rel(k)

    O denominador é '|R_q|', o número TOTAL de relevantes da consulta,
    não o número de relevantes recuperados. Assim, relevantes não
    recuperados contribuem com zero e a métrica penaliza a baixa revocação.
    """
    if not relevant:
        return 0.0
    ranking = ranked_ids if cutoff is None else ranked_ids[:cutoff]
    hits = 0
    precision_sum = 0.0
    for position, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / position
    return precision_sum / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[int], relevant: set[int]) -> float:
    """'1 / posição' do primeiro documento relevante; 0 se não houver nenhum."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


# ----------------------------------------------------------------------
# Métrica graduada
# ----------------------------------------------------------------------
def dcg_at_k(gains: Iterable[int], k: int) -> float:
    """'DCG@k = sum_i (2^rel_i - 1) / log2(i + 1)'."""
    total = 0.0
    for position, relevance in enumerate(list(gains)[:k], start=1):
        if relevance > 0:
            total += (2.0**relevance - 1.0) / math.log2(position + 1)
    return total


def ndcg_at_k(
    ranked_ids: Sequence[int],
    qrels_for_query: dict[int, int],
    k: int,
    mapping: GainMapping = GainMapping.INVERTED,
) -> float:
    """NDCG@k com relevância graduada (Aula 05, slides 35-38)."""
    gains = [
        grade_to_gain_level(qrels_for_query.get(doc_id, 0), mapping)
        for doc_id in ranked_ids[:k]
    ]
    dcg = dcg_at_k(gains, k)

    ideal = sorted(
        (grade_to_gain_level(grade, mapping) for grade in qrels_for_query.values()),
        reverse=True,
    )
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------
# Curva precisão x revocação interpolada em 11 pontos
# ----------------------------------------------------------------------
def interpolated_precision_recall(
    ranked_ids: Sequence[int], relevant: set[int]
) -> list[float]:
    """Precisão interpolada nos 11 níveis padrão de revocação.

    Interpolação da Aula 05 (slide 20): 'P(r_j) = max_{r >= r_j} P(r)'.
    """
    if not relevant:
        return [0.0] * len(STANDARD_RECALL_LEVELS)

    observed: list[tuple[float, float]] = []
    hits = 0
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            observed.append((hits / len(relevant), hits / position))

    result = []
    for level in STANDARD_RECALL_LEVELS:
        candidates = [p for r, p in observed if r >= level]
        result.append(max(candidates) if candidates else 0.0)
    return result


# ----------------------------------------------------------------------
# Avaliação completa de uma consulta / de um run
# ----------------------------------------------------------------------
def evaluate_query(
    ranked_ids: Sequence[int],
    qrels_for_query: dict[int, int],
    k: int = 10,
    gain_mapping: GainMapping = GainMapping.INVERTED,
) -> dict[str, float]:
    """Todas as métricas para uma única consulta."""
    relevant = binary_relevant(qrels_for_query)
    return {
        f"P@{k}": precision_at_k(ranked_ids, relevant, k),
        f"R@{k}": recall_at_k(ranked_ids, relevant, k),
        f"F1@{k}": f1_at_k(ranked_ids, relevant, k),
        "AP": average_precision(ranked_ids, relevant),
        "RR": reciprocal_rank(ranked_ids, relevant),
        f"NDCG@{k}": ndcg_at_k(ranked_ids, qrels_for_query, k, gain_mapping),
        "num_relevant": float(len(relevant)),
    }


#: Ordem canônica das métricas em tabelas e gráficos.
METRIC_ORDER = ["P@10", "R@10", "F1@10", "MAP", "MRR", "NDCG@10"]

#: Como as métricas por consulta se agregam no nome final da média.
_AGGREGATE_NAME = {"AP": "MAP", "RR": "MRR"}


def aggregate(per_query: dict[int, dict[str, float]]) -> dict[str, float]:
    """Agrega as métricas por consulta em médias sobre todas as consultas.

    'AP' vira 'MAP' e 'RR' vira 'MRR', que é exatamente a definição
    dessas duas métricas: a média sobre as consultas.
    """
    if not per_query:
        return {}
    names = [n for n in next(iter(per_query.values())) if n != "num_relevant"]
    result = {}
    for name in names:
        values = [metrics[name] for metrics in per_query.values()]
        result[_AGGREGATE_NAME.get(name, name)] = float(np.mean(values))
    return result


def evaluate_run(
    run: dict[int, Sequence[int]],
    qrels: dict[int, dict[int, int]],
    k: int = 10,
    gain_mapping: GainMapping = GainMapping.INVERTED,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    """Avalia um run completo, devolvendo '(por_consulta, agregado)'."""
    per_query = {
        query_id: evaluate_query(ranked, qrels.get(query_id, {}), k, gain_mapping)
        for query_id, ranked in run.items()
    }
    return per_query, aggregate(per_query)


def mean_interpolated_curve(
    run: dict[int, Sequence[int]], qrels: dict[int, dict[int, int]]
) -> list[float]:
    """Curva precisão-revocação média em 11 pontos (Aula 05, slide 21)."""
    curves = [
        interpolated_precision_recall(ranked, binary_relevant(qrels.get(query_id, {})))
        for query_id, ranked in run.items()
    ]
    if not curves:
        return [0.0] * len(STANDARD_RECALL_LEVELS)
    return list(np.mean(np.asarray(curves), axis=0))
