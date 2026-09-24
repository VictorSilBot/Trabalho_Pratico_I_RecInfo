"""Modelo Vetorial (requisito 2): ponderação TF-IDF e similaridade do cosseno.

Esquema de ponderação
---------------------
Seguimos exatamente o apresentado na Aula 04 (esquema ``ltc.ltc`` na notação
SMART), aplicado tanto ao documento quanto à consulta:

    tf_{i,j}  = 1 + log10(f_{i,j})      se f_{i,j} > 0, senão 0
    idf_i     = log10(N / n_i)
    w_{i,j}   = tf_{i,j} * idf_i

O documento ``d_j`` é então um vetor de dimensão ``|V|`` (uma coordenada por
termo do vocabulário), extremamente esparso, e a consulta é ponderada da
mesma forma. O score é a similaridade do cosseno:

                      sum_i w_{i,q} * w_{i,j}
    sim(d_j, q) = ------------------------------------
                   sqrt(sum_i w_{i,j}^2) * sqrt(sum_i w_{i,q}^2)

Como normalizamos os vetores de documento para norma L2 unitária no momento
da indexação, e o vetor de consulta também é normalizado, o cosseno se reduz
a um simples produto interno -- é o mesmo cálculo, apenas reorganizado para
ser feito de uma vez para os 1400 documentos.

Observações sobre a formulação
------------------------------
* ``idf_i = log10(N/n_i)`` vale exatamente 0 para um termo que ocorre em
  TODOS os documentos, que assim é silenciosamente eliminado do ranking --
  comportamento discutido no slide 25 da Aula 04.
* A base do logaritmo é uma escolha (usamos 10, convenção do Manning et al.).
  A base afeta a escala relativa entre ``tf`` e ``idf``, mas o efeito é
  desprezível na prática; ``log_base`` permite verificar isso empiricamente.
* A normalização pelo cosseno é o que impede documentos longos de dominarem
  o ranking só por acumularem mais termos (Aula 04, slides 26-28).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .indexing import InvertedIndex


@dataclass
class VectorSpaceModel:
    """Modelo vetorial TF-IDF com similaridade do cosseno.

    Parâmetros
    ----------
    index : índice invertido já construído.
    log_base : base do logaritmo usada em ``1 + log(f)`` e em ``log(N/n)``.
    """

    index: InvertedIndex
    log_base: float = 10.0

    def __post_init__(self) -> None:
        idx = self.index
        log = np.log(self.log_base)

        # idf_i = log(N / n_i). Termos com n_i == N recebem idf = 0.
        with np.errstate(divide="ignore"):
            self.idf = np.log(idx.num_docs / np.maximum(idx.df, 1.0)) / log

        # Pesos dos documentos: w_ij = (1 + log f_ij) * idf_i, sobre a matriz
        # esparsa. Operamos apenas nos elementos não nulos (`.data`), que são
        # exatamente os pares (documento, termo) com f_ij > 0.
        weights = idx.tf.copy().astype(np.float64)
        weights.data = 1.0 + np.log(weights.data) / log
        # Multiplica cada coluna pelo idf do respectivo termo.
        weights = weights @ sparse.diags(self.idf)
        weights = sparse.csr_matrix(weights)

        # Normalização L2 por documento (o denominador do cosseno).
        norms = np.sqrt(weights.multiply(weights).sum(axis=1)).A.ravel()
        self.doc_norms = norms
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        self.doc_weights = sparse.csr_matrix(sparse.diags(inverse) @ weights)

    # ------------------------------------------------------------------
    # Ponderação da consulta
    # ------------------------------------------------------------------
    def query_weights(self, query_text: str) -> dict[str, float]:
        """Pesos TF-IDF (já normalizados) dos termos da consulta.

        Termos fora do vocabulário da coleção são descartados: seu ``n_i`` é
        zero e eles não podem casar com documento algum.
        """
        counts = self.index.query_vector(query_text)
        log = np.log(self.log_base)

        raw: dict[str, float] = {}
        for term, frequency in counts.items():
            column = self.index.term_id(term)
            if column is None:
                continue
            tf = 1.0 + np.log(frequency) / log
            raw[term] = tf * float(self.idf[column])

        norm = float(np.sqrt(sum(w * w for w in raw.values())))
        if norm == 0.0:
            return {}
        return {term: w / norm for term, w in raw.items()}

    # ------------------------------------------------------------------
    # Ranqueamento
    # ------------------------------------------------------------------
    def score(self, query_text: str) -> np.ndarray:
        """Vetor (N,) com a similaridade do cosseno de cada documento."""
        weights = self.query_weights(query_text)
        scores = np.zeros(self.index.num_docs, dtype=np.float64)
        if not weights:
            return scores

        # Produto interno entre o vetor da consulta e os vetores de documento
        # (ambos unitários) == cosseno. Percorremos apenas os termos da
        # consulta, acumulando a contribuição de cada um em suas postings.
        doc_weights_csc = self.doc_weights.tocsc()
        for term, weight in weights.items():
            column = self.index.term_id(term)
            start, end = doc_weights_csc.indptr[column], doc_weights_csc.indptr[column + 1]
            rows = doc_weights_csc.indices[start:end]
            scores[rows] += weight * doc_weights_csc.data[start:end]
        return scores

    def rank(self, query_text: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Ranking ``[(doc_id, score), ...]`` em ordem decrescente de score.

        O desempate é feito pelo id do documento (crescente), para que o
        ranking seja determinístico e reprodutível.
        """
        scores = self.score(query_text)
        return _rank_from_scores(scores, self.index.doc_ids, top_k)

    # ------------------------------------------------------------------
    # Explicabilidade (útil para justificar o score de um documento)
    # ------------------------------------------------------------------
    def explain(self, query_text: str, doc_id: int) -> dict:
        """Decompõe o score de um documento, termo a termo."""
        weights = self.query_weights(query_text)
        row = self.index.doc_ids.index(doc_id)
        contributions = []
        total = 0.0
        for term, query_weight in sorted(weights.items()):
            column = self.index.term_id(term)
            doc_weight = float(self.doc_weights[row, column])
            contribution = query_weight * doc_weight
            total += contribution
            contributions.append(
                {
                    "term": term,
                    "f_ij": float(self.index.tf[row, column]),
                    "n_i": float(self.index.df[column]),
                    "idf": float(self.idf[column]),
                    "w_iq_normalized": query_weight,
                    "w_ij_normalized": doc_weight,
                    "contribution": contribution,
                }
            )
        contributions.sort(key=lambda c: c["contribution"], reverse=True)
        return {
            "model": "VSM",
            "doc_id": doc_id,
            "doc_length": float(self.index.doc_lengths[row]),
            "doc_norm": float(self.doc_norms[row]),
            "score": total,
            "terms": contributions,
        }


def _rank_from_scores(
    scores: np.ndarray, doc_ids: list[int], top_k: int | None
) -> list[tuple[int, float]]:
    """Ordena por score decrescente, desempatando por doc_id crescente."""
    doc_id_array = np.asarray(doc_ids)
    # np.lexsort ordena pela ÚLTIMA chave primeiro: score desc, depois id asc.
    order = np.lexsort((doc_id_array, -scores))
    if top_k is not None:
        order = order[:top_k]
    return [(int(doc_id_array[i]), float(scores[i])) for i in order]
