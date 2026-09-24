"""Modelo Probabilístico BM25 (requisito 3), implementado explicitamente.

Função de ranqueamento (Aula 03, slide 25):

                          f(t,d) * (k1 + 1)
    score(d,q) = sum   IDF(t) * ---------------------------------------
                 t in q          f(t,d) + k1 * (1 - b + b * |d| / avgdl)

Os três ingredientes, e o que cada parâmetro controla:

* **Raridade** -- ``IDF(t)``. Termos que ocorrem em poucos documentos pesam
  mais. Duas variantes estão implementadas (ver :class:`IDFVariant`).
* **Saturação da frequência** -- o quociente cresce com ``f(t,d)`` mas tende
  assintoticamente a ``k1 + 1``. Com ``k1 = 0`` o modelo vira binário (só
  importa se o termo ocorre); quanto maior ``k1``, mais perto de linear fica
  a resposta à frequência. É a diferença essencial em relação ao ``tf``
  logarítmico do modelo vetorial, que não satura.
* **Normalização por comprimento** -- ``b`` interpola entre nenhuma
  normalização (``b = 0``, documentos longos são favorecidos por acumularem
  ocorrências) e normalização total (``b = 1``, ``f(t,d)`` é integralmente
  dividido pelo comprimento relativo ``|d|/avgdl``).

Frequência do termo na consulta
-------------------------------
A fórmula da aula não inclui o componente ``k3`` de saturação do lado da
consulta. Adotamos a convenção usual de **somar a contribuição do termo
tantas vezes quanto ele ocorre na consulta** (equivalente a ``k3 -> infinito``),
implementada como uma multiplicação pela frequência ``qtf``. Nas consultas do
Cranfield isso raramente muda algo, pois termos repetidos em uma consulta são
quase sempre stopwords.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .indexing import InvertedIndex
from .vsm import _rank_from_scores


class IDFVariant(str, Enum):
    """Variantes de IDF para o BM25."""

    #: IDF de Robertson/Spärck Jones com suavização, na forma usada pelo
    #: Lucene: log(1 + (N - n + 0.5) / (n + 0.5)). O "1 +" garante que o
    #: valor nunca seja negativo -- sem ele, termos presentes em mais da
    #: metade da coleção receberiam peso NEGATIVO e penalizariam documentos
    #: que os contêm, o que é indesejável numa coleção pequena e temática
    #: como a Cranfield (onde termos como "flow" ocorrem em muitos documentos).
    ROBERTSON = "robertson"

    #: Forma apresentada na Aula 03 para a seleção inicial do BIM:
    #: log((N + 0.5) / (n + 0.5)). Sempre positiva, decresce mais suavemente.
    BIM = "bim"


def compute_idf(df: np.ndarray, num_docs: int, variant: IDFVariant) -> np.ndarray:
    """Calcula o vetor de IDF para todos os termos do vocabulário."""
    n = df.astype(np.float64)
    N = float(num_docs)
    if variant is IDFVariant.ROBERTSON:
        return np.log(1.0 + (N - n + 0.5) / (n + 0.5))
    if variant is IDFVariant.BIM:
        return np.log((N + 0.5) / (n + 0.5))
    raise ValueError(f"Variante de IDF desconhecida: {variant}")


@dataclass
class BM25:
    """BM25 sobre um índice invertido.

    Parâmetros
    ----------
    index : índice invertido já construído.
    k1 : saturação da frequência do termo (``k1 >= 0``).
    b  : intensidade da normalização por comprimento (``0 <= b <= 1``).
    idf_variant : forma do IDF (ver :class:`IDFVariant`).
    """

    index: InvertedIndex
    k1: float = 1.2
    b: float = 0.75
    idf_variant: IDFVariant = IDFVariant.ROBERTSON

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 deve ser >= 0")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b deve estar em [0, 1]")

        self.idf = compute_idf(self.index.df, self.index.num_docs, self.idf_variant)

        # Denominador da normalização, pré-computado por documento:
        #     K_d = k1 * (1 - b + b * |d| / avgdl)
        # Ele não depende do termo, apenas do documento e dos parâmetros.
        avgdl = self.index.avg_doc_length
        self.length_norm = self.k1 * (
            1.0 - self.b + self.b * (self.index.doc_lengths / avgdl)
        )

    # ------------------------------------------------------------------
    # Ranqueamento
    # ------------------------------------------------------------------
    def score(self, query_text: str) -> np.ndarray:
        """Vetor (N,) com o score BM25 de cada documento.

        Percorremos os termos da consulta e, para cada um, apenas a sua lista
        de postings -- documentos que não contêm nenhum termo da consulta
        ficam com score 0, como esperado.
        """
        counts = self.index.query_vector(query_text)
        scores = np.zeros(self.index.num_docs, dtype=np.float64)

        for term, query_frequency in counts.items():
            column = self.index.term_id(term)
            if column is None:
                continue  # termo fora do vocabulário: não contribui

            doc_rows, term_frequencies = self.index.postings(term)
            if doc_rows.size == 0:
                continue

            idf = float(self.idf[column])

            # Componente de saturação/normalização, calculado somente nos
            # documentos onde o termo de fato ocorre.
            numerator = term_frequencies * (self.k1 + 1.0)
            denominator = term_frequencies + self.length_norm[doc_rows]

            scores[doc_rows] += query_frequency * idf * (numerator / denominator)

        return scores

    def rank(self, query_text: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Ranking ``[(doc_id, score), ...]``, desempatado por doc_id crescente."""
        scores = self.score(query_text)
        return _rank_from_scores(scores, self.index.doc_ids, top_k)

    # ------------------------------------------------------------------
    # Explicabilidade
    # ------------------------------------------------------------------
    def explain(self, query_text: str, doc_id: int) -> dict:
        """Decompõe o score BM25 de um documento, termo a termo."""
        counts = self.index.query_vector(query_text)
        row = self.index.doc_ids.index(doc_id)
        doc_length = float(self.index.doc_lengths[row])
        norm = float(self.length_norm[row])

        contributions = []
        total = 0.0
        for term, query_frequency in sorted(counts.items()):
            column = self.index.term_id(term)
            if column is None:
                continue
            tf = float(self.index.tf[row, column])
            idf = float(self.idf[column])
            saturation = (tf * (self.k1 + 1.0)) / (tf + norm) if tf > 0 else 0.0
            contribution = query_frequency * idf * saturation
            total += contribution
            contributions.append(
                {
                    "term": term,
                    "qtf": query_frequency,
                    "f_td": tf,
                    "n_i": float(self.index.df[column]),
                    "idf": idf,
                    "saturation_factor": saturation,
                    "contribution": contribution,
                }
            )
        contributions.sort(key=lambda c: c["contribution"], reverse=True)
        return {
            "model": f"BM25(k1={self.k1}, b={self.b})",
            "doc_id": doc_id,
            "doc_length": doc_length,
            "avg_doc_length": self.index.avg_doc_length,
            "length_norm_K": norm,
            "score": total,
            "terms": contributions,
        }
