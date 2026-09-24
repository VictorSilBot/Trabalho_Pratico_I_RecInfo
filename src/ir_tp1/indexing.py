"""Índice invertido da coleção, compartilhado pelo Modelo Vetorial e pelo BM25.

A estrutura central é uma matriz esparsa CSR ``tf`` de dimensão
``(N documentos x V termos)`` contendo frequências BRUTAS de termo. Ambos os
modelos derivam seus pesos dessa mesma matriz, o que garante que qualquer
diferença de desempenho entre eles venha da função de ranqueamento e não de
diferenças de indexação.

Guardar a matriz em CSR (linhas = documentos) e também em CSC (colunas =
termos) dá acesso barato tanto ao vetor de um documento quanto à lista de
postings de um termo -- que é o acesso natural do BM25.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .preprocess import Preprocessor


@dataclass
class InvertedIndex:
    """Índice invertido com estatísticas de coleção.

    Atributos
    ---------
    vocabulary : mapeia termo -> coluna da matriz.
    terms      : lista de termos na ordem das colunas (inverso de ``vocabulary``).
    tf         : matriz CSR (N x V) de frequências brutas de termo.
    tf_csc     : a mesma matriz em CSC, para varrer postings por termo.
    df         : vetor (V,) com a frequência de documentos ``n_i`` de cada termo.
    doc_lengths: vetor (N,) com ``|d|`` em número de tokens (após o
                 pré-processamento), usado na normalização do BM25.
    doc_ids    : identificadores originais dos documentos, na ordem das linhas.
    """

    vocabulary: dict[str, int]
    terms: list[str]
    tf: sparse.csr_matrix
    df: np.ndarray
    doc_lengths: np.ndarray
    doc_ids: list[int]
    preprocessor: Preprocessor
    tf_csc: sparse.csc_matrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tf_csc = self.tf.tocsc()

    @property
    def num_docs(self) -> int:
        return self.tf.shape[0]

    @property
    def vocab_size(self) -> int:
        return self.tf.shape[1]

    @property
    def avg_doc_length(self) -> float:
        return float(self.doc_lengths.mean())

    def term_id(self, term: str) -> int | None:
        return self.vocabulary.get(term)

    def postings(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        """Retorna ``(indices_de_documentos, frequencias)`` para um termo.

        Se o termo não estiver no vocabulário, retorna arrays vazios -- é o
        comportamento desejado para termos de consulta fora da coleção, que
        simplesmente não contribuem para nenhum score.
        """
        column = self.term_id(term)
        if column is None:
            empty_i = np.empty(0, dtype=np.int32)
            return empty_i, np.empty(0, dtype=np.float64)
        start, end = self.tf_csc.indptr[column], self.tf_csc.indptr[column + 1]
        return self.tf_csc.indices[start:end], self.tf_csc.data[start:end]

    def doc_term_frequency(self, doc_id: int, term: str) -> float:
        """``f_{i,j}`` para um par (documento, termo), pelo id original do doc."""
        column = self.term_id(term)
        if column is None:
            return 0.0
        row = self.doc_ids.index(doc_id)
        return float(self.tf[row, column])

    def query_vector(self, text: str) -> dict[str, int]:
        """Pré-processa a consulta e devolve ``{termo: frequência na consulta}``."""
        counts: dict[str, int] = {}
        for token in self.preprocessor(text):
            counts[token] = counts.get(token, 0) + 1
        return counts


def build_index(
    texts: list[str],
    doc_ids: list[int],
    preprocessor: Preprocessor,
) -> InvertedIndex:
    """Constrói o índice invertido aplicando ``preprocessor`` a cada documento."""
    if len(texts) != len(doc_ids):
        raise ValueError("texts e doc_ids devem ter o mesmo tamanho")

    vocabulary: dict[str, int] = {}
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    doc_lengths = np.zeros(len(texts), dtype=np.float64)

    for row, text in enumerate(texts):
        tokens = preprocessor(text)
        doc_lengths[row] = len(tokens)

        counts: dict[int, int] = {}
        for token in tokens:
            column = vocabulary.get(token)
            if column is None:
                column = len(vocabulary)
                vocabulary[token] = column
            counts[column] = counts.get(column, 0) + 1

        indices.extend(counts.keys())
        data.extend(float(v) for v in counts.values())
        indptr.append(len(indices))

    tf = sparse.csr_matrix(
        (np.asarray(data), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
        shape=(len(texts), len(vocabulary)),
        dtype=np.float64,
    )
    tf.sort_indices()

    # n_i = número de documentos em que o termo ocorre (não a frequência total).
    df = np.diff(tf.tocsc().indptr).astype(np.float64)

    terms = [""] * len(vocabulary)
    for term, column in vocabulary.items():
        terms[column] = term

    return InvertedIndex(
        vocabulary=vocabulary,
        terms=terms,
        tf=tf,
        df=df,
        doc_lengths=doc_lengths,
        doc_ids=list(doc_ids),
        preprocessor=preprocessor,
    )
