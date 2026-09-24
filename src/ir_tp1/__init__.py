"""Trabalho Prático 1 -- SCC0282 Recuperação de Informação (ICMC/USP).

Sistema de recuperação textual sobre a coleção Cranfield, com Modelo Vetorial
(TF-IDF + cosseno) e Modelo Probabilístico (BM25), avaliados com Precision@10,
Recall@10, F1@10, MAP, MRR e NDCG@10.
"""

from .bm25 import BM25, IDFVariant
from .dataset import Cranfield, load_cranfield
from .indexing import InvertedIndex, build_index
from .preprocess import CONFIG_ORDER, Preprocessor, build_configs
from .vsm import VectorSpaceModel

__all__ = [
    "BM25",
    "CONFIG_ORDER",
    "Cranfield",
    "IDFVariant",
    "InvertedIndex",
    "Preprocessor",
    "VectorSpaceModel",
    "build_configs",
    "build_index",
    "load_cranfield",
]
