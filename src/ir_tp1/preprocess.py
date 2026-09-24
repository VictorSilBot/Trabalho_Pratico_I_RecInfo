"""Pré-processamento textual: tokenização, normalização, stopwords e stemming.

As quatro configurações exigidas pelo enunciado (requisito 1) são expostas em
:data:`PREPROCESSING_CONFIGS`:

    ``none``      -- sem remoção de stopwords e sem stemming (linha de base)
    ``stop``      -- apenas remoção de stopwords
    ``stem``      -- apenas stemming (Porter)
    ``stop_stem`` -- remoção de stopwords + stemming

Decisões de implementação
-------------------------
Tokenização: expressão regular ``[a-z]+`` sobre o texto já convertido para
minúsculas. É determinística, não depende de modelos baixados em tempo de
execução e casa bem com o Cranfield, cujo texto é ASCII técnico. O efeito
colateral relevante é que hifens e barras viram separadores
(``boundary-layer-control`` -> ``boundary``, ``layer``, ``control``) e que
números são descartados -- ambos discutidos no relatório.

Stopwords: lista ``english`` do NLTK (198 formas). Uma cópia é gravada em
``data/stopwords_en.txt`` no primeiro uso para que a execução seja
reprodutível mesmo sem rede.

Stemming: ``nltk.stem.PorterStemmer``, com memoização, pois o mesmo tipo
reaparece milhares de vezes na coleção.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z]+")

# Preenchido sob demanda por :func:`load_stopwords`.
_STOPWORDS_CACHE: frozenset[str] | None = None


def tokenize(text: str) -> list[str]:
    """Normaliza para minúsculas e extrai sequências alfabéticas."""
    return _TOKEN_RE.findall(text.lower())


def load_stopwords(cache_path: Path | str | None = None) -> frozenset[str]:
    """Carrega a lista de stopwords do inglês (NLTK), com cache em disco."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    cache_file = Path(cache_path) if cache_path else None
    words: set[str] | None = None

    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            words = set(nltk_stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(nltk_stopwords.words("english"))
    except Exception as exc:  # rede indisponível, NLTK ausente, etc.
        if cache_file is None or not cache_file.exists():
            raise RuntimeError(
                "Não foi possível obter as stopwords do NLTK e não há cópia "
                f"local em {cache_file}. Execute com rede ao menos uma vez."
            ) from exc
        words = set(cache_file.read_text(encoding="utf-8").split())

    # As stopwords do NLTK contêm apóstrofos ("don't"); como nosso tokenizador
    # produz apenas [a-z]+, expandimos cada entrada em seus fragmentos
    # alfabéticos para que "don't" remova tanto "don" quanto "t".
    expanded = {fragment for word in words for fragment in _TOKEN_RE.findall(word)}

    if cache_file is not None and not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("\n".join(sorted(expanded)), encoding="utf-8")

    _STOPWORDS_CACHE = frozenset(expanded)
    return _STOPWORDS_CACHE


@lru_cache(maxsize=1)
def _porter_stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Radical de Porter para um token, memoizado."""
    return _porter_stemmer().stem(token)


@dataclass(frozen=True)
class Preprocessor:
    """Pipeline configurável de pré-processamento.

    A ordem das etapas importa: removemos stopwords ANTES do stemming, pois o
    radical de uma stopword pode coincidir com o de um termo de conteúdo
    (e.g. "being" -> "be"), e a lista de stopwords é definida sobre formas
    plenas, não sobre radicais.
    """

    name: str
    remove_stopwords: bool
    apply_stemming: bool
    stopwords_path: Path | None = None

    @property
    def label(self) -> str:
        parts = []
        parts.append("stopwords" if self.remove_stopwords else "sem stopwords")
        parts.append("stemming" if self.apply_stemming else "sem stemming")
        return " + ".join(parts)

    def __call__(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if self.remove_stopwords:
            stops = load_stopwords(self.stopwords_path)
            tokens = [t for t in tokens if t not in stops]
        if self.apply_stemming:
            tokens = [stem(t) for t in tokens]
        return tokens


def build_configs(stopwords_path: Path | str | None = None) -> dict[str, Preprocessor]:
    """Constrói as quatro configurações exigidas no requisito 1."""
    path = Path(stopwords_path) if stopwords_path else None
    specs = [
        ("none", False, False),
        ("stop", True, False),
        ("stem", False, True),
        ("stop_stem", True, True),
    ]
    return {
        name: Preprocessor(
            name=name,
            remove_stopwords=rm,
            apply_stemming=st,
            stopwords_path=path,
        )
        for name, rm, st in specs
    }


#: Configurações padrão (sem cache em disco das stopwords).
PREPROCESSING_CONFIGS = build_configs()

#: Ordem canônica usada em tabelas e gráficos.
CONFIG_ORDER = ["none", "stop", "stem", "stop_stem"]
