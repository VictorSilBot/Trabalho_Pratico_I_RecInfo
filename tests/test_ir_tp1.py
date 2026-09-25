"""Testes de correção da implementação.

Executável tanto com 'pytest' quanto diretamente:

    python tests/test_ir_tp1.py

Três famílias de teste:

1. Métricas contra valores calculados à mão, usando o exercício do slide
   41 da Aula 05 (assim os números conferem com o que foi visto em aula).
2. Implementação de referência ingênua: uma versão em Python puro, com
   dicionários e laços, escrita diretamente a partir das fórmulas, é
   comparada com a versão vetorizada (scipy) sobre a coleção real. Se as
   duas concordarem em todos os 1400 documentos de várias consultas, a
   otimização com matrizes esparsas está correta.
3. Verificação cruzada com o scikit-learn para o modelo vetorial, com a
   ressalva de que o 'TfidfVectorizer' usa 'idf = ln(N/df) + 1' (o "+1"
   impede que termos muito frequentes zerem), então esperamos rankings
   altamente correlacionados, não idênticos.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_tp1 import BM25, VectorSpaceModel, build_configs, build_index, load_cranfield  # noqa: E402
from ir_tp1 import metrics as M  # noqa: E402

DATA_DIR = ROOT / "data"


# ======================================================================
# 1. Métricas contra o exercício do slide 41 da Aula 05
# ======================================================================
# "Suponha uma base com 100 documentos, dos quais apenas 4 são relevantes.
#  O sistema retorna: R,N,R,N,R,R,N,N,N,N  com graus 2,0,3,0,3,1,0,0,0,0"
SLIDE_RANKING = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SLIDE_QRELS = {1: 2, 3: 3, 5: 3, 6: 1}  # 4 relevantes, nas posições 1,3,5,6


def test_precision_recall_f1_slide_example():
    relevant = M.binary_relevant(SLIDE_QRELS)
    assert relevant == {1, 3, 5, 6}
    # 3 dos 5 primeiros são relevantes.
    assert M.precision_at_k(SLIDE_RANKING, relevant, 5) == 3 / 5
    # 3 dos 4 relevantes aparecem nos 5 primeiros.
    assert M.recall_at_k(SLIDE_RANKING, relevant, 5) == 3 / 4
    expected_f1 = 2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4))
    assert math.isclose(M.f1_at_k(SLIDE_RANKING, relevant, 5), expected_f1)


def test_reciprocal_rank_slide_example():
    # O primeiro documento já é relevante.
    assert M.reciprocal_rank(SLIDE_RANKING, M.binary_relevant(SLIDE_QRELS)) == 1.0


def test_average_precision_slide_example():
    relevant = M.binary_relevant(SLIDE_QRELS)
    # AP@3: relevantes nas posições 1 e 3 -> (1/1 + 2/3) / 4
    expected = (1.0 + 2.0 / 3.0) / 4
    assert math.isclose(M.average_precision(SLIDE_RANKING, relevant, cutoff=3), expected)
    # AP completo: posições 1, 3, 5, 6 -> (1/1 + 2/3 + 3/5 + 4/6) / 4
    expected_full = (1.0 + 2 / 3 + 3 / 5 + 4 / 6) / 4
    assert math.isclose(M.average_precision(SLIDE_RANKING, relevant), expected_full)


def test_ndcg_slide_example():
    # Aqui os graus já são níveis de relevância convencionais (maior = melhor),
    # então usamos o mapeamento RAW.
    value = M.ndcg_at_k(SLIDE_RANKING, SLIDE_QRELS, 5, M.GainMapping.RAW)
    # rel@5 = [2,0,3,0,3] -> ganhos [3,0,7,0,7]
    dcg = 3 / math.log2(2) + 7 / math.log2(4) + 7 / math.log2(6)
    # ideal = [3,3,2,1] -> ganhos [7,7,3,1]
    idcg = 7 / math.log2(2) + 7 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
    assert math.isclose(value, dcg / idcg)
    assert math.isclose(value, 0.6899, abs_tol=1e-4)


def test_interpolated_curve_is_monotonically_non_increasing():
    relevant = M.binary_relevant(SLIDE_QRELS)
    curve = M.interpolated_precision_recall(SLIDE_RANKING, relevant)
    assert len(curve) == 11
    # A interpolação P(r) = max_{r' >= r} P(r') é não crescente por construção.
    assert all(curve[i] >= curve[i + 1] - 1e-12 for i in range(len(curve) - 1))
    assert curve[0] == 1.0  # há um relevante na posição 1


def test_cranfield_inverted_gain_mapping():
    # No Cranfield o grau 1 é o MELHOR; o mapeamento padrão deve invertê-lo.
    assert M.grade_to_gain_level(1) == 4
    assert M.grade_to_gain_level(4) == 1
    assert M.grade_to_gain_level(-1) == 0
    assert M.grade_to_gain_level(0) == 0
    # O mapeamento RAW preserva o grau.
    assert M.grade_to_gain_level(1, M.GainMapping.RAW) == 1
    assert M.grade_to_gain_level(4, M.GainMapping.RAW) == 4


def test_ndcg_rewards_putting_best_graded_document_first():
    # Documento 100 tem grau 1 (o melhor) e documento 200 grau 4 (o pior).
    qrels = {100: 1, 200: 4}
    good = M.ndcg_at_k([100, 200], qrels, 10)
    bad = M.ndcg_at_k([200, 100], qrels, 10)
    assert good > bad, "o mapeamento invertido deve premiar o grau 1 no topo"


# ======================================================================
# 2. Implementação de referência ingênua vs. implementação vetorizada
# ======================================================================
def naive_vsm_scores(index, query_text: str, log_base: float = 10.0) -> np.ndarray:
    """TF-IDF + cosseno escrito diretamente a partir da fórmula, sem scipy."""
    log = math.log(log_base)
    num_docs = index.num_docs

    query_counts = index.query_vector(query_text)
    query_weights: dict[str, float] = {}
    for term, frequency in query_counts.items():
        column = index.term_id(term)
        if column is None:
            continue
        idf = math.log(num_docs / index.df[column]) / log
        query_weights[term] = (1.0 + math.log(frequency) / log) * idf
    query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

    scores = np.zeros(num_docs)
    if query_norm == 0.0:
        return scores

    dense = index.tf.toarray()
    for row in range(num_docs):
        doc_weights: dict[str, float] = {}
        for column in np.nonzero(dense[row])[0]:
            frequency = dense[row, column]
            idf = math.log(num_docs / index.df[column]) / log
            doc_weights[index.terms[column]] = (1.0 + math.log(frequency) / log) * idf
        doc_norm = math.sqrt(sum(w * w for w in doc_weights.values()))
        if doc_norm == 0.0:
            continue
        dot = sum(w * doc_weights.get(term, 0.0) for term, w in query_weights.items())
        scores[row] = dot / (doc_norm * query_norm)
    return scores


def naive_bm25_scores(index, query_text: str, k1: float, b: float) -> np.ndarray:
    """BM25 escrito termo a termo, documento a documento, sem vetorização."""
    num_docs = index.num_docs
    avgdl = index.avg_doc_length
    dense = index.tf.toarray()
    query_counts = index.query_vector(query_text)

    scores = np.zeros(num_docs)
    for row in range(num_docs):
        doc_length = index.doc_lengths[row]
        total = 0.0
        for term, query_frequency in query_counts.items():
            column = index.term_id(term)
            if column is None:
                continue
            tf = dense[row, column]
            if tf == 0:
                continue
            n_i = index.df[column]
            idf = math.log(1.0 + (num_docs - n_i + 0.5) / (n_i + 0.5))
            denominator = tf + k1 * (1.0 - b + b * doc_length / avgdl)
            total += query_frequency * idf * (tf * (k1 + 1.0)) / denominator
        scores[row] = total
    return scores


def _small_index():
    """Índice sobre um subconjunto de 120 documentos (mantém os testes rápidos)."""
    collection = load_cranfield(DATA_DIR, download=False)
    documents = collection.documents[:120]
    config = build_configs(DATA_DIR / "stopwords_en.txt")["stop_stem"]
    index = build_index([d.text for d in documents], [d.doc_id for d in documents], config)
    return collection, index


def test_vsm_matches_naive_reference():
    collection, index = _small_index()
    model = VectorSpaceModel(index)
    for query in collection.queries[:5]:
        fast = model.score(query.text)
        slow = naive_vsm_scores(index, query.text)
        assert np.allclose(fast, slow, atol=1e-10), f"divergência na consulta {query.query_id}"


def test_bm25_matches_naive_reference():
    collection, index = _small_index()
    for k1, b in [(0.5, 0.0), (1.2, 0.75), (2.0, 1.0)]:
        model = BM25(index, k1=k1, b=b)
        for query in collection.queries[:5]:
            fast = model.score(query.text)
            slow = naive_bm25_scores(index, query.text, k1, b)
            assert np.allclose(fast, slow, atol=1e-10), f"divergência em k1={k1}, b={b}"


def test_explain_reproduces_the_score():
    """O score total do ranking deve bater com a soma das contribuições."""
    collection, index = _small_index()
    query = collection.queries[0]

    vsm = VectorSpaceModel(index)
    top_doc, top_score = vsm.rank(query.text, top_k=1)[0]
    explanation = vsm.explain(query.text, top_doc)
    assert math.isclose(explanation["score"], top_score, abs_tol=1e-10)
    assert math.isclose(
        sum(t["contribution"] for t in explanation["terms"]), top_score, abs_tol=1e-10
    )

    bm25 = BM25(index)
    top_doc, top_score = bm25.rank(query.text, top_k=1)[0]
    explanation = bm25.explain(query.text, top_doc)
    assert math.isclose(explanation["score"], top_score, abs_tol=1e-10)


def test_cosine_similarity_is_bounded():
    """0 <= sim(d,q) <= 1, como afirmado no slide 34 da Aula 04."""
    collection, index = _small_index()
    model = VectorSpaceModel(index)
    for query in collection.queries[:5]:
        scores = model.score(query.text)
        assert scores.min() >= -1e-12
        assert scores.max() <= 1.0 + 1e-12


def test_bm25_with_k1_zero_is_binary():
    """Com k1 = 0 o BM25 ignora a frequência: o score vira a soma dos IDF."""
    collection, index = _small_index()
    model = BM25(index, k1=0.0, b=0.0)
    query = collection.queries[0]
    scores = model.score(query.text)

    counts = index.query_vector(query.text)
    expected = np.zeros(index.num_docs)
    for term, query_frequency in counts.items():
        column = index.term_id(term)
        if column is None:
            continue
        rows, _ = index.postings(term)
        expected[rows] += query_frequency * model.idf[column]
    assert np.allclose(scores, expected, atol=1e-10)


def test_ranking_tie_break_is_deterministic():
    """Empates de score são desfeitos pelo doc_id crescente."""
    collection, index = _small_index()
    model = BM25(index)
    ranking = model.rank(collection.queries[0].text)
    for (id_a, score_a), (id_b, score_b) in zip(ranking, ranking[1:]):
        assert score_a > score_b or (math.isclose(score_a, score_b) and id_a < id_b)


# ======================================================================
# 3. Verificação cruzada com o scikit-learn
# ======================================================================
def test_vsm_agrees_with_sklearn_tfidf():
    """O ranking deve ser fortemente correlacionado com o do scikit-learn.

    Não é esperada igualdade exata: o 'TfidfVectorizer' usa
    'idf = ln(N/df) + 1', enquanto a Aula 04 define 'idf = log10(N/n)'.
    O "+1" do sklearn impede que termos presentes em toda a coleção sejam
    anulados, o que muda ligeiramente a ordenação.
    """
    from scipy.stats import spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    collection, index = _small_index()
    documents = collection.documents[:120]
    config = index.preprocessor

    vectorizer = TfidfVectorizer(
        analyzer=config,  # exatamente o mesmo pré-processamento
        sublinear_tf=True,  # tf = 1 + ln(f)
        smooth_idf=False,  # idf = ln(N/df) + 1
        norm="l2",
    )
    doc_matrix = vectorizer.fit_transform([d.text for d in documents])

    model = VectorSpaceModel(index)
    for query in collection.queries[:5]:
        ours = model.score(query.text)
        theirs = cosine_similarity(vectorizer.transform([query.text]), doc_matrix).ravel()
        correlation = spearmanr(ours, theirs).statistic
        assert correlation > 0.95, f"correlação baixa ({correlation:.3f})"


# ======================================================================
# 4. Integridade da coleção
# ======================================================================
def test_query_qrel_alignment():
    """As consultas devem ser renumeradas por POSIÇÃO, não pelo campo .I."""
    collection = load_cranfield(DATA_DIR, download=False)
    # A terceira consulta do arquivo tem .I == 4: prova de que os ids
    # originais não são sequenciais e precisam ser renumerados.
    assert collection.queries[2].original_id == 4
    assert collection.queries[2].query_id == 3
    # Todas as 225 consultas têm ao menos um documento relevante.
    assert all(collection.relevant_docs(q.query_id) for q in collection.queries)


def test_document_text_does_not_duplicate_title():
    collection = load_cranfield(DATA_DIR, download=False)
    document = collection.document_by_id(184)
    assert document.text.startswith("scale models for thermo-aeroelastic research")
    # O título aparece uma única vez, e não duas.
    assert document.text.count("scale models for thermo-aeroelastic research") == 1


def _main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FALHOU  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok      {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
