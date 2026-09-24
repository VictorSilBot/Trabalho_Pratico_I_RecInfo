"""Download e parsing da coleção Cranfield (1400 documentos, 225 consultas).

Arquivos originais (University of Glasgow):
    http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz

Conteúdo do tarball:
    cran.all.1400 -- documentos, formato SMART (.I .T .A .B .W)
    cran.qry      -- consultas, formato SMART (.I .W)
    cranqrel      -- julgamentos de relevância: <query> <doc> <grau>

ATENÇÃO -- detalhe que quebra implementações ingênuas:
    Os identificadores ``.I`` de ``cran.qry`` NÃO são sequenciais
    (001, 002, 004, 008, ..., 365), mas o arquivo ``cranqrel`` numera as
    consultas de 1 a 225 pela POSIÇÃO no arquivo. Portanto a consulta de
    número ``n`` no qrel corresponde à ``n``-ésima consulta de ``cran.qry``,
    e não à consulta cujo ``.I`` vale ``n``. Este módulo renumera as
    consultas por posição (mesma convenção adotada pelo ``ir_datasets``) e
    preserva o identificador original em ``Query.original_id``.

Escala de relevância do Cranfield (ver ``cranqrel.readme``), na qual o grau
1 é o MELHOR e o 4 é o PIOR:
    1 -- resposta completa para a pergunta
    2 -- alto grau de relevância
    3 -- útil como background / sugestão de método
    4 -- interesse mínimo (e.g. histórico)
   -1 -- sem interesse (equivale ao código 5 de Cleverdon)
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CRANFIELD_URL = "http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"

# Verificado no download de referência usado nos experimentos deste trabalho.
CRANFIELD_SHA256 = "8c48a8412e4a7e6e0dd5af99c959f8d75fdf88135602adc1a56aa22770652d64"

_EXPECTED_FILES = ("cran.all.1400", "cran.qry", "cranqrel")

# Graus considerados relevantes para as métricas binárias, conforme o
# enunciado: "considere como relevante todo documento com grau de relevância
# maior ou igual a 1". O grau -1 e os documentos não julgados são tratados
# como não relevantes.
MIN_BINARY_RELEVANT_GRADE = 1


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    author: str
    bib: str
    abstract: str

    @property
    def text(self) -> str:
        """Texto efetivamente indexado: título + resumo, sem duplicar o título.

        Em 1395 dos 1400 documentos o campo ``.W`` já começa com o texto do
        campo ``.T``; repetir o título dobraria artificialmente o ``tf`` dos
        seus termos. Só concatenamos o título quando ele ainda não é prefixo
        do resumo (e quando o resumo está vazio).
        """
        abstract = self.abstract.strip()
        title = self.title.strip()
        if not abstract:
            return title
        if _alpha_norm(abstract).startswith(_alpha_norm(title)[:50]):
            return abstract
        return f"{title} {abstract}"


@dataclass(frozen=True)
class Query:
    query_id: int  # posição no arquivo (1..225) -- é a chave usada no qrel
    original_id: int  # valor do campo .I no cran.qry (não sequencial)
    text: str


def _alpha_norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z]+", s.lower()))


def download_cranfield(data_dir: Path, force: bool = False) -> Path:
    """Baixa e extrai o tarball do Cranfield em ``data_dir/cran``.

    Se os arquivos já existirem, nada é baixado novamente. O SHA-256 do
    tarball é conferido e uma divergência gera apenas um aviso (a coleção é
    servida por um host acadêmico antigo, que já trocou o empacotamento).
    """
    target = Path(data_dir) / "cran"
    target.mkdir(parents=True, exist_ok=True)

    if not force and all((target / name).exists() for name in _EXPECTED_FILES):
        return target

    with urllib.request.urlopen(CRANFIELD_URL, timeout=120) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != CRANFIELD_SHA256:
        print(
            f"[aviso] SHA-256 do cran.tar.gz é {digest}, "
            f"esperado {CRANFIELD_SHA256}. Prosseguindo mesmo assim."
        )

    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        tar.extractall(target, filter="data")

    missing = [name for name in _EXPECTED_FILES if not (target / name).exists()]
    if missing:
        raise RuntimeError(f"Arquivos ausentes após a extração: {missing}")
    return target


def _split_smart_records(raw: str) -> list[str]:
    """Divide um arquivo no formato SMART nos blocos iniciados por ``.I``."""
    return re.split(r"^\.I ", raw, flags=re.MULTILINE)[1:]


def _parse_smart_fields(block: str, field_markers: tuple[str, ...]) -> tuple[int, dict[str, str]]:
    """Extrai o id (primeira linha) e os campos de um bloco SMART.

    Uma linha só é tratada como marcador de campo quando, após ``strip()``,
    ela é exatamente igual ao marcador (e.g. ``.W``). Isso evita interpretar
    como campo linhas do corpo que porventura comecem com ponto.
    """
    lines = block.split("\n")
    record_id = int(lines[0].strip())
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped in field_markers:
            current = stripped[1:]
            fields.setdefault(current, [])
        elif current is not None:
            fields[current].append(line)
    flat = {key: " ".join(" ".join(value).split()) for key, value in fields.items()}
    return record_id, flat


def load_documents(cran_dir: Path) -> list[Document]:
    raw = (Path(cran_dir) / "cran.all.1400").read_text(encoding="utf-8", errors="replace")
    documents: list[Document] = []
    for block in _split_smart_records(raw):
        doc_id, fields = _parse_smart_fields(block, (".T", ".A", ".B", ".W"))
        documents.append(
            Document(
                doc_id=doc_id,
                title=fields.get("T", ""),
                author=fields.get("A", ""),
                bib=fields.get("B", ""),
                abstract=fields.get("W", ""),
            )
        )
    if len(documents) != 1400:
        raise RuntimeError(f"Esperados 1400 documentos, obtidos {len(documents)}")
    return documents


def load_queries(cran_dir: Path) -> list[Query]:
    """Carrega as consultas RENUMERANDO-AS por posição (1..225).

    Ver a nota no topo do módulo: o ``cranqrel`` usa a posição, não o ``.I``.
    """
    raw = (Path(cran_dir) / "cran.qry").read_text(encoding="utf-8", errors="replace")
    queries: list[Query] = []
    for position, block in enumerate(_split_smart_records(raw), start=1):
        original_id, fields = _parse_smart_fields(block, (".W",))
        queries.append(
            Query(query_id=position, original_id=original_id, text=fields.get("W", ""))
        )
    if len(queries) != 225:
        raise RuntimeError(f"Esperadas 225 consultas, obtidas {len(queries)}")
    return queries


def load_qrels(cran_dir: Path) -> dict[int, dict[int, int]]:
    """Retorna ``{query_id: {doc_id: grau}}`` com os graus brutos do Cranfield.

    Os graus são mantidos como estão no arquivo (incluindo ``-1``); a
    conversão para relevância binária ou para ganho graduado é feita em
    :mod:`ir_tp1.metrics`, para deixar a política de conversão explícita e
    em um único lugar.
    """
    raw = (Path(cran_dir) / "cranqrel").read_text(encoding="utf-8", errors="replace")
    qrels: dict[int, dict[int, int]] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        query_id, doc_id, grade = (int(p) for p in parts)
        qrels.setdefault(query_id, {})[doc_id] = grade
    return qrels


@dataclass(frozen=True)
class Cranfield:
    documents: list[Document]
    queries: list[Query]
    qrels: dict[int, dict[int, int]]

    @property
    def doc_ids(self) -> list[int]:
        return [d.doc_id for d in self.documents]

    def relevant_docs(self, query_id: int) -> set[int]:
        """Documentos relevantes para as métricas binárias (grau >= 1)."""
        judgements = self.qrels.get(query_id, {})
        return {
            doc_id
            for doc_id, grade in judgements.items()
            if grade >= MIN_BINARY_RELEVANT_GRADE
        }

    def document_by_id(self, doc_id: int) -> Document:
        return self.documents[doc_id - 1]

    def query_by_id(self, query_id: int) -> Query:
        return self.queries[query_id - 1]


def load_cranfield(data_dir: Path | str, download: bool = True) -> Cranfield:
    data_dir = Path(data_dir)
    cran_dir = data_dir / "cran"
    if download:
        cran_dir = download_cranfield(data_dir)
    collection = Cranfield(
        documents=load_documents(cran_dir),
        queries=load_queries(cran_dir),
        qrels=load_qrels(cran_dir),
    )
    _sanity_check(collection)
    return collection


def _sanity_check(collection: Cranfield) -> None:
    """Confere invariantes da coleção; falha cedo e alto em caso de problema."""
    if sorted(collection.qrels) != list(range(1, 226)):
        raise RuntimeError("O qrel deveria cobrir exatamente as consultas 1..225")
    empty = [q.query_id for q in collection.queries if not q.text.strip()]
    if empty:
        raise RuntimeError(f"Consultas sem texto: {empty}")
    without_relevant = [
        q.query_id for q in collection.queries if not collection.relevant_docs(q.query_id)
    ]
    if without_relevant:
        raise RuntimeError(f"Consultas sem documento relevante: {without_relevant}")
    # Verificação do alinhamento posicional consulta <-> qrel: a consulta 1
    # trata de leis de similaridade para modelos aeroelásticos e o documento
    # 184 ("scale models for thermo-aeroelastic research") é julgado relevante.
    if 184 not in collection.relevant_docs(1):
        raise RuntimeError("Alinhamento consulta/qrel parece incorreto")
