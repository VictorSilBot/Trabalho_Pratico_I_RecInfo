"""Reformulações manuais de consultas (requisito 8).

Cinco consultas foram escolhidas e reescritas **manualmente**, cada uma
ilustrando um tipo diferente de operação de reformulação. Os julgamentos de
relevância NÃO foram usados para escolher as edições -- as reformulações
partem apenas da leitura da consulta e do conhecimento do domínio
(aerodinâmica), e a hipótese de cada uma foi registrada ANTES de rodar os
modelos, para que a análise não seja construída depois do resultado.

O campo ``hypothesis`` é justamente esse registro prévio.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reformulation:
    query_id: int
    operation: str
    original: str
    modified: str
    hypothesis: str


REFORMULATIONS: list[Reformulation] = [
    Reformulation(
        query_id=219,
        operation="sinônimo (troca mínima de uma palavra)",
        original=(
            "what are the general effects on flow fields when the reynolds "
            "number is small ."
        ),
        modified=(
            "what are the general effects on flow fields when the reynolds "
            "number is low ."
        ),
        hypothesis=(
            "A literatura usa consagradamente a expressão 'low Reynolds number', "
            "enquanto a consulta diz 'the reynolds number is small'. Como ambos "
            "os modelos fazem casamento puramente léxico, 'small' não casa com "
            "'low' e a evidência mais discriminativa da consulta se perde. "
            "Trocar uma única palavra deve melhorar o ranking sem alterar o "
            "significado -- é o caso mais puro de incompatibilidade de "
            "vocabulário (Aula 04, slide 39)."
        ),
    ),
    Reformulation(
        query_id=69,
        operation="mais específica (acréscimo de termos discriminativos)",
        original=(
            "what is known regarding asymptotic solutions to the exact "
            "boundary layer equations ."
        ),
        modified=(
            "asymptotic series solutions for the laminar compressible "
            "boundary layer equations"
        ),
        hypothesis=(
            "Os termos de conteúdo da consulta original são quase todos de "
            "baixo poder discriminativo na coleção ('solution', 'boundary', "
            "'layer', 'equation' ocorrem em 400-470 dos 1400 documentos), e "
            "ainda são diluídos pelo enquadramento interrogativo 'what is "
            "known regarding'. Acrescentar 'series', 'laminar' e "
            "'compressible' deve concentrar o ranking nos artigos de solução "
            "em série da camada limite laminar."
        ),
    ),
    Reformulation(
        query_id=109,
        operation="mais específica (consulta curta expandida)",
        original="panels subjected to aerodynamic heating .",
        modified=(
            "thermal stress buckling and flutter of structural panels "
            "subjected to aerodynamic heating in high speed flight"
        ),
        hypothesis=(
            "Com apenas 4 termos após o pré-processamento, a consulta é curta "
            "demais para separar os documentos: 'heat' e 'aerodynam' são "
            "frequentes e 'panel' sozinho traz painéis de qualquer natureza. "
            "Explicitar a consequência física de interesse (tensão térmica, "
            "flambagem, flutter) deve aproximar a consulta do vocabulário dos "
            "documentos efetivamente relevantes."
        ),
    ),
    Reformulation(
        query_id=171,
        operation="mais genérica (remoção de termos pouco informativos)",
        original=(
            "has a comparison been made between interference-free drag "
            "measurements using free-flight models and similar measurements "
            "made in a low-blockage wind tunnel ."
        ),
        modified="free-flight drag measurements wind tunnel interference blockage",
        hypothesis=(
            "A consulta original tem 17 tokens, vários deles genéricos e "
            "repetidos ('made' 2x, 'measur' 2x, 'use', 'similar', "
            "'comparison'). No modelo vetorial esses termos inflam a norma do "
            "vetor de consulta e diluem o peso de 'blockag' (df=8) e "
            "'interfer' (df=46). Enxugar a consulta para o seu núcleo "
            "conceitual deve beneficiar sobretudo o modelo vetorial, que é o "
            "mais prejudicado pela diluição."
        ),
    ),
    Reformulation(
        query_id=15,
        operation="expansão de vocabulário técnico (sinônimo morfológico)",
        original="material properties of photoelastic materials .",
        modified=(
            "physical properties of plastics for photo-thermoelastic stress "
            "investigation"
        ),
        hypothesis=(
            "O termo 'photoelastic' ocorre em um único documento da coleção, "
            "mas os dois documentos relevantes usam a variante "
            "'photo-thermoelastic'. Como o tokenizador quebra o hífen, essa "
            "variante gera os tokens 'photo' e 'thermoelast', que não casam "
            "com o radical 'photoelast'. Usar a forma empregada pelos autores "
            "deve resolver o descasamento, ilustrando que nem o stemming nem "
            "o IDF conseguem contornar uma diferença de vocabulário."
        ),
    ),
]


def by_query_id() -> dict[int, Reformulation]:
    return {r.query_id: r for r in REFORMULATIONS}
