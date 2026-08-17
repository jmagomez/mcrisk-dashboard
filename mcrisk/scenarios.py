"""
Analise de cenarios e estresse sobre uma especificacao ja montada.

Duas perguntas diferentes, dois mecanismos diferentes -- e confundi-los e o erro
comum nesta area:

1. CONDICIONAL ("e nos casos em que o cambio passou de 6?")
   Nao ha o que re-simular. A resposta ja esta na amostra: basta filtrar as
   iteracoes que satisfazem a condicao e resumir a saida nelas. Continua sendo o
   mesmo modelo, com as mesmas probabilidades. O que muda e o recorte.
   O risco aqui e amostral: condicionar em cauda estreita deixa poucas
   iteracoes, e a estatistica do subconjunto fica ruidosa. Por isso `conditional`
   devolve SEMPRE a contagem, e avisa quando ela e pequena.

2. ESTRESSE ("e se a inflacao tivesse media 12% em vez de 5%?")
   Aqui o modelo mudou: a distribuicao de entrada e outra. Exige nova simulacao.
   O resultado NAO e "a probabilidade do cenario": e o resultado condicional a
   um mundo diferente, cuja probabilidade a ferramenta nao sabe e nao estima.

A distincao importa porque so a primeira preserva as probabilidades do modelo.
Um numero de estresse apresentado como se fosse percentil da distribuicao
original e leitura errada -- e e por isso que `stress` devolve os dois lados
rotulados, nunca um numero solto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

from .engine import SimulationResult, SimulationSpec, Variable, run
from .summary import conditional_value_at_risk, describe, value_at_risk

# Abaixo disso a estatistica condicional e ruido: 30 pontos nao estimam um P95.
MIN_ITERACOES_CONDICIONAL = 200


@dataclass
class CenarioCondicional:
    nome: str
    n: int
    fracao: float
    resumo: Dict[str, float]
    avisos: List[str] = field(default_factory=list)


def conditional(
    result: SimulationResult,
    condicao: Callable[[Dict[str, np.ndarray]], np.ndarray],
    nome: str = "cenario",
) -> CenarioCondicional:
    """Resumo da saida nas iteracoes que satisfazem `condicao`.

    `condicao` recebe um dicionario {nome_da_variavel: vetor} e devolve mascara
    booleana. Ex.: lambda v: v["cambio"] > 6.0
    """
    dados = {nm: result.inputs[:, j] for j, nm in enumerate(result.names)}
    mask = np.asarray(condicao(dados), dtype=bool)
    if mask.shape != (result.n,):
        raise ValueError(
            f"a condicao devolveu mascara de forma {mask.shape}, esperado {(result.n,)}"
        )
    y = result.output[mask]
    y = y[np.isfinite(y)]
    avisos: List[str] = []
    n = int(y.size)
    frac = n / result.n if result.n else 0.0
    if n == 0:
        avisos.append(
            "Nenhuma iteracao satisfez a condicao. O cenario e possivel no modelo, "
            "mas nao apareceu nesta amostra -- aumente as iteracoes ou relaxe o corte."
        )
        return CenarioCondicional(nome, 0, 0.0, {}, avisos)
    if n < MIN_ITERACOES_CONDICIONAL:
        avisos.append(
            f"Apenas {n} iteracoes no cenario ({frac:.3%} da amostra). Percentis e "
            f"CVaR calculados sobre tao poucos pontos tem erro grande; trate-os como "
            f"indicacao de ordem de grandeza."
        )
    resumo = dict(describe(y))
    resumo["var_95"] = value_at_risk(y, 0.95)
    resumo["cvar_95"] = conditional_value_at_risk(y, 0.95)
    return CenarioCondicional(nome, n, frac, resumo, avisos)


@dataclass
class CenarioEstresse:
    nome: str
    resumo_base: Dict[str, float]
    resumo_estressado: Dict[str, float]
    delta: Dict[str, float]
    delta_pct: Dict[str, float]
    avisos: List[str] = field(default_factory=list)


def _resumir(y: np.ndarray) -> Dict[str, float]:
    y = y[np.isfinite(y)]
    d = dict(describe(y))
    d["var_95"] = value_at_risk(y, 0.95)
    d["cvar_95"] = conditional_value_at_risk(y, 0.95)
    return d


def aplicar_overrides(
    spec: SimulationSpec, overrides: Dict[str, Dict[str, Any]]
) -> SimulationSpec:
    """Copia a especificacao trocando parametros de variaveis nomeadas.

    Nao altera `spec` no lugar: cenario que muda o objeto original contamina a
    base de comparacao, e o erro so aparece quando os numeros ja foram lidos.
    """
    nomes = {v.name for v in spec.variables}
    desconhecidas = set(overrides) - nomes
    if desconhecidas:
        raise KeyError(
            f"variaveis inexistentes no modelo: {', '.join(sorted(desconhecidas))}"
        )
    novas: List[Variable] = []
    for v in spec.variables:
        if v.name not in overrides:
            novas.append(v)
            continue
        ov = overrides[v.name]
        params = dict(v.params)
        params.update({k: val for k, val in ov.items() if k != "dist_key"})
        novas.append(
            Variable(
                name=v.name,
                label=v.label,
                dist_key=ov.get("dist_key", v.dist_key),
                params=params,
                values=v.values,
                probs=v.probs,
                data=v.data,
            )
        )
    return SimulationSpec(
        variables=novas,
        formula=spec.formula,
        iterations=spec.iterations,
        method=spec.method,
        seed=spec.seed,
        correlation=spec.correlation,
        spearman_adjust=spec.spearman_adjust,
        dependence=spec.dependence,
        copula_df=spec.copula_df,
    )


def stress(
    spec: SimulationSpec,
    overrides: Dict[str, Dict[str, Any]],
    nome: str = "estresse",
    base: SimulationResult | None = None,
) -> CenarioEstresse:
    """Re-simula com parametros trocados e compara com a base.

    A mesma seed e usada nos dois lados de proposito: parte da diferenca
    observada seria ruido de amostragem se as sequencias fossem distintas, e
    a pergunta do estresse e sobre o efeito da MUDANCA, nao sobre ruido.
    """
    base_res = base if base is not None else run(spec)
    est_res = run(aplicar_overrides(spec, overrides))
    rb, re_ = _resumir(base_res.output), _resumir(est_res.output)
    chaves = [k for k in rb if k in re_ and isinstance(rb[k], (int, float))]
    delta = {k: float(re_[k] - rb[k]) for k in chaves}
    delta_pct = {
        k: (float((re_[k] - rb[k]) / rb[k] * 100.0) if rb[k] not in (0, 0.0) else float("nan"))
        for k in chaves
    }
    avisos = [
        "Cenario de estresse: a distribuicao de entrada foi alterada. O resultado "
        "e condicional a esse mundo hipotetico e NAO tem a probabilidade do modelo "
        "original -- nao o leia como percentil da distribuicao base."
    ]
    return CenarioEstresse(nome, rb, re_, delta, delta_pct, avisos)


def tabela_estresse(cenarios: List[CenarioEstresse], metrica: str = "media"):
    """DataFrame comparando varios cenarios em uma metrica."""
    import pandas as pd

    linhas = []
    for c in cenarios:
        linhas.append({
            "cenario": c.nome,
            "base": c.resumo_base.get(metrica, float("nan")),
            "estressado": c.resumo_estressado.get(metrica, float("nan")),
            "delta": c.delta.get(metrica, float("nan")),
            "delta_%": c.delta_pct.get(metrica, float("nan")),
        })
    return pd.DataFrame(linhas)
