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

Uma terceira pergunta, na secao final: quais ENTRADAS levam ao cenario. As duas
acima olham para a saida; `scenario_significance` olha para as entradas.
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


# ===========================================================================
# Significancia de cenario (analise de mediana condicional)
# ===========================================================================
#
# `conditional` acima responde "como fica a SAIDA quando o cenario ocorre?".
# Esta secao responde a pergunta inversa e mais acionavel: "QUAIS ENTRADAS
# levaram ate o cenario?".
#
# O criterio compara, para cada entrada, a mediana dos valores nas iteracoes
# que atingiram o alvo (mediana do subconjunto) com a mediana em todas as
# iteracoes (mediana geral), medindo a diferenca em desvios-padrao:
#
#     significancia = (mediana_subconjunto - mediana_geral) / desvio_padrao
#
# Se a entrada nao influencia o cenario, condicionar nao desloca a mediana
# dela e a significancia fica em torno de zero. Usa-se mediana, e nao media,
# porque o subconjunto costuma ser uma cauda: a media ali e dominada por
# poucos valores extremos e balanca de rodada para rodada.
#
# LIMIAR. O @RISK trata como insignificante toda entrada com significancia
# absoluta abaixo de 0,5, e o mesmo padrao e adotado aqui. E convencao, nao
# teste de hipotese: nao ha p-valor associado, e o limiar nao se ajusta ao
# tamanho do subconjunto. Com poucas iteracoes no recorte, ruido sozinho
# produz significancias que passam de 0,5 - por isso a funcao avisa.

LIMIAR_SIGNIFICANCIA = 0.5


@dataclass
class SignificanciaCenario:
    """Ranking das entradas que levam a um cenario da saida."""

    nome: str
    n: int
    fracao: float
    names: List[str]
    labels: List[str]
    significancia: np.ndarray  # (k,) em desvios-padrao
    mediana_subconjunto: np.ndarray
    mediana_geral: np.ndarray
    desvio: np.ndarray
    limiar: float = LIMIAR_SIGNIFICANCIA
    avisos: List[str] = field(default_factory=list)

    def ordem(self) -> List[int]:
        return list(np.argsort(-np.abs(np.nan_to_num(self.significancia, nan=0.0))))

    def significativas(self) -> List[int]:
        return [
            i
            for i in self.ordem()
            if abs(float(self.significancia[i])) >= self.limiar
        ]

    def as_records(self) -> List[Dict[str, float]]:
        return [
            {
                "variavel": self.labels[i],
                "significancia": float(self.significancia[i]),
                "mediana_no_cenario": float(self.mediana_subconjunto[i]),
                "mediana_geral": float(self.mediana_geral[i]),
                "significativa": bool(
                    abs(float(self.significancia[i])) >= self.limiar
                ),
            }
            for i in range(len(self.names))
        ]


def scenario_significance(
    result: SimulationResult,
    alvo: Callable[[np.ndarray], np.ndarray] | None = None,
    percentil: float | None = None,
    cauda: str = "superior",
    nome: str = "cenario",
    limiar: float = LIMIAR_SIGNIFICANCIA,
    n_minimo: int = 100,
) -> SignificanciaCenario:
    """Quais entradas levam a saida ate o cenario alvo.

    O alvo pode vir de duas formas, e exatamente uma delas deve ser usada:

    - `alvo`: funcao que recebe o vetor da saida e devolve mascara booleana.
      Total liberdade ("saida negativa", "saida entre 10 e 20").
    - `percentil` + `cauda`: recorta a cauda superior (saida acima do
      percentil) ou inferior. E o caso comum e evita escrever a lambda.

    O desvio-padrao no denominador e o da entrada na amostra INTEIRA, nao no
    subconjunto. Usar o desvio do subconjunto tornaria a medida circular:
    condicionar estreita a distribuicao da entrada, entao o denominador
    encolheria justamente para as entradas mais influentes, inflando a
    significancia delas.
    """
    if (alvo is None) == (percentil is None):
        raise ValueError("informe exatamente um entre `alvo` e `percentil`")

    y = np.asarray(result.output, dtype=float)
    X = np.asarray(result.inputs, dtype=float)
    finito = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    avisos: List[str] = []
    if not finito.all():
        avisos.append(
            f"{int((~finito).sum())} iteracoes descartadas por conterem valores "
            f"nao finitos."
        )
    y_ok, X_ok = y[finito], X[finito]
    n_total = y_ok.shape[0]
    if n_total == 0:
        raise ValueError("nenhuma iteracao finita para analisar")

    if percentil is not None:
        if not 0.0 < percentil < 100.0:
            raise ValueError(f"percentil deve estar em (0, 100), recebido {percentil}")
        corte = float(np.percentile(y_ok, percentil))
        if cauda == "superior":
            mascara = y_ok > corte
            nome = f"{nome}: saida acima do P{percentil:g}"
        elif cauda == "inferior":
            mascara = y_ok < corte
            nome = f"{nome}: saida abaixo do P{percentil:g}"
        else:
            raise ValueError("cauda deve ser 'superior' ou 'inferior'")
    else:
        mascara = np.asarray(alvo(y_ok), dtype=bool)
        if mascara.shape != y_ok.shape:
            raise ValueError(
                f"a mascara do alvo tem forma {mascara.shape}, esperado "
                f"{y_ok.shape}"
            )

    n = int(mascara.sum())
    fracao = n / n_total
    k = X_ok.shape[1]
    if n == 0:
        avisos.append(
            "Nenhuma iteracao atingiu o cenario. Ou o alvo e impossivel no "
            "modelo, ou e raro demais para esta quantidade de iteracoes."
        )
        vazio = np.full(k, np.nan)
        return SignificanciaCenario(
            nome=nome, n=0, fracao=0.0, names=list(result.names),
            labels=list(result.labels), significancia=vazio,
            mediana_subconjunto=vazio, mediana_geral=vazio, desvio=vazio,
            limiar=limiar, avisos=avisos,
        )
    if n < n_minimo:
        avisos.append(
            f"So {n} iteracoes atingiram o cenario ({fracao:.2%}). Abaixo de "
            f"{n_minimo} a mediana do subconjunto e instavel e o limiar de "
            f"{limiar:g} passa a ser cruzado por ruido de amostragem. Rode com "
            f"mais iteracoes antes de agir sobre este ranking."
        )

    mediana_geral = np.median(X_ok, axis=0)
    mediana_sub = np.median(X_ok[mascara], axis=0)
    desvio = X_ok.std(axis=0, ddof=1)

    significancia = np.full(k, np.nan)
    vivos = desvio > 0
    significancia[vivos] = (mediana_sub[vivos] - mediana_geral[vivos]) / desvio[vivos]
    significancia[~vivos] = 0.0
    if not vivos.all():
        avisos.append(
            "Variaveis constantes na amostra (significancia 0 por construcao): "
            + ", ".join(result.labels[j] for j in np.flatnonzero(~vivos))
        )

    return SignificanciaCenario(
        nome=nome,
        n=n,
        fracao=fracao,
        names=list(result.names),
        labels=list(result.labels),
        significancia=significancia,
        mediana_subconjunto=mediana_sub,
        mediana_geral=mediana_geral,
        desvio=desvio,
        limiar=limiar,
        avisos=avisos,
    )
