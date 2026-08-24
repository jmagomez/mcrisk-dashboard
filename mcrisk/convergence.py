"""
Monitoramento de convergencia: quando parar de iterar.

O grafico de media acumulada, ja presente no pacote, responde "parece ter
estabilizado?" - pergunta visual, sem criterio. Este modulo responde a versao
quantitativa: "a estimativa desta estatistica ja esta dentro de X% do valor
verdadeiro, com Y% de confianca?".

CRITERIO. Para cada estatistica monitorada calcula-se a meia-largura do
intervalo de confianca ao nivel pedido. A estatistica e declarada convergida
quando essa meia-largura cabe dentro da tolerancia relativa:

    meia_largura <= tolerancia * |valor da estatistica|

A tolerancia e RELATIVA porque tolerancia absoluta nao se transporta entre
modelos: 0,01 e apertado para uma saida em milhoes e frouxo para uma em
unidades. O preco e que estatisticas proximas de zero nunca convergem pelo
criterio relativo - a divisao por |valor| explode. A funcao detecta esse caso
e avisa em vez de rodar para sempre.

O QUE ISTO NAO MEDE. Convergencia e sobre erro de AMOSTRAGEM, o menor dos
erros em analise de risco. Uma simulacao convergida com premissas erradas
produz um numero errado com barra de erro estreita, que e pior que um numero
errado com barra larga - a barra estreita convida a confiar. Ver
LIMITATIONS.md secao 1.

SOB LATIN HYPERCUBE. As iteracoes nao sao independentes e `s/sqrt(n)` deixa
de ser estimador valido do erro (Stein, 1987). Sob LHS ele tende a
SUPERESTIMAR o erro, entao o criterio erra para o lado seguro: declara
convergencia mais tarde do que o necessario. Isso e conservador, nao correto -
a medida valida exige replicacoes independentes. O modulo avisa quando recebe
uma amostra vinda de LHS.

Referencias:
  - Conover, W.J. (1999). "Practical Nonparametric Statistics", 3a ed., Wiley.
    (intervalo de confianca de quantis por estatisticas de ordem)
  - Stein, M. (1987). "Large Sample Properties of Simulations Using Latin
    Hypercube Sampling". Technometrics 29(2):143-151.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

ESTATISTICAS = ("media", "desvio", "p05", "p50", "p95")


@dataclass
class EstadoConvergencia:
    """Situacao de uma estatistica num ponto da simulacao."""

    estatistica: str
    n: int
    valor: float
    meia_largura: float
    tolerancia_absoluta: float
    convergiu: bool

    @property
    def erro_relativo(self) -> float:
        return (
            self.meia_largura / abs(self.valor)
            if self.valor != 0 and np.isfinite(self.valor)
            else float("inf")
        )


@dataclass
class RelatorioConvergencia:
    tolerancia: float
    confianca: float
    passo: int
    n_total: int
    trilha: Dict[str, List[EstadoConvergencia]]
    convergiu_em: Dict[str, int | None]
    avisos: List[str] = field(default_factory=list)

    @property
    def tudo_convergiu(self) -> bool:
        return bool(self.convergiu_em) and all(
            v is not None for v in self.convergiu_em.values()
        )

    @property
    def iteracoes_necessarias(self) -> int | None:
        """Maior n entre as estatisticas monitoradas; None se alguma nao convergiu."""
        if not self.tudo_convergiu:
            return None
        return max(v for v in self.convergiu_em.values() if v is not None)

    def as_records(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for nome, estados in self.trilha.items():
            ultimo = estados[-1]
            out.append(
                {
                    "estatistica": nome,
                    "valor": float(ultimo.valor),
                    "meia_largura_IC": float(ultimo.meia_largura),
                    "erro_relativo": float(ultimo.erro_relativo),
                    "tolerancia": float(self.tolerancia),
                    "convergiu": ultimo.convergiu,
                    "iteracoes_ate_convergir": self.convergiu_em[nome],
                }
            )
        return out


def _meia_largura(y: np.ndarray, estatistica: str, confianca: float) -> tuple[float, float]:
    """(valor da estatistica, meia-largura do IC) para a amostra dada."""
    n = y.size
    alfa = 1.0 - confianca
    z = float(stats.norm.ppf(1.0 - alfa / 2.0))

    if estatistica == "media":
        valor = float(np.mean(y))
        return valor, z * float(np.std(y, ddof=1)) / np.sqrt(n)

    if estatistica == "desvio":
        # IC do desvio-padrao pela qui-quadrado, valido sob normalidade.
        # Fora dela e aproximacao; e o motivo de o relatorio marcar o desvio
        # como a estatistica de IC menos confiavel das tres familias aqui.
        s = float(np.std(y, ddof=1))
        gl = n - 1
        lo = s * np.sqrt(gl / stats.chi2.ppf(1.0 - alfa / 2.0, gl))
        hi = s * np.sqrt(gl / stats.chi2.ppf(alfa / 2.0, gl))
        return s, float((hi - lo) / 2.0)

    if estatistica.startswith("p"):
        q = float(estatistica[1:]) / 100.0
        if not 0.0 < q < 1.0:
            raise ValueError(f"percentil fora de (0, 100): {estatistica!r}")
        valor = float(np.quantile(y, q))
        # IC nao parametrico por estatisticas de ordem (contagem binomial).
        centro = n * q
        desv = np.sqrt(n * q * (1.0 - q))
        i_lo = int(np.floor(centro - z * desv))
        i_hi = int(np.ceil(centro + z * desv))
        i_lo = max(0, min(n - 1, i_lo))
        i_hi = max(0, min(n - 1, i_hi))
        ordenada = np.sort(y)
        return valor, float((ordenada[i_hi] - ordenada[i_lo]) / 2.0)

    raise ValueError(
        f"estatistica invalida: {estatistica!r}; use uma de {ESTATISTICAS} "
        f"ou 'pNN'"
    )


def check(
    y: np.ndarray,
    estatistica: str = "media",
    tolerancia: float = 0.03,
    confianca: float = 0.95,
) -> EstadoConvergencia:
    """Testa uma unica estatistica sobre a amostra inteira."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        raise ValueError("sao necessarias ao menos 3 iteracoes finitas")
    if not 0.0 < tolerancia < 1.0:
        raise ValueError(f"tolerancia deve estar em (0, 1), recebido {tolerancia}")
    if not 0.0 < confianca < 1.0:
        raise ValueError(f"confianca deve estar em (0, 1), recebido {confianca}")

    valor, meia = _meia_largura(y, estatistica, confianca)
    tol_abs = tolerancia * abs(valor)
    return EstadoConvergencia(
        estatistica=estatistica,
        n=int(y.size),
        valor=valor,
        meia_largura=meia,
        tolerancia_absoluta=float(tol_abs),
        convergiu=bool(np.isfinite(meia) and tol_abs > 0 and meia <= tol_abs),
    )


def monitor(
    y: np.ndarray,
    estatisticas: Sequence[str] = ("media",),
    tolerancia: float = 0.03,
    confianca: float = 0.95,
    passo: int | None = None,
    metodo_amostragem: str | None = None,
) -> RelatorioConvergencia:
    """Percorre a amostra em blocos e registra quando cada estatistica converge.

    Reproduz o que uma simulacao faria ao testar convergencia a cada `passo`
    iteracoes, mas sobre uma amostra ja existente - o que permite responder
    "quantas iteracoes eu precisava?" DEPOIS de rodar, sem re-simular.

    `convergiu_em` guarda a PRIMEIRA vez que o criterio passou. Ele pode
    voltar a falhar mais adiante: com poucas iteracoes o proprio estimador do
    erro e ruidoso, e uma passagem precoce pelo criterio nao e evidencia
    solida. Por isso o relatorio tambem guarda a trilha inteira, e o estado
    final e o que a tabela de resumo mostra.
    """
    y = np.asarray(y, dtype=float)
    finito = np.isfinite(y)
    avisos: List[str] = []
    if not finito.all():
        avisos.append(
            f"{int((~finito).sum())} iteracoes descartadas por conterem valores "
            f"nao finitos."
        )
    y = y[finito]
    n_total = y.size
    if n_total < 30:
        raise ValueError(
            f"{n_total} iteracoes finitas: poucas demais para monitorar "
            f"convergencia de forma significativa"
        )
    if passo is None:
        passo = max(50, n_total // 40)
    if passo < 3:
        raise ValueError(f"passo deve ser >= 3, recebido {passo}")

    if metodo_amostragem is not None and metodo_amostragem != "mc":
        avisos.append(
            f"Amostra gerada por '{metodo_amostragem}': as iteracoes NAO sao "
            f"independentes e os intervalos de confianca usados aqui supoem que "
            f"sejam. Sob Latin Hypercube o erro tende a ser SUPERESTIMADO, "
            f"entao o criterio declara convergencia mais tarde que o necessario "
            f"- conservador, nao correto. Para medida valida do erro, use "
            f"replicacoes independentes."
        )

    pontos = list(range(passo, n_total + 1, passo))
    if not pontos or pontos[-1] != n_total:
        pontos.append(n_total)

    trilha: Dict[str, List[EstadoConvergencia]] = {e: [] for e in estatisticas}
    convergiu_em: Dict[str, int | None] = {e: None for e in estatisticas}
    quase_zero: List[str] = []

    for e in estatisticas:
        for m in pontos:
            if m < 3:
                continue
            estado = check(y[:m], e, tolerancia, confianca)
            trilha[e].append(estado)
            if estado.convergiu and convergiu_em[e] is None:
                convergiu_em[e] = m
        if trilha[e] and trilha[e][-1].tolerancia_absoluta == 0.0:
            quase_zero.append(e)

    if quase_zero:
        avisos.append(
            "Estatistica(s) com valor zero ou proximo de zero ("
            + ", ".join(quase_zero)
            + "): o criterio RELATIVO nao se aplica, porque a tolerancia "
            "absoluta vira zero e nenhuma quantidade de iteracoes satisfaz. "
            "Use tolerancia absoluta explicita para estes casos."
        )
    nao_convergidas = [e for e, v in convergiu_em.items() if v is None]
    if nao_convergidas:
        avisos.append(
            "Nao convergiu em "
            + f"{n_total:,} iteracoes: "
            + ", ".join(nao_convergidas)
            + ". A simulacao ainda vale - so nao ha garantia de que a "
            "estimativa esteja dentro da tolerancia pedida."
        )

    return RelatorioConvergencia(
        tolerancia=float(tolerancia),
        confianca=float(confianca),
        passo=int(passo),
        n_total=int(n_total),
        trilha=trilha,
        convergiu_em=convergiu_em,
        avisos=avisos,
    )


def iteracoes_para_tolerancia(
    y: np.ndarray, tolerancia: float = 0.03, confianca: float = 0.95
) -> float:
    """Quantas iteracoes a MEDIA precisaria para caber na tolerancia.

    Extrapola de `z*s/sqrt(n) <= tol*|media|` invertendo em n. Supoe que s e
    a media ja estao bem estimados e que as iteracoes sao independentes: e
    projecao, nao garantia, e vale so para a media.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return float("nan")
    m = float(np.mean(y))
    s = float(np.std(y, ddof=1))
    if m == 0 or s == 0:
        return float("nan")
    z = float(stats.norm.ppf(1.0 - (1.0 - confianca) / 2.0))
    return float(np.ceil((z * s / (tolerancia * abs(m))) ** 2))
