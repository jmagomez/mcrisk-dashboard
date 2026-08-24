"""
Analise de sensibilidade (graficos tornado).

Metodos implementados, todos baseados na propria amostra da simulacao
(nao exigem rodadas adicionais do modelo):

  1. Correlacao de posto (Spearman) entre cada entrada e a saida.
  2. Coeficiente de regressao de posto padronizado (SRRC): regressao linear
     multipla dos POSTOS da saida sobre os POSTOS das entradas, com
     variaveis padronizadas.

Por que os dois: a correlacao de Spearman e marginal e ignora que as
entradas podem estar correlacionadas entre si; o SRRC controla as demais
entradas. Quando as entradas sao independentes, os dois praticamente
coincidem. Quando ha correlacao imposta, eles divergem - e essa divergencia
e informacao, nao erro.

RESSALVA CENTRAL (reportada junto com os resultados): ambos os metodos so
sao validos se a relacao entrada-saida for MONOTONA. O R^2 da regressao de
postos e devolvido como medida de quanto do comportamento do modelo esses
indices conseguem explicar. Com R^2 baixo, o tornado e enganoso e o caminho
correto sao indices de sensibilidade globais baseados em variancia
(indices de Sobol), que exigem um desenho amostral proprio e nao estao
implementados aqui.

Referencias:
  - Helton, J.C. & Davis, F.J. (2002). "Illustration of Sampling-Based Methods
    for Uncertainty and Sensitivity Analysis". Risk Analysis 22(3):591-622.
  - Saltelli, A. & Sobol', I.M. (1995). "About the use of rank transformation
    in sensitivity analysis of model output". Reliability Engineering & System
    Safety 50(3):225-239.
    https://www.andreasaltelli.eu/file/repository/Saltelli_Sobol1995.pdf
  - Saltelli, A. et al. (2008). "Global Sensitivity Analysis: The Primer", Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import stats


@dataclass
class SensitivityResult:
    names: List[str]
    spearman: np.ndarray  # correlacao de posto entrada-saida
    spearman_p: np.ndarray  # p-valor bilateral
    srrc: np.ndarray  # coef. de regressao de posto padronizado
    rank_r2: float  # R^2 da regressao de postos
    contribution: np.ndarray  # SRRC^2 normalizado (soma = 1), so valido se R^2 alto
    warnings: List[str]

    def as_records(self) -> List[Dict[str, float]]:
        return [
            {
                "variavel": self.names[i],
                "spearman": float(self.spearman[i]),
                "p_valor": float(self.spearman_p[i]),
                "srrc": float(self.srrc[i]),
                "contrib_variancia_pct": float(self.contribution[i] * 100.0),
            }
            for i in range(len(self.names))
        ]


def _ranks(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a, axis=0)


def _max_vif(Z: np.ndarray) -> float:
    """Maior fator de inflacao de variancia entre as colunas de Z.

    VIF_j = 1 / (1 - R^2_j), onde R^2_j vem da regressao da coluna j sobre
    as demais. Calculado pela diagonal da inversa da matriz de correlacao,
    que e a identidade algebrica equivalente e bem mais barata.
    """
    k = Z.shape[1]
    if k < 2:
        return 1.0
    R = np.corrcoef(Z, rowvar=False)
    try:
        vifs = np.diag(np.linalg.inv(R))
    except np.linalg.LinAlgError:
        return float("inf")
    vifs = vifs[np.isfinite(vifs)]
    return float(np.max(vifs)) if vifs.size else 1.0


def analyze(
    X: np.ndarray, y: np.ndarray, names: List[str], r2_warn: float = 0.7
) -> SensitivityResult:
    """Calcula os indices de sensibilidade e os avisos associados."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X deve ser 2D (n_iteracoes x n_variaveis)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo numero de linhas")
    if len(names) != X.shape[1]:
        raise ValueError("names deve ter um nome por coluna de X")

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    n, k = X.shape
    warnings: List[str] = []
    if int((~ok).sum()) > 0:
        warnings.append(
            f"{int((~ok).sum())} iteracoes descartadas por conterem valores "
            f"nao finitos."
        )

    spearman = np.full(k, np.nan)
    spearman_p = np.full(k, np.nan)
    constant_cols: List[int] = []
    for j in range(k):
        if np.allclose(X[:, j], X[0, j]):
            constant_cols.append(j)
            spearman[j], spearman_p[j] = 0.0, 1.0
            continue
        res = stats.spearmanr(X[:, j], y)
        spearman[j] = float(res.statistic)
        spearman_p[j] = float(res.pvalue)
    if constant_cols:
        warnings.append(
            "Variaveis constantes na amostra (sensibilidade indefinida, "
            "reportada como 0): "
            + ", ".join(names[j] for j in constant_cols)
        )

    # --- regressao de postos padronizada ---
    srrc = np.zeros(k)
    rank_r2 = float("nan")
    if n > k + 1 and not np.allclose(y, y[0]):
        Rx = _ranks(X)
        Ry = _ranks(y)
        sx = Rx.std(axis=0, ddof=1)
        sy = Ry.std(ddof=1)
        live = sx > 0
        if live.any() and sy > 0:
            Z = (Rx[:, live] - Rx[:, live].mean(axis=0)) / sx[live]
            zy = (Ry - Ry.mean()) / sy
            # Colinearidade forte entre entradas torna os coeficientes
            # instaveis; lstsq resolve via SVD e nao explode, mas avisamos.
            # VIF_j = 1 / (1 - R^2_j), com R^2_j da regressao de Z_j nas demais.
            # Regra de bolso classica: VIF > 10 indica multicolinearidade seria
            # (Belsley, Kuh & Welsch, 1980; Kutner et al., 2005).
            max_vif = _max_vif(Z)
            beta, *_ = np.linalg.lstsq(Z, zy, rcond=None)
            srrc[live] = beta
            resid = zy - Z @ beta
            ss_tot = float(np.sum(zy**2))
            rank_r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else np.nan
            if max_vif > 10.0:
                warnings.append(
                    f"Entradas fortemente correlacionadas entre si (VIF maximo = "
                    f"{max_vif:.1f}, acima do limiar usual de 10). Os SRRC ficam "
                    f"instaveis e a atribuicao de importancia entre variaveis "
                    f"correlacionadas e ambigua por construcao: prefira ler a "
                    f"coluna de Spearman e tratar o grupo correlacionado em bloco."
                )
    else:
        warnings.append(
            "Amostra pequena demais (ou saida constante) para a regressao de "
            "postos; apenas a correlacao de Spearman foi calculada."
        )

    if np.isfinite(rank_r2) and rank_r2 < r2_warn:
        warnings.append(
            f"R^2 da regressao de postos = {rank_r2:.3f}. Como e menor que "
            f"{r2_warn:.2f}, uma parte relevante do comportamento do modelo NAO "
            f"e monotona/linear nos postos. O grafico tornado abaixo pode "
            f"ordenar mal as variaveis: considere indices de Sobol."
        )

    s2 = srrc**2
    contribution = s2 / s2.sum() if s2.sum() > 0 else np.zeros_like(s2)

    return SensitivityResult(
        names=list(names),
        spearman=spearman,
        spearman_p=spearman_p,
        srrc=srrc,
        rank_r2=rank_r2,
        contribution=contribution,
        warnings=warnings,
    )


def tornado_order(
    result: SensitivityResult, by: str = "srrc"
) -> List[int]:
    """Indices ordenados por importancia decrescente (|valor|)."""
    if by == "spearman":
        vals = np.abs(result.spearman)
    elif by == "srrc":
        vals = np.abs(result.srrc)
    else:
        raise ValueError("by deve ser 'srrc' ou 'spearman'")
    vals = np.nan_to_num(vals, nan=0.0)
    return list(np.argsort(-vals))


# ===========================================================================
# Metodo 3: Change in Output Statistic (sensibilidade condicional)
# ===========================================================================
#
# Os dois metodos acima resumem a relacao entrada-saida num numero so, e por
# isso NAO conseguem enxergar relacao nao monotona: uma variavel em U tem
# correlacao de posto proxima de zero e SRRC proximo de zero, mesmo dominando
# o modelo. Este terceiro metodo nao tem esse ponto cego, porque nao supoe
# forma nenhuma - ele so pergunta "como a estatistica da saida muda conforme
# eu ando pela faixa da entrada?".
#
# O preco e que ele responde uma pergunta diferente e mais fraca: e uma medida
# MARGINAL (nao controla as demais entradas) e depende da escolha do numero de
# faixas. Poucas faixas suavizam demais; muitas deixam cada faixa com poucas
# iteracoes e o resultado vira ruido.

STATS_CONDICIONAIS = ("media", "mediana", "desvio", "p05", "p10", "p90", "p95")


def _stat_de(y: np.ndarray, stat: str) -> float:
    if stat == "media":
        return float(np.mean(y))
    if stat == "mediana":
        return float(np.median(y))
    if stat == "desvio":
        return float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    if stat.startswith("p"):
        try:
            q = float(stat[1:])
        except ValueError as exc:
            raise ValueError(f"estatistica invalida: {stat!r}") from exc
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"percentil fora de [0, 100]: {stat!r}")
        return float(np.percentile(y, q))
    raise ValueError(
        f"estatistica invalida: {stat!r}; use uma de {STATS_CONDICIONAIS} "
        f"ou 'pNN' com NN entre 0 e 100"
    )


@dataclass
class SensibilidadeCondicional:
    """Resultado do metodo Change in Output Statistic."""

    names: List[str]
    stat: str
    bins: int
    valores: np.ndarray  # (k, bins) estatistica da saida em cada faixa
    centros: np.ndarray  # (k, bins) valor medio da entrada em cada faixa
    swing: np.ndarray  # (k,) amplitude da estatistica ao longo das faixas
    base: float  # estatistica sobre a amostra inteira
    n_por_faixa: np.ndarray  # (bins,) iteracoes em cada faixa
    warnings: List[str] = field(default_factory=list)

    def ordem(self) -> List[int]:
        return list(np.argsort(-np.nan_to_num(self.swing, nan=0.0)))

    def as_records(self) -> List[Dict[str, float]]:
        return [
            {
                "variavel": self.names[i],
                "swing": float(self.swing[i]),
                "minimo_da_faixa": float(np.nanmin(self.valores[i])),
                "maximo_da_faixa": float(np.nanmax(self.valores[i])),
                "swing_pct_da_base": (
                    float(self.swing[i] / abs(self.base) * 100.0)
                    if self.base != 0
                    else float("nan")
                ),
            }
            for i in range(len(self.names))
        ]


def change_in_output_statistic(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    stat: str = "media",
    bins: int = 10,
    min_por_faixa: int = 30,
) -> SensibilidadeCondicional:
    """Sensibilidade por faixas equiprovaveis da entrada.

    Para cada entrada: ordena as iteracoes pelo valor da entrada, divide em
    `bins` faixas de tamanho aproximadamente igual, calcula `stat` da SAIDA
    dentro de cada faixa, e mede a amplitude (swing) dessa estatistica.

    A divisao e por CONTAGEM, nao por largura: faixas equiprovaveis. Dividir
    por largura deixaria as faixas das caudas quase vazias em distribuicoes
    assimetricas, e a estatistica ali seria ruido puro.

    `min_por_faixa` nao altera o calculo - ele so dispara um aviso quando as
    faixas ficam pequenas demais para a estatistica ser estavel. Silenciar
    isso seria pior que nao ter o aviso, porque o grafico continua bonito.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X deve ser 2D (n_iteracoes x n_variaveis)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo numero de linhas")
    if len(names) != X.shape[1]:
        raise ValueError("names deve ter um nome por coluna de X")
    if bins < 2:
        raise ValueError(f"bins deve ser >= 2, recebido {bins}")

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    n, k = X.shape
    warnings: List[str] = []
    if int((~ok).sum()):
        warnings.append(
            f"{int((~ok).sum())} iteracoes descartadas por conterem valores "
            f"nao finitos."
        )
    if n < bins * 2:
        raise ValueError(
            f"{n} iteracoes nao dao para {bins} faixas; reduza bins ou "
            f"aumente o numero de iteracoes"
        )

    base = _stat_de(y, stat)  # valida `stat` cedo
    cortes = np.array_split(np.arange(n), bins)
    n_por_faixa = np.array([c.size for c in cortes], dtype=int)
    if n_por_faixa.min() < min_por_faixa:
        warnings.append(
            f"As faixas tem no minimo {n_por_faixa.min()} iteracoes (abaixo de "
            f"{min_por_faixa}). A estatistica dentro de cada faixa fica "
            f"instavel e o swing passa a medir ruido de amostragem, nao "
            f"sensibilidade. Rode com mais iteracoes ou reduza o numero de "
            f"faixas."
        )

    valores = np.full((k, bins), np.nan)
    centros = np.full((k, bins), np.nan)
    constantes: List[int] = []
    for j in range(k):
        col = X[:, j]
        if np.allclose(col, col[0]):
            constantes.append(j)
            valores[j, :] = base
            centros[j, :] = col[0]
            continue
        ordem = np.argsort(col, kind="stable")
        for b, idx in enumerate(cortes):
            sel = ordem[idx]
            valores[j, b] = _stat_de(y[sel], stat)
            centros[j, b] = float(np.mean(col[sel]))
    if constantes:
        warnings.append(
            "Variaveis constantes na amostra (swing 0 por construcao): "
            + ", ".join(names[j] for j in constantes)
        )

    swing = np.nanmax(valores, axis=1) - np.nanmin(valores, axis=1)
    return SensibilidadeCondicional(
        names=list(names),
        stat=stat,
        bins=bins,
        valores=valores,
        centros=centros,
        swing=swing,
        base=base,
        n_por_faixa=n_por_faixa,
        warnings=warnings,
    )


# ===========================================================================
# Metodo 4: Contribuicao para a variancia (soma de quadrados sequencial)
# ===========================================================================
#
# Os tres metodos anteriores respondem "quanto a saida se move quando esta
# entrada se move?". Este responde outra coisa: "que FRACAO da variancia da
# saida esta entrada explica?". A diferenca importa na hora de decidir onde
# gastar dinheiro reduzindo incerteza - o que se quer reduzir e variancia.
#
# O metodo constroi a regressao por passos, acrescentando uma entrada por vez,
# sempre a que mais aumenta o R^2 no passo. O incremento de R^2 de cada
# entrada e a sua contribuicao. Como o incremento depende do que ja entrou, a
# ORDEM importa quando as entradas sao correlacionadas - e por isso a soma de
# quadrados e chamada "sequencial" (tipo I), em contraste com a "parcial"
# (tipo III).
#
# ATRIBUICAO E AMBIGUA COM ENTRADAS CORRELACIONADAS. Isso nao e defeito da
# implementacao: com duas entradas quase colineares, qualquer divisao do
# credito entre elas e arbitraria. A funcao mede a correlacao maxima entre
# entradas e avisa. O @RISK documenta a mesma ressalva para o seu metodo
# homonimo.

@dataclass
class ContribuicaoVariancia:
    names: List[str]
    fracao: np.ndarray  # (k,) fracao da variancia explicada por cada entrada
    ordem_entrada: List[int]  # ordem em que as entradas foram acrescentadas
    r2_total: float  # R^2 do modelo completo
    nao_explicada: float  # 1 - r2_total
    usa_postos: bool
    warnings: List[str] = field(default_factory=list)

    def as_records(self) -> List[Dict[str, float]]:
        pos = {j: p for p, j in enumerate(self.ordem_entrada)}
        return [
            {
                "variavel": self.names[i],
                "fracao_variancia": float(self.fracao[i]),
                "pct_variancia": float(self.fracao[i] * 100.0),
                "passo": int(pos.get(i, -1)) + 1,
            }
            for i in range(len(self.names))
        ]


def contribution_to_variance(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    use_ranks: bool = False,
    corr_warn: float = 0.7,
) -> ContribuicaoVariancia:
    """Fracao da variancia da saida atribuida a cada entrada.

    Selecao para a frente: em cada passo entra a variavel que mais aumenta o
    R^2, e o incremento de R^2 e a contribuicao dela. As contribuicoes somam
    exatamente `r2_total`; o que sobra ate 1 e `nao_explicada` e mede o quanto
    do modelo escapa de uma regressao linear (nos valores ou nos postos).

    `use_ranks=True` roda a regressao sobre os POSTOS, o que capta relacoes
    monotonas nao lineares - ao custo de as fracoes passarem a se referir a
    variancia dos postos, nao a da saida. As duas leituras sao legitimas e
    diferentes; por isso o resultado registra qual foi usada.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X deve ser 2D (n_iteracoes x n_variaveis)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo numero de linhas")
    if len(names) != X.shape[1]:
        raise ValueError("names deve ter um nome por coluna de X")

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    n, k = X.shape
    warnings: List[str] = []
    if int((~ok).sum()):
        warnings.append(
            f"{int((~ok).sum())} iteracoes descartadas por conterem valores "
            f"nao finitos."
        )
    if n <= k + 1:
        raise ValueError(
            f"{n} iteracoes para {k} variaveis nao permitem a regressao; "
            f"sao necessarias mais que k+1 observacoes"
        )

    A = _ranks(X) if use_ranks else X
    b = _ranks(y) if use_ranks else y

    sd = A.std(axis=0, ddof=1)
    vivos = sd > 0
    if not vivos.any() or float(np.std(b, ddof=1)) == 0.0:
        return ContribuicaoVariancia(
            names=list(names),
            fracao=np.zeros(k),
            ordem_entrada=[],
            r2_total=0.0,
            nao_explicada=1.0,
            usa_postos=use_ranks,
            warnings=warnings
            + ["Saida constante ou todas as entradas constantes: nada a atribuir."],
        )
    if not vivos.all():
        warnings.append(
            "Variaveis constantes na amostra (contribuicao 0): "
            + ", ".join(names[j] for j in np.flatnonzero(~vivos))
        )

    Z = np.zeros_like(A)
    Z[:, vivos] = (A[:, vivos] - A[:, vivos].mean(axis=0)) / sd[vivos]
    zy = (b - b.mean()) / float(np.std(b, ddof=1))
    ss_tot = float(np.sum(zy**2))

    if vivos.sum() > 1:
        R = np.corrcoef(Z[:, vivos], rowvar=False)
        fora = ~np.eye(R.shape[0], dtype=bool)
        rmax = float(np.abs(R[fora]).max())
        if rmax > corr_warn:
            warnings.append(
                f"Correlacao maxima entre entradas = {rmax:.2f}. Com entradas "
                f"correlacionadas a divisao do credito entre elas e ambigua por "
                f"construcao: a soma de quadrados e SEQUENCIAL, entao quem entra "
                f"antes fica com a parte compartilhada. Leia as contribuicoes "
                f"do grupo correlacionado em bloco, nao uma a uma."
            )

    restantes = list(np.flatnonzero(vivos))
    escolhidas: List[int] = []
    ordem_entrada: List[int] = []
    fracao = np.zeros(k)
    r2_ant = 0.0
    while restantes:
        melhor_j, melhor_r2 = None, r2_ant
        for j in restantes:
            cols = escolhidas + [j]
            beta, *_ = np.linalg.lstsq(Z[:, cols], zy, rcond=None)
            resid = zy - Z[:, cols] @ beta
            r2 = 1.0 - float(np.sum(resid**2)) / ss_tot
            if r2 > melhor_r2 + 1e-15:
                melhor_j, melhor_r2 = j, r2
        if melhor_j is None:  # nenhuma entrada restante acrescenta explicacao
            break
        fracao[melhor_j] = melhor_r2 - r2_ant
        escolhidas.append(melhor_j)
        ordem_entrada.append(melhor_j)
        restantes.remove(melhor_j)
        r2_ant = melhor_r2

    r2_total = float(np.clip(r2_ant, 0.0, 1.0))
    if r2_total < 0.5:
        warnings.append(
            f"R^2 total = {r2_total:.3f}: mais da metade da variancia da saida "
            f"NAO e explicada por nenhuma combinacao linear das entradas. As "
            f"fracoes abaixo somam so {r2_total:.1%} e nao devem ser lidas como "
            f"se cobrissem o modelo inteiro."
        )
    return ContribuicaoVariancia(
        names=list(names),
        fracao=fracao,
        ordem_entrada=ordem_entrada,
        r2_total=r2_total,
        nao_explicada=float(max(0.0, 1.0 - r2_total)),
        usa_postos=use_ranks,
        warnings=warnings,
    )
