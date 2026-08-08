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

from dataclasses import dataclass
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
