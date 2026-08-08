"""
Estatisticas de saida e quantificacao do erro de simulacao.

A parte facil e calcular media, desvio e percentis. A parte que costuma ser
omitida - e que este modulo trata explicitamente - e QUANTO desses numeros e
ruido de amostragem.

Tres mecanismos:
  1. `mc_standard_error`: erro padrao classico s/sqrt(n). SO e valido para
     Monte Carlo simples i.i.d.
  2. `quantile_ci`: intervalo de confianca nao parametrico para quantis via
     estatisticas de ordem (metodo binomial). Tambem assume i.i.d.
  3. `replicate_summary`: agrega R replicacoes INDEPENDENTES da simulacao
     inteira. E o unico caminho valido para LHS, e o mais confiavel em geral.

Referencias:
  - Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering",
    Springer, cap.1 (erro padrao e intervalos de confianca).
  - Conover, W.J. (1999). "Practical Nonparametric Statistics", 3a ed.,
    Wiley (IC para quantis via distribuicao binomial).
  - Stein, M. (1987). Technometrics 29(2):143-151 (dependencia sob LHS).
  - Artzner et al. (1999), "Coherent Measures of Risk", Mathematical Finance
    9(3):203-228 (por que VaR nao e subaditivo e CVaR e).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats

DEFAULT_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def describe(x: np.ndarray, percentiles: Sequence[float] = DEFAULT_PERCENTILES):
    """Resumo descritivo da distribuicao de saida."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    n_bad = int((~finite).sum())
    xf = x[finite]
    if xf.size == 0:
        raise ValueError("nenhum valor finito na saida da simulacao")
    out: Dict[str, float] = {
        "n": int(x.size),
        "n_nao_finitos": n_bad,
        "media": float(np.mean(xf)),
        "desvio": float(np.std(xf, ddof=1)) if xf.size > 1 else float("nan"),
        "minimo": float(np.min(xf)),
        "maximo": float(np.max(xf)),
        "assimetria": float(stats.skew(xf)) if xf.size > 2 else float("nan"),
        "curtose_excesso": float(stats.kurtosis(xf)) if xf.size > 3 else float("nan"),
    }
    out["coef_variacao"] = (
        out["desvio"] / abs(out["media"]) if out["media"] != 0 else float("nan")
    )
    for p in percentiles:
        out[f"P{p:g}"] = float(np.percentile(xf, p))
    return out


def mc_standard_error(x: np.ndarray) -> float:
    """Erro padrao da media sob amostragem i.i.d.: s / sqrt(n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """IC para a media via TCL. Invalido sob LHS e sob caudas muito pesadas."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    se = mc_standard_error(x)
    if not np.isfinite(se):
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2.0)
    m = float(np.mean(x))
    return (m - z * se, m + z * se)


def quantile_ci(
    x: np.ndarray, p: float, level: float = 0.95
) -> tuple[float, float]:
    """IC nao parametrico para o quantil p via estatisticas de ordem.

    O numero de observacoes abaixo do quantil populacional segue
    Binomial(n, p). Os limites sao as estatisticas de ordem correspondentes
    aos quantis da binomial. Nao assume forma da distribuicao, mas assume
    observacoes i.i.d.
    """
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))])
    n = x.size
    if n < 2 or not (0 < p < 1):
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    lo_k = int(stats.binom.ppf(alpha / 2.0, n, p))
    hi_k = int(stats.binom.ppf(1.0 - alpha / 2.0, n, p)) + 1
    lo_k = max(lo_k, 1)
    hi_k = min(hi_k, n)
    return (float(x[lo_k - 1]), float(x[hi_k - 1]))


def prob_below(x: np.ndarray, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x <= threshold))


def prob_above(x: np.ndarray, threshold: float) -> float:
    return 1.0 - prob_below(x, threshold)


def value_at_risk(x: np.ndarray, alpha: float = 0.05, loss_is_low: bool = True):
    """VaR ao nivel alpha.

    `loss_is_low=True` (padrao): a saida e um ganho (lucro, VPL) e a perda
    esta na cauda ESQUERDA, entao VaR = percentil alpha.
    `loss_is_low=False`: a saida ja e uma perda/custo e a cauda de interesse
    e a DIREITA, entao VaR = percentil (1-alpha).

    Aviso: VaR nao e uma medida de risco coerente (nao e subaditivo) -
    Artzner et al. (1999). Reporte tambem o CVaR.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = alpha * 100 if loss_is_low else (1 - alpha) * 100
    return float(np.percentile(x, q))


def conditional_value_at_risk(
    x: np.ndarray, alpha: float = 0.05, loss_is_low: bool = True
):
    """CVaR / Expected Shortfall: media condicional alem do VaR."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    v = value_at_risk(x, alpha, loss_is_low)
    tail = x[x <= v] if loss_is_low else x[x >= v]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))


def replicate_summary(
    values: Sequence[float], level: float = 0.95
) -> Dict[str, float]:
    """Agrega uma estatistica calculada em R replicacoes independentes.

    Esta e a forma correta de medir o erro de simulacao sob LHS, amostragem
    estratificada ou qualquer esquema com dependencia entre iteracoes: cada
    replicacao completa e uma observacao i.i.d. da estatistica.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    r = v.size
    if r < 2:
        return {
            "replicacoes": int(r),
            "media": float(v[0]) if r == 1 else float("nan"),
            "erro_padrao": float("nan"),
            "ic_inf": float("nan"),
            "ic_sup": float("nan"),
        }
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(r))
    t = stats.t.ppf(0.5 + level / 2.0, df=r - 1)
    return {
        "replicacoes": int(r),
        "media": m,
        "erro_padrao": se,
        "ic_inf": m - t * se,
        "ic_sup": m + t * se,
    }


def iterations_for_precision(
    x: np.ndarray, target_halfwidth: float, level: float = 0.95
) -> float:
    """Quantas iteracoes i.i.d. seriam necessarias para dado semi-IC da media.

    n >= (z * s / semi_amplitude)^2. Serve como ordem de grandeza, nao como
    garantia: `s` e ele proprio estimado, e a formula nao vale para LHS nem
    para distribuicoes de variancia infinita.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or target_halfwidth <= 0:
        return float("nan")
    z = stats.norm.ppf(0.5 + level / 2.0)
    s = float(np.std(x, ddof=1))
    return float((z * s / target_halfwidth) ** 2)


def convergence_path(x: np.ndarray, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Media acumulada ao longo das iteracoes, para inspecao visual.

    Sob LHS a ordem das iteracoes e arbitraria, entao o grafico serve apenas
    como diagnostico qualitativo de estabilizacao.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.array([]), np.array([])
    running = np.cumsum(x) / np.arange(1, n + 1)
    idx = np.unique(np.linspace(1, n, min(points, n)).astype(int)) - 1
    return idx + 1, running[idx]
