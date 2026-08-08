"""
Ajuste de distribuicoes a dados historicos (distribution fitting).

Estimacao por maxima verossimilhanca via scipy, com ranking por AIC/BIC e
testes de aderencia de Kolmogorov-Smirnov e Anderson-Darling.

O PONTO METODOLOGICO MAIS IMPORTANTE DESTE MODULO
-------------------------------------------------
Quando os parametros da distribuicao sao estimados A PARTIR DOS MESMOS
DADOS usados no teste, as distribuicoes nulas tabeladas de K-S e A-D
deixam de valer. O ajuste "puxa" a distribuicao na direcao da amostra,
o que reduz a estatistica de teste e infla o p-valor: o teste passa a
aceitar ajustes ruins com frequencia excessiva.

Este modulo por isso NAO reporta o p-valor assintotico do K-S. Ele calcula
o p-valor por BOOTSTRAP PARAMETRICO (procedimento de Lilliefors
generalizado): simula B amostras do modelo ajustado, reajusta o modelo em
cada uma e compara a estatistica observada com a distribuicao empirica
resultante. Isso e mais caro, e correto, e normalmente devolve p-valores
bem menores que o teste ingenuo.

Segunda ressalva: AIC/BIC ordenam modelos, mas nao dizem que o melhor
colocado e adequado. Uma lista de ajustes ruins ainda produz um "vencedor".
Sempre olhe o Q-Q plot.

Referencias:
  - Akaike, H. (1974). IEEE Transactions on Automatic Control 19(6):716-723.
  - Schwarz, G. (1978). Annals of Statistics 6(2):461-464.
  - Burnham, K.P. & Anderson, D.R. (2002). "Model Selection and Multimodel
    Inference", 2a ed., Springer (AICc e uso pratico do AIC).
  - Lilliefors, H.W. (1967). JASA 62(318):399-402 (K-S com parametros estimados).
  - Stephens, M.A. (1974). "EDF Statistics for Goodness of Fit and Some
    Comparisons". JASA 69(347):730-737 (Anderson-Darling).
  - Babu, G.J. & Rao, C.R. (2004). "Goodness-of-fit tests when parameters are
    estimated". Sankhya 66(1):63-74 (justificativa do bootstrap parametrico).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

# Candidatas continuas oferecidas no ajuste. Todas com `fit` no scipy.
FITTABLE = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "exponential": stats.expon,
    "beta": stats.beta,
    "logistic": stats.logistic,
    "gumbel_r": stats.gumbel_r,
    "uniform": stats.uniform,
    "triangular": stats.triang,
    "student_t": stats.t,
    "pareto": stats.pareto,
    "genpareto": stats.genpareto,
}

# Distribuicoes com suporte estritamente positivo: nao faz sentido
# ajusta-las a dados com valores <= 0 sem deslocamento.
_POSITIVE_SUPPORT = {"lognormal", "gamma", "weibull", "exponential", "pareto"}


@dataclass
class FitResult:
    name: str
    dist_key: str
    params: tuple
    loglik: float
    n_params: int
    aic: float
    aicc: float
    bic: float
    ks_stat: float
    ks_pvalue_bootstrap: float | None
    ad_stat: float
    ad_pvalue_bootstrap: float | None
    converged: bool
    message: str = ""

    def as_record(self) -> Dict[str, object]:
        return {
            "distribuicao": self.name,
            "n_params": self.n_params,
            "logL": self.loglik,
            "AIC": self.aic,
            "AICc": self.aicc,
            "BIC": self.bic,
            "KS": self.ks_stat,
            "KS_p_boot": self.ks_pvalue_bootstrap,
            "AD": self.ad_stat,
            "AD_p_boot": self.ad_pvalue_bootstrap,
            "parametros": ", ".join(f"{p:.6g}" for p in self.params),
        }


# ---------------------------------------------------------------------------
# Estatisticas de aderencia
# ---------------------------------------------------------------------------


def ks_statistic(data: np.ndarray, frozen) -> float:
    """Estatistica D de Kolmogorov-Smirnov (sem p-valor assintotico)."""
    x = np.sort(np.asarray(data, dtype=float))
    n = x.size
    cdf = frozen.cdf(x)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - cdf)
    d_minus = np.max(cdf - (i - 1) / n)
    return float(max(d_plus, d_minus))


def anderson_darling_statistic(data: np.ndarray, frozen) -> float:
    """Estatistica A^2 de Anderson-Darling.

    A^2 = -n - (1/n) * sum_{i=1..n} (2i-1) * [ln F(x_i) + ln(1 - F(x_{n+1-i}))]

    Pesa mais as caudas que o K-S, motivo pelo qual e preferivel em analise
    de risco: e justamente na cauda que o modelo importa.
    """
    x = np.sort(np.asarray(data, dtype=float))
    n = x.size
    if n < 2:
        return float("nan")
    f = np.clip(frozen.cdf(x), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(f) + np.log(1.0 - f[::-1])))
    return float(-n - s / n)


# ---------------------------------------------------------------------------
# Ajuste
# ---------------------------------------------------------------------------


def _fit_one(dist, x: np.ndarray, floc_zero: bool = False):
    kwargs = {}
    if floc_zero:
        kwargs["floc"] = 0.0
    with np.errstate(all="ignore"):
        params = dist.fit(x, **kwargs)
    return params


def fit_distribution(
    data: Sequence[float],
    dist_key: str,
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
) -> FitResult:
    """Ajusta uma distribuicao por MLE e calcula os criterios de aderencia."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        raise ValueError("sao necessarias ao menos 5 observacoes finitas")
    if dist_key not in FITTABLE:
        raise KeyError(f"distribuicao nao ajustavel: {dist_key}")
    dist = FITTABLE[dist_key]
    rng = np.random.default_rng() if rng is None else rng

    converged, message = True, ""
    try:
        # Para suportes positivos, fixa loc=0 quando os dados sao positivos:
        # deixar loc livre costuma colapsar o ajuste no minimo amostral.
        floc0 = dist_key in _POSITIVE_SUPPORT and x.min() > 0
        params = _fit_one(dist, x, floc_zero=floc0)
        frozen = dist(*params)
        logpdf = frozen.logpdf(x)
        if not np.all(np.isfinite(logpdf)):
            converged = False
            message = (
                "verossimilhanca infinita em parte da amostra (dados fora do "
                "suporte do modelo ajustado)"
            )
        loglik = float(np.sum(logpdf[np.isfinite(logpdf)]))
    except Exception as e:  # ajuste pode falhar legitimamente
        return FitResult(
            name=dist_key,
            dist_key=dist_key,
            params=tuple(),
            loglik=float("nan"),
            n_params=0,
            aic=float("inf"),
            aicc=float("inf"),
            bic=float("inf"),
            ks_stat=float("nan"),
            ks_pvalue_bootstrap=None,
            ad_stat=float("nan"),
            ad_pvalue_bootstrap=None,
            converged=False,
            message=f"falha no ajuste: {e}",
        )

    # graus de liberdade efetivos (loc fixado nao conta)
    k = len(params) - (1 if (dist_key in _POSITIVE_SUPPORT and x.min() > 0) else 0)
    k = max(k, 1)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else float("inf")

    ks = ks_statistic(x, frozen)
    ad = anderson_darling_statistic(x, frozen)

    ks_p = ad_p = None
    if bootstrap and bootstrap > 0:
        ks_p, ad_p = _bootstrap_pvalues(
            x, dist, dist_key, ks, ad, bootstrap, rng
        )

    return FitResult(
        name=dist_key,
        dist_key=dist_key,
        params=tuple(float(p) for p in params),
        loglik=loglik,
        n_params=k,
        aic=float(aic),
        aicc=float(aicc),
        bic=float(bic),
        ks_stat=ks,
        ks_pvalue_bootstrap=ks_p,
        ad_stat=ad,
        ad_pvalue_bootstrap=ad_p,
        converged=converged,
        message=message,
    )


def _bootstrap_pvalues(
    x: np.ndarray,
    dist,
    dist_key: str,
    ks_obs: float,
    ad_obs: float,
    B: int,
    rng: np.random.Generator,
):
    """p-valores por bootstrap parametrico (Lilliefors generalizado).

    Sob H0 (os dados vieram do modelo ajustado), gera B amostras do modelo,
    REAJUSTA o modelo em cada uma e recalcula a estatistica. O p-valor e a
    fracao de replicas com estatistica >= a observada. O reajuste dentro do
    laco e essencial: e ele que reproduz o "encolhimento" da estatistica
    causado pela estimacao.
    """
    n = x.size
    floc0 = dist_key in _POSITIVE_SUPPORT and x.min() > 0
    try:
        params = _fit_one(dist, x, floc_zero=floc0)
        frozen0 = dist(*params)
    except Exception:
        return None, None

    ks_null: List[float] = []
    ad_null: List[float] = []
    for _ in range(int(B)):
        try:
            xb = frozen0.rvs(size=n, random_state=rng)
            xb = xb[np.isfinite(xb)]
            if xb.size < 5:
                continue
            pb = _fit_one(dist, xb, floc_zero=floc0)
            fb = dist(*pb)
            ks_null.append(ks_statistic(xb, fb))
            ad_null.append(anderson_darling_statistic(xb, fb))
        except Exception:
            continue

    if len(ks_null) < 20:
        return None, None
    ks_null_arr = np.asarray(ks_null)
    ad_null_arr = np.asarray(ad_null)
    # Estimador com correcao +1 (evita p-valor exatamente 0); Davison & Hinkley (1997)
    ks_p = float((np.sum(ks_null_arr >= ks_obs) + 1) / (ks_null_arr.size + 1))
    ad_p = float((np.sum(ad_null_arr >= ad_obs) + 1) / (ad_null_arr.size + 1))
    return ks_p, ad_p


def fit_many(
    data: Sequence[float],
    dist_keys: Sequence[str] | None = None,
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
) -> List[FitResult]:
    """Ajusta varias candidatas e devolve a lista ordenada por AICc."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    keys = list(dist_keys) if dist_keys else list(FITTABLE)
    if x.size and x.min() <= 0:
        keys = [k for k in keys if k not in _POSITIVE_SUPPORT]
    out: List[FitResult] = []
    for k in keys:
        try:
            out.append(fit_distribution(x, k, bootstrap=bootstrap, rng=rng))
        except Exception:
            continue
    out.sort(key=lambda r: (np.isnan(r.aicc), r.aicc))
    return out


def akaike_weights(results: Sequence[FitResult]) -> np.ndarray:
    """Pesos de Akaike: peso relativo de evidencia entre os modelos testados.

    w_i = exp(-delta_i/2) / sum_j exp(-delta_j/2), delta_i = AICc_i - min(AICc).

    Interpretacao honesta: e evidencia RELATIVA dentro do conjunto testado.
    Se todos os modelos forem ruins, o peso alto do primeiro nao significa
    que ele seja bom (Burnham & Anderson, 2002, secao 2.9).
    """
    a = np.asarray([r.aicc for r in results], dtype=float)
    finite = np.isfinite(a)
    w = np.zeros_like(a)
    if not finite.any():
        return w
    d = a[finite] - a[finite].min()
    e = np.exp(-d / 2.0)
    w[finite] = e / e.sum()
    return w


def qq_points(data: Sequence[float], result: FitResult):
    """Pontos (quantil teorico, quantil amostral) para Q-Q plot."""
    x = np.sort(np.asarray(data, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    dist = FITTABLE[result.dist_key]
    frozen = dist(*result.params)
    # posicoes de plotagem de Hazen
    p = (np.arange(1, n + 1) - 0.5) / n
    return np.asarray(frozen.ppf(p), dtype=float), x


def describe_data(data: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "n": int(x.size),
        "media": float(np.mean(x)) if x.size else float("nan"),
        "desvio": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        "minimo": float(np.min(x)) if x.size else float("nan"),
        "mediana": float(np.median(x)) if x.size else float("nan"),
        "maximo": float(np.max(x)) if x.size else float("nan"),
        "assimetria": float(stats.skew(x)) if x.size > 2 else float("nan"),
        "curtose_excesso": float(stats.kurtosis(x)) if x.size > 3 else float("nan"),
    }
