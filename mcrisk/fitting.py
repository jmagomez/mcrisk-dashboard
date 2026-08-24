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


# ===========================================================================
# Incerteza de parametros e media de modelos
# ===========================================================================
#
# Ate aqui o ajuste termina num vencedor com parametros pontuais, e a simulacao
# usa esse vencedor como se fosse a verdade. Isso descarta DUAS fontes de
# incerteza, e as duas empurram o resultado na mesma direcao: para menos
# incerteza do que existe de fato.
#
#   1. INCERTEZA DE PARAMETRO. Os parametros foram estimados de uma amostra
#      finita. Com n = 30 observacoes, o desvio-padrao estimado de uma normal
#      tem ele proprio uns 13% de erro padrao. Usa-lo como numero exato faz a
#      simulacao herdar uma precisao que os dados nao sustentam.
#
#   2. INCERTEZA DE MODELO. Quando os pesos de Akaike ficam repartidos - 0,45
#      para a lognormal e 0,40 para a gama, digamos - os dados nao escolheram.
#      Rodar so com a lognormal apresenta como certa uma decisao que foi quase
#      um empate, e a diferenca entre as duas costuma estar exatamente na
#      cauda, que e onde o modelo de risco importa.
#
# As duas funcoes abaixo tratam uma cada. A ordem de grandeza do efeito nao e
# afirmada aqui: esta medida no BENCHMARK.md.
#
# Referencias
# -----------
#   - Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the Bootstrap".
#     Chapman & Hall.
#   - Burnham, K.P. & Anderson, D.R. (2002). "Model Selection and Multimodel
#     Inference", 2a ed., Springer, cap. 4 (multimodel inference).
#   - Hoeting, J.A. et al. (1999). "Bayesian Model Averaging: A Tutorial".
#     Statistical Science 14(4):382-401.


@dataclass
class IncertezaParametros:
    """Distribuicao amostral dos parametros de um ajuste, por bootstrap."""

    dist_key: str
    name: str
    params_pontuais: tuple
    amostras: np.ndarray  # (B, n_params)
    n_dados: int
    replicas_uteis: int
    replicas_pedidas: int
    avisos: List[str]

    def erro_padrao(self) -> np.ndarray:
        return self.amostras.std(axis=0, ddof=1)

    def intervalo(self, level: float = 0.95) -> np.ndarray:
        """(n_params, 2) por percentis do bootstrap."""
        alfa = (1.0 - level) / 2.0
        return np.column_stack(
            [
                np.percentile(self.amostras, 100 * alfa, axis=0),
                np.percentile(self.amostras, 100 * (1.0 - alfa), axis=0),
            ]
        )

    def as_records(self, level: float = 0.95) -> List[Dict[str, object]]:
        ic = self.intervalo(level)
        ep = self.erro_padrao()
        return [
            {
                "parametro": f"p{j + 1}",
                "estimativa": float(self.params_pontuais[j]),
                "erro_padrao": float(ep[j]),
                "IC_inf": float(ic[j, 0]),
                "IC_sup": float(ic[j, 1]),
                "erro_relativo_pct": (
                    float(ep[j] / abs(self.params_pontuais[j]) * 100.0)
                    if self.params_pontuais[j] != 0
                    else float("nan")
                ),
            }
            for j in range(self.amostras.shape[1])
        ]

    def _refletidas(self) -> np.ndarray:
        """Bootstrap basico (pivotal): theta* -> 2*theta_chapeu - theta*.

        POR QUE NAO USAR AS REPLICAS CRUAS. As replicas estao centradas no
        estimador, e o estimador de maxima verossimilhanca de um parametro de
        ESCALA e viesado para baixo. Reamostrar direto propaga esse vies para
        a preditiva - e ele cancela quase exatamente o alargamento que a
        incerteza de locacao deveria produzir. Medido para a normal com n=40:
        E[sigma*^2] + Var(mu*) = 0,8097 contra sigma_chapeu^2 = 0,8043, ou seja
        +0,7% de variancia, menos que o ruido do proprio bootstrap. Refletindo
        em torno da estimativa, a variancia sobe para 0,8639 - proxima do valor
        exato 0,8912 da preditiva bayesiana com priori de Jeffreys.

        A reflexao pode produzir combinacao invalida (escala negativa, por
        exemplo). Essas linhas sao descartadas e contadas, nunca corrigidas em
        silencio.
        """
        p = np.asarray(self.params_pontuais, dtype=float)
        refl = 2.0 * p - self.amostras
        dist = FITTABLE[self.dist_key]
        sonda = np.array([0.01, 0.5, 0.99])
        ok = np.zeros(refl.shape[0], dtype=bool)
        for i, th in enumerate(refl):
            if not np.all(np.isfinite(th)):
                continue
            try:
                q = dist.ppf(sonda, *th)
            except Exception:
                continue
            ok[i] = bool(np.all(np.isfinite(q)) and np.all(np.diff(q) > 0))
        return refl[ok]

    def sortear(
        self, n: int, rng: np.random.Generator, corrigir_vies: bool = True
    ) -> np.ndarray:
        """Sorteia n conjuntos de parametros da distribuicao amostral.

        `corrigir_vies=False` devolve as replicas cruas. Existe para o teste
        que MEDE a diferenca entre as duas escolhas - nao como opcao de uso.
        """
        base = self._refletidas() if corrigir_vies else self.amostras
        if base.shape[0] < 10:
            base = self.amostras
        idx = rng.integers(0, base.shape[0], size=n)
        return base[idx]

    def fracao_refletida_valida(self) -> float:
        return float(self._refletidas().shape[0] / self.amostras.shape[0])


def parameter_uncertainty(
    data: Sequence[float],
    dist_key: str,
    replicas: int = 200,
    rng: np.random.Generator | None = None,
) -> IncertezaParametros:
    """Distribuicao amostral dos parametros por bootstrap NAO parametrico.

    Reamostra as observacoes com reposicao e reajusta em cada replica. Nao
    parametrico de proposito: o bootstrap parametrico simularia do proprio
    modelo ajustado e, se o modelo estiver errado, devolveria um intervalo
    estreito e igualmente errado. Reamostrar os dados observados mantem a
    forma empirica da amostra, seja ela qual for.

    `replicas` abaixo de ~100 produz percentis de cauda instaveis; a funcao
    avisa em vez de silenciar.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    avisos: List[str] = []
    if n < 10:
        raise ValueError(
            f"{n} observacoes finitas: poucas demais para estimar incerteza de "
            f"parametro de forma util"
        )
    if dist_key not in FITTABLE:
        raise ValueError(f"distribuicao desconhecida: {dist_key!r}")
    if replicas < 20:
        raise ValueError(f"replicas deve ser >= 20, recebido {replicas}")
    if replicas < 100:
        avisos.append(
            f"{replicas} replicas: os percentis do intervalo ficam instaveis. "
            f"200 ou mais e o usual."
        )
    if n < 30:
        avisos.append(
            f"So {n} observacoes. O bootstrap reamostra o que existe: ele mede "
            f"a incerteza CONDICIONAL a esta amostra, e nao consegue inventar "
            f"a informacao que a amostra nao tem. Com n pequeno o intervalo "
            f"sai otimista."
        )

    rng = np.random.default_rng() if rng is None else rng
    base = fit_distribution(x, dist_key, bootstrap=0)
    floc_zero = dist_key in _POSITIVE_SUPPORT and x.min() > 0
    dist = FITTABLE[dist_key]

    linhas: List[tuple] = []
    for _ in range(replicas):
        amostra = x[rng.integers(0, n, size=n)]
        try:
            p = _fit_one(dist, amostra, floc_zero=floc_zero)
        except Exception:
            continue
        if np.all(np.isfinite(p)):
            linhas.append(tuple(float(v) for v in p))

    if len(linhas) < max(20, replicas // 4):
        raise ValueError(
            f"so {len(linhas)} de {replicas} replicas convergiram para "
            f"{dist_key}: a estimativa de incerteza nao seria confiavel"
        )
    if len(linhas) < replicas:
        avisos.append(
            f"{replicas - len(linhas)} de {replicas} replicas nao convergiram e "
            f"foram descartadas. Se a fracao for grande, o ajuste desta familia "
            f"e fragil nestes dados."
        )

    return IncertezaParametros(
        dist_key=dist_key,
        name=base.name,
        params_pontuais=base.params,
        amostras=np.asarray(linhas, dtype=float),
        n_dados=int(n),
        replicas_uteis=len(linhas),
        replicas_pedidas=int(replicas),
        avisos=avisos,
    )


def simular_com_incerteza(
    inc: IncertezaParametros,
    n: int,
    rng: np.random.Generator | None = None,
    corrigir_vies: bool = True,
) -> np.ndarray:
    """Amostra da PREDITIVA: parametros sorteados a cada iteracao.

    Cada iteracao usa um conjunto de parametros sorteado da distribuicao
    amostral, em vez do conjunto pontual. O resultado tem a variancia da
    distribuicao MAIS a variancia de nao se saber onde ela esta - que e a
    quantidade correta a propagar para uma decisao.
    """
    rng = np.random.default_rng() if rng is None else rng
    dist = FITTABLE[inc.dist_key]
    thetas = inc.sortear(n, rng, corrigir_vies=corrigir_vies)
    u = rng.uniform(size=n)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = float(dist.ppf(u[i], *thetas[i]))
    return out


@dataclass
class MediaDeModelos:
    """Mistura de candidatas ponderada pelos pesos de Akaike."""

    nomes: List[str]
    chaves: List[str]
    pesos: np.ndarray
    params: List[tuple]
    peso_do_primeiro: float
    avisos: List[str]

    def as_records(self) -> List[Dict[str, object]]:
        return [
            {
                "distribuicao": self.nomes[i],
                "peso_akaike": float(self.pesos[i]),
                "peso_pct": float(self.pesos[i] * 100.0),
            }
            for i in range(len(self.nomes))
        ]


def model_average(
    results: Sequence[FitResult],
    peso_minimo: float = 0.01,
    limiar_dominancia: float = 0.9,
) -> MediaDeModelos:
    """Monta a mistura ponderada por pesos de Akaike sobre as candidatas.

    Candidatas com peso abaixo de `peso_minimo` sao descartadas e os pesos
    renormalizados - manter dezenas de modelos com peso 0,001 nao muda o
    resultado e atrapalha a leitura.

    Quando a primeira colocada domina (peso acima de `limiar_dominancia`), a
    media de modelos e praticamente identica a usar so ela, e a funcao diz
    isso: nesse caso a complexidade extra nao compra nada, e fingir que compra
    seria o mesmo erro de sentido contrario.
    """
    if not results:
        raise ValueError("nenhum ajuste para combinar")
    w = akaike_weights(results)
    if w.sum() <= 0:
        raise ValueError("pesos de Akaike degenerados: nenhum ajuste utilizavel")

    manter = w >= peso_minimo
    if not manter.any():
        manter = w == w.max()
    w2 = w[manter]
    w2 = w2 / w2.sum()
    escolhidos = [r for r, m in zip(results, manter) if m]

    avisos: List[str] = []
    primeiro = float(w2.max())
    if primeiro >= limiar_dominancia:
        avisos.append(
            f"A primeira colocada concentra {primeiro:.1%} do peso de Akaike. A "
            f"media de modelos aqui e praticamente indistinguivel de usar so "
            f"ela: a incerteza de MODELO e pequena neste conjunto. (A de "
            f"PARAMETRO continua existindo e e outra conta.)"
        )
    elif primeiro >= 0.6:
        avisos.append(
            f"A primeira colocada tem {primeiro:.1%} do peso: ha preferencia "
            f"clara, mas nao decisiva. A media de modelos desloca pouco o corpo "
            f"da distribuicao e mais a cauda, que e onde as familias divergem."
        )
    else:
        avisos.append(
            f"Maior peso = {primeiro:.1%}: os dados NAO escolheram entre as "
            f"familias. Usar so a vencedora apresentaria como certa uma decisao "
            f"que foi quase empate - e o desempate cairia justamente na cauda, "
            f"que e a regiao com menos observacoes para decidir."
        )
    todas_ruins = [
        r.ks_pvalue_bootstrap
        for r in escolhidos
        if r.ks_pvalue_bootstrap is not None
    ]
    if todas_ruins and max(todas_ruins) < 0.05:
        avisos.append(
            "Todas as candidatas mantidas sao rejeitadas pelo teste K-S. Media "
            "de modelos combina candidatas; nao conserta um conjunto em que "
            "nenhuma serve."
        )

    return MediaDeModelos(
        nomes=[r.name for r in escolhidos],
        chaves=[r.dist_key for r in escolhidos],
        pesos=w2,
        params=[r.params for r in escolhidos],
        peso_do_primeiro=primeiro,
        avisos=avisos,
    )


def simular_media_de_modelos(
    media: MediaDeModelos, n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Amostra da mistura: sorteia a familia por peso, depois o valor.

    Mistura, nao media de quantis. Sortear a familia iteracao a iteracao
    produz a distribuicao preditiva correta; fazer a media dos quantis das
    candidatas produziria uma curva que nao e a preditiva de nada - e que,
    pior, tem cauda mais leve que a mais pesada das candidatas.
    """
    rng = np.random.default_rng() if rng is None else rng
    escolha = rng.choice(len(media.pesos), size=n, p=media.pesos)
    u = rng.uniform(size=n)
    out = np.empty(n, dtype=float)
    for j in range(len(media.pesos)):
        sel = escolha == j
        if sel.any():
            out[sel] = FITTABLE[media.chaves[j]].ppf(u[sel], *media.params[j])
    return out
