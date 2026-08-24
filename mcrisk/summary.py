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


# ===========================================================================
# Medidas de dispersao e forma que a media e o desvio nao capturam
# ===========================================================================


def mean_absolute_deviation(x: np.ndarray) -> float:
    """Media dos desvios absolutos em torno da media.

    Alternativa ao desvio-padrao que NAO eleva ao quadrado, e por isso da
    muito menos peso aos extremos. Em distribuicoes de cauda pesada as duas
    medidas contam historias diferentes, e a diferenca entre elas ja e um
    diagnostico: quanto maior a razao desvio/MAD, mais a dispersao esta
    concentrada em poucas observacoes extremas.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x - x.mean())))


def semi_variance(x: np.ndarray, threshold: float | None = None) -> float:
    """Variancia calculada so sobre os valores ABAIXO do limiar.

    Risco de queda ("downside risk"). O desvio-padrao comum trata surpresa
    boa e surpresa ruim como equivalentes, o que raramente corresponde a
    preferencia de quem decide: ninguem se protege contra lucro inesperado.

    O limiar padrao e a media da amostra. Passar outro valor permite medir a
    dispersao abaixo de uma meta (um retorno minimo aceitavel, um orcamento).

    Denominador: n de TODA a amostra, nao o numero de observacoes abaixo do
    limiar. E a convencao usual em financas, e a razao e comparabilidade -
    com denominador variavel, uma distribuicao com poucas observacoes ruins
    mas muito ruins pareceria mais arriscada que outra com muitas.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    alvo = float(np.mean(x)) if threshold is None else float(threshold)
    desvios = np.minimum(x - alvo, 0.0)
    return float(np.sum(desvios**2) / (x.size - 1))


def semi_std(x: np.ndarray, threshold: float | None = None) -> float:
    """Raiz da semi-variancia, na mesma unidade da saida."""
    sv = semi_variance(x, threshold)
    return float(np.sqrt(sv)) if np.isfinite(sv) else float("nan")


def value_range(x: np.ndarray) -> float:
    """Maximo menos minimo.

    Estatistica instavel por construcao: cresce indefinidamente com o numero
    de iteracoes em distribuicoes ilimitadas, e e viesada para baixo mesmo nas
    limitadas. Existe porque e pedida com frequencia, nao porque seja boa
    medida de dispersao - para isso, use o intervalo interpercentil.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(x.max() - x.min())


def mode(x: np.ndarray, bins: int | str = "auto") -> float:
    """Moda amostral.

    Para dados discretos, o valor mais frequente. Para dados continuos, o
    centro do intervalo mais povoado de um histograma - o que torna a moda
    dependente do numero de intervalos e, portanto, a MENOS estavel das tres
    medidas de tendencia central. Reportada porque e pedida, com esta
    ressalva anexada.

    A deteccao de "discreto" e pratica, nao teorica: poucos valores distintos
    em relacao ao tamanho da amostra.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    unicos, contagens = np.unique(x, return_counts=True)
    if unicos.size <= max(2, min(50, x.size // 20)):
        return float(unicos[int(np.argmax(contagens))])
    freq, bordas = np.histogram(x, bins=bins)
    j = int(np.argmax(freq))
    return float((bordas[j] + bordas[j + 1]) / 2.0)


def kurtosis_pearson(x: np.ndarray) -> float:
    """Curtose na convencao de Pearson: 3 para a normal, nao 0.

    Existem duas convencoes em uso e elas diferem por exatamente 3. O resto
    deste pacote reporta a de EXCESSO (normal = 0), que e o padrao do SciPy e
    do NumPy; o @RISK reporta esta. Ter as duas nomeadas evita o erro de
    comparar um numero desta ferramenta com um de outra sem notar o
    deslocamento - que faz uma distribuicao normal parecer ter cauda pesada.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    # bias=True para casar com `describe`, que usa o padrao do SciPy. As duas
    # convencoes precisam diferir por EXATAMENTE 3, que e a unica razao de esta
    # funcao existir; misturar o estimador viesado com o nao viesado faria a
    # diferenca ser 3,00006 e destruiria o proposito.
    return float(stats.kurtosis(x, fisher=False, bias=True))


def target_probability(x: np.ndarray, target: float) -> float:
    """P(X <= alvo). Nomenclatura X->P: do valor para a probabilidade."""
    return prob_below(x, target)


def percentile_descending(x: np.ndarray, q: float) -> float:
    """Percentil na convencao DESCENDENTE: q% da probabilidade ACIMA do valor.

    Parte da literatura de risco - e parte das ferramentas comerciais - usa a
    convencao oposta a do `numpy.percentile`. "P5" pode significar tanto o
    valor com 5% abaixo quanto o valor com 5% acima, e os dois numeros sao
    muito diferentes numa cauda. A funcao existe para tornar a escolha
    explicita no codigo em vez de implicita na cabeca de quem le.
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentil deve estar em [0, 100], recebido {q}")
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, 100.0 - q))
