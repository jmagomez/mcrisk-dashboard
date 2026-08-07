"""
Registro de distribuicoes de probabilidade.

Cada distribuicao e definida por um `DistSpec` que sabe:
  - quais parametros aceita e como valida-los;
  - como construir uma distribuicao congelada do scipy.stats;
  - como amostrar por transformada inversa (ppf), requisito para que
    Latin Hypercube Sampling e o metodo de Iman-Conover funcionem.

DECISAO DE PROJETO (e limitacao assumida): toda amostragem passa pela
funcao quantil (ppf). Isso garante que qualquer esquema de amostragem
sobre o cubo unitario [0,1]^k possa ser reaproveitado sem alterar a
marginal. O custo e que distribuicoes sem ppf fechada ficam mais lentas.

Referencias de parametrizacao:
  - scipy.stats (loc/scale) - https://docs.scipy.org/doc/scipy/reference/stats.html
  - PERT/BetaPERT: Vose, D. (2008) "Risk Analysis: A Quantitative Guide", 3a ed.,
    Wiley; ver tambem https://riskwiki.vosesoftware.com/PERTdistribution.php
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Infraestrutura
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """Um parametro escalar de uma distribuicao."""

    name: str
    label: str
    default: float | None = None  # None = obrigatorio, sem valor sugerido
    help: str = ""


@dataclass(frozen=True)
class DistSpec:
    """Especificacao de uma distribuicao disponivel no dashboard."""

    key: str
    label: str
    kind: str  # "continuous" | "discrete"
    params: Sequence[Param]
    builder: Callable[..., object]  # -> scipy frozen distribution
    validator: Callable[[Dict[str, float]], List[str]] = field(
        default=lambda p: []
    )
    notes: str = ""
    reference: str = ""

    def validate(self, params: Dict[str, float]) -> List[str]:
        """Retorna lista de mensagens de erro (vazia = valido)."""
        missing = [p.name for p in self.params if params.get(p.name) is None]
        if missing:
            return [f"parametro(s) ausente(s): {', '.join(missing)}"]
        bad = [
            p.name for p in self.params if not np.isfinite(float(params[p.name]))
        ]
        if bad:
            return [f"parametro(s) nao finito(s): {', '.join(bad)}"]
        return list(self.validator(params))

    def frozen(self, params: Dict[str, float]):
        errs = self.validate(params)
        if errs:
            raise ValueError(f"{self.label}: " + "; ".join(errs))
        kwargs = {p.name: float(params[p.name]) for p in self.params}
        return self.builder(**kwargs)

    def ppf(self, u: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Transformada inversa. `u` em (0,1)."""
        return np.asarray(self.frozen(params).ppf(u), dtype=float)


# ---------------------------------------------------------------------------
# Validadores reutilizaveis
# ---------------------------------------------------------------------------


def _positive(*names: str):
    def _v(p: Dict[str, float]) -> List[str]:
        return [f"{n} deve ser > 0" for n in names if float(p[n]) <= 0]

    return _v


def _ordered(*names: str):
    """Exige a < b < c ... (estritamente crescente)."""

    def _v(p: Dict[str, float]) -> List[str]:
        vals = [float(p[n]) for n in names]
        errs: List[str] = []
        for i in range(len(vals) - 1):
            if not vals[i] < vals[i + 1]:
                errs.append(f"exige {names[i]} < {names[i + 1]}")
        return errs

    return _v


def _combine(*vs):
    def _v(p: Dict[str, float]) -> List[str]:
        out: List[str] = []
        for v in vs:
            out.extend(v(p))
        return out

    return _v


def _prob(name: str):
    def _v(p: Dict[str, float]) -> List[str]:
        x = float(p[name])
        return [] if 0.0 <= x <= 1.0 else [f"{name} deve estar em [0, 1]"]

    return _v


# ---------------------------------------------------------------------------
# PERT (BetaPERT)
# ---------------------------------------------------------------------------


def pert_shape_params(
    minimo: float, moda: float, maximo: float, lam: float = 4.0
) -> tuple[float, float]:
    """Parametros alpha/beta da Beta subjacente a PERT.

    Convencao classica (lambda = 4):
        mu    = (min + lambda*moda + max) / (lambda + 2)
        alpha = 1 + lambda * (moda - min) / (max - min)
        beta  = 1 + lambda * (max - moda) / (max - min)

    Referencia: Vose (2008); Malcolm et al. (1959) para a origem no PERT.
    """
    rng = maximo - minimo
    if rng <= 0:
        raise ValueError("max deve ser > min")
    a = 1.0 + lam * (moda - minimo) / rng
    b = 1.0 + lam * (maximo - moda) / rng
    return a, b


def _pert_builder(minimo: float, moda: float, maximo: float, lam: float):
    a, b = pert_shape_params(minimo, moda, maximo, lam)
    return stats.beta(a, b, loc=minimo, scale=maximo - minimo)


def _triang_builder(minimo: float, moda: float, maximo: float):
    scale = maximo - minimo
    c = (moda - minimo) / scale
    return stats.triang(c, loc=minimo, scale=scale)


# ---------------------------------------------------------------------------
# Lognormal - duas parametrizacoes, porque a confusao entre elas e uma das
# fontes mais comuns de erro em modelagem de risco.
# ---------------------------------------------------------------------------


def _lognorm_log_builder(mu_log: float, sigma_log: float):
    """Parametros na escala LOG (mu e sigma de ln(X))."""
    return stats.lognorm(s=sigma_log, scale=np.exp(mu_log))


def _lognorm_real_builder(media: float, desvio: float):
    """Parametros na escala REAL (media e desvio-padrao de X)."""
    if media <= 0 or desvio <= 0:
        raise ValueError("media e desvio devem ser > 0")
    var = desvio ** 2
    sigma_log = np.sqrt(np.log(1.0 + var / media ** 2))
    mu_log = np.log(media) - 0.5 * sigma_log ** 2
    return stats.lognorm(s=sigma_log, scale=np.exp(mu_log))


# ---------------------------------------------------------------------------
# Discreta customizada e empirica
# ---------------------------------------------------------------------------


def discrete_ppf(
    u: np.ndarray, values: Sequence[float], probs: Sequence[float]
) -> np.ndarray:
    """ppf de uma discreta arbitraria, via CDF acumulada.

    Equivalente conceitual ao RiskDiscrete. As probabilidades sao
    normalizadas explicitamente; a UI avisa quando a soma difere de 1.
    """
    v = np.asarray(values, dtype=float)
    p = np.asarray(probs, dtype=float)
    if v.size != p.size or v.size == 0:
        raise ValueError("values e probs devem ter o mesmo tamanho, nao vazio")
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError("probabilidades invalidas")
    order = np.argsort(v)
    v, p = v[order], p[order] / p.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, np.asarray(u, dtype=float), side="left")
    idx = np.clip(idx, 0, v.size - 1)
    return v[idx]


def empirical_ppf(u: np.ndarray, data: Sequence[float]) -> np.ndarray:
    """Reamostragem da distribuicao empirica (sem suavizacao).

    ATENCAO: nao extrapola alem do minimo/maximo observados. Para caudas,
    isso subestima sistematicamente o risco extremo. Ver LIMITATIONS.md.
    """
    d = np.sort(np.asarray(data, dtype=float))
    if d.size == 0:
        raise ValueError("serie empirica vazia")
    idx = np.clip((np.asarray(u) * d.size).astype(int), 0, d.size - 1)
    return d[idx]


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

_SPECS: List[DistSpec] = [
    # ---------------- continuas ----------------
    DistSpec(
        key="normal",
        label="Normal",
        kind="continuous",
        params=[Param("mu", "media (mu)"), Param("sigma", "desvio-padrao (sigma)")],
        builder=lambda mu, sigma: stats.norm(loc=mu, scale=sigma),
        validator=_positive("sigma"),
        notes=(
            "Suporte infinito nos dois lados: pode gerar valores negativos. "
            "Cuidado ao usar para quantidades estritamente positivas (precos, "
            "custos, prazos)."
        ),
        reference=(
            "Johnson, Kotz & Balakrishnan (1994), Continuous Univariate "
            "Distributions v.1, cap.13"
        ),
    ),
    DistSpec(
        key="lognormal_log",
        label="Lognormal (parametros na escala log)",
        kind="continuous",
        params=[
            Param("mu_log", "mu de ln(X)"),
            Param("sigma_log", "sigma de ln(X)"),
        ],
        builder=_lognorm_log_builder,
        validator=_positive("sigma_log"),
        notes="mu_log e sigma_log NAO sao a media e o desvio de X.",
        reference="Johnson, Kotz & Balakrishnan (1994), v.1, cap.14",
    ),
    DistSpec(
        key="lognormal_real",
        label="Lognormal (media e desvio na escala real)",
        kind="continuous",
        params=[Param("media", "media de X"), Param("desvio", "desvio-padrao de X")],
        builder=_lognorm_real_builder,
        validator=_positive("media", "desvio"),
        notes="Converte internamente para a escala log. Equivale ao RiskLognorm.",
        reference="Vose (2008), Risk Analysis, 3a ed.",
    ),
    DistSpec(
        key="triangular",
        label="Triangular",
        kind="continuous",
        params=[
            Param("minimo", "minimo"),
            Param("moda", "mais provavel"),
            Param("maximo", "maximo"),
        ],
        builder=_triang_builder,
        validator=lambda p: (
            []
            if float(p["minimo"]) <= float(p["moda"]) <= float(p["maximo"])
            and float(p["maximo"]) > float(p["minimo"])
            else ["exige minimo <= moda <= maximo e maximo > minimo"]
        ),
        notes=(
            "Facil de elicitar, mas atribui peso alto as caudas e tem densidade "
            "com quinas. Trate min/max como limites REAIS, nao como percentis "
            "otimista/pessimista."
        ),
        reference="Vose (2008), cap. 'Triangular distribution'",
    ),
    DistSpec(
        key="pert",
        label="PERT (BetaPERT)",
        kind="continuous",
        params=[
            Param("minimo", "minimo"),
            Param("moda", "mais provavel"),
            Param("maximo", "maximo"),
            Param(
                "lam",
                "lambda (peso da moda)",
                default=4.0,
                help="4.0 e a convencao classica; valores maiores concentram na moda",
            ),
        ],
        builder=_pert_builder,
        validator=lambda p: (
            []
            if float(p["minimo"]) <= float(p["moda"]) <= float(p["maximo"])
            and float(p["maximo"]) > float(p["minimo"])
            and float(p["lam"]) > 0
            else ["exige minimo <= moda <= maximo, maximo > minimo e lambda > 0"]
        ),
        notes=(
            "Alternativa suavizada a Triangular, com menos massa nas caudas. "
            "Padrao em analise de cronograma."
        ),
        reference=(
            "Malcolm et al. (1959), Operations Research 7(5):646-669; Vose (2008)"
        ),
    ),
    DistSpec(
        key="uniform",
        label="Uniforme",
        kind="continuous",
        params=[Param("a", "minimo"), Param("b", "maximo")],
        builder=lambda a, b: stats.uniform(loc=a, scale=b - a),
        validator=_ordered("a", "b"),
        notes=(
            "Maxima entropia dado apenas o intervalo. Assume ignorancia total "
            "dentro dele."
        ),
        reference="Jaynes (2003), Probability Theory: The Logic of Science",
    ),
    DistSpec(
        key="beta",
        label="Beta (geral, com limites)",
        kind="continuous",
        params=[
            Param("alpha", "alpha"),
            Param("beta_", "beta"),
            Param("minimo", "minimo", default=0.0),
            Param("maximo", "maximo", default=1.0),
        ],
        builder=lambda alpha, beta_, minimo, maximo: stats.beta(
            alpha, beta_, loc=minimo, scale=maximo - minimo
        ),
        validator=_combine(_positive("alpha", "beta_"), _ordered("minimo", "maximo")),
        reference="Johnson, Kotz & Balakrishnan (1995), v.2, cap.25",
    ),
    DistSpec(
        key="gamma",
        label="Gama",
        kind="continuous",
        params=[
            Param("forma", "forma (k)"),
            Param("escala", "escala (theta)"),
            Param("loc", "deslocamento", default=0.0),
        ],
        builder=lambda forma, escala, loc: stats.gamma(forma, loc=loc, scale=escala),
        validator=_positive("forma", "escala"),
        reference="Johnson, Kotz & Balakrishnan (1994), v.1, cap.17",
    ),
    DistSpec(
        key="exponential",
        label="Exponencial",
        kind="continuous",
        params=[
            Param("escala", "escala (1/taxa)"),
            Param("loc", "deslocamento", default=0.0),
        ],
        builder=lambda escala, loc: stats.expon(loc=loc, scale=escala),
        validator=_positive("escala"),
        notes="Sem memoria. Frequentemente usada para tempo entre eventos.",
        reference="Johnson, Kotz & Balakrishnan (1994), v.1, cap.19",
    ),
    DistSpec(
        key="weibull",
        label="Weibull",
        kind="continuous",
        params=[
            Param("forma", "forma (c)"),
            Param("escala", "escala"),
            Param("loc", "deslocamento", default=0.0),
        ],
        builder=lambda forma, escala, loc: stats.weibull_min(
            forma, loc=loc, scale=escala
        ),
        validator=_positive("forma", "escala"),
        notes="Padrao em confiabilidade e analise de vida util.",
        reference="Johnson, Kotz & Balakrishnan (1994), v.1, cap.21",
    ),
    DistSpec(
        key="student_t",
        label="t de Student (escalonada)",
        kind="continuous",
        params=[
            Param("gl", "graus de liberdade"),
            Param("loc", "locacao", default=0.0),
            Param("escala", "escala", default=1.0),
        ],
        builder=lambda gl, loc, escala: stats.t(gl, loc=loc, scale=escala),
        validator=_positive("gl", "escala"),
        notes=(
            "Caudas mais pesadas que a Normal. Com gl <= 2 a variancia e infinita "
            "e com gl <= 1 a media nao existe: estatisticas amostrais nao convergem."
        ),
        reference="Johnson, Kotz & Balakrishnan (1995), v.2, cap.28",
    ),
    DistSpec(
        key="logistic",
        label="Logistica",
        kind="continuous",
        params=[Param("loc", "locacao"), Param("escala", "escala")],
        builder=lambda loc, escala: stats.logistic(loc=loc, scale=escala),
        validator=_positive("escala"),
        reference="Johnson, Kotz & Balakrishnan (1995), v.2, cap.23",
    ),
    DistSpec(
        key="gumbel_r",
        label="Gumbel (maximos, VEG tipo I)",
        kind="continuous",
        params=[Param("loc", "locacao"), Param("escala", "escala")],
        builder=lambda loc, escala: stats.gumbel_r(loc=loc, scale=escala),
        validator=_positive("escala"),
        notes="Distribuicao limite de maximos de blocos para caudas exponenciais.",
        reference=(
            "Coles (2001), An Introduction to Statistical Modeling of Extreme Values"
        ),
    ),
    DistSpec(
        key="pareto",
        label="Pareto (tipo I)",
        kind="continuous",
        params=[Param("b", "indice de cauda (b)"), Param("escala", "escala (x_m)")],
        builder=lambda b, escala: stats.pareto(b, scale=escala),
        validator=_positive("b", "escala"),
        notes=(
            "Cauda pesada: com b <= 2 a variancia e infinita, com b <= 1 a media "
            "e infinita. Medias amostrais NAO estabilizam nesses regimes."
        ),
        reference="Embrechts, Kluppelberg & Mikosch (1997), Modelling Extremal Events",
    ),
    DistSpec(
        key="genpareto",
        label="Pareto Generalizada (GPD, excedencias)",
        kind="continuous",
        params=[
            Param("xi", "indice de forma (xi)"),
            Param("loc", "limiar (u)", default=0.0),
            Param("escala", "escala", default=1.0),
        ],
        builder=lambda xi, loc, escala: stats.genpareto(xi, loc=loc, scale=escala),
        validator=_positive("escala"),
        notes="Para modelagem de excedencias sobre um limiar (POT).",
        reference="Coles (2001), cap.4; Embrechts et al. (1997)",
    ),
    # ---------------- discretas ----------------
    DistSpec(
        key="bernoulli",
        label="Bernoulli (evento sim/nao)",
        kind="discrete",
        params=[Param("p", "probabilidade de ocorrencia")],
        builder=lambda p: stats.bernoulli(p),
        validator=_prob("p"),
        notes="Util como gatilho de risco: multiplique pelo impacto.",
        reference="Johnson, Kemp & Kotz (2005), Univariate Discrete Distributions",
    ),
    DistSpec(
        key="binomial",
        label="Binomial",
        kind="discrete",
        params=[Param("n", "numero de tentativas"), Param("p", "probabilidade")],
        builder=lambda n, p: stats.binom(int(round(n)), p),
        validator=_combine(
            _prob("p"), lambda p: [] if float(p["n"]) >= 1 else ["n deve ser >= 1"]
        ),
        reference="Johnson, Kemp & Kotz (2005), cap.3",
    ),
    DistSpec(
        key="poisson",
        label="Poisson",
        kind="discrete",
        params=[Param("lam", "taxa media (lambda)")],
        builder=lambda lam: stats.poisson(lam),
        validator=_positive("lam"),
        notes="Assume media = variancia. Dados reais costumam ser superdispersos.",
        reference="Johnson, Kemp & Kotz (2005), cap.4",
    ),
    DistSpec(
        key="negbinom",
        label="Binomial Negativa",
        kind="discrete",
        params=[Param("n", "numero de sucessos"), Param("p", "probabilidade")],
        builder=lambda n, p: stats.nbinom(int(round(n)), p),
        validator=_combine(
            _prob("p"), lambda p: [] if float(p["n"]) >= 1 else ["n deve ser >= 1"]
        ),
        notes="Alternativa a Poisson quando ha superdispersao.",
        reference="Johnson, Kemp & Kotz (2005), cap.5",
    ),
    DistSpec(
        key="geometric",
        label="Geometrica",
        kind="discrete",
        params=[Param("p", "probabilidade de sucesso")],
        builder=lambda p: stats.geom(p),
        validator=lambda p: (
            [] if 0 < float(p["p"]) <= 1 else ["p deve estar em (0, 1]"]
        ),
        reference="Johnson, Kemp & Kotz (2005), cap.5",
    ),
    DistSpec(
        key="discrete_uniform",
        label="Uniforme Discreta (inteiros)",
        kind="discrete",
        params=[Param("a", "minimo (inclusive)"), Param("b", "maximo (inclusive)")],
        builder=lambda a, b: stats.randint(int(round(a)), int(round(b)) + 1),
        validator=lambda p: [] if float(p["a"]) < float(p["b"]) else ["exige a < b"],
        reference="Johnson, Kemp & Kotz (2005), cap.10",
    ),
]

REGISTRY: Dict[str, DistSpec] = {s.key: s for s in _SPECS}

# Distribuicoes que precisam de entrada tabular, tratadas fora do REGISTRY
SPECIAL_KINDS = {
    "discrete_custom": "Discreta customizada (valores + probabilidades)",
    "empirical": "Empirica (reamostragem de dados historicos)",
}


def list_distributions(kind: str | None = None) -> List[DistSpec]:
    out = list(_SPECS)
    if kind:
        out = [s for s in out if s.kind == kind]
    return out


def get(key: str) -> DistSpec:
    if key not in REGISTRY:
        raise KeyError(f"distribuicao desconhecida: {key}")
    return REGISTRY[key]


def theoretical_moments(key: str, params: Dict[str, float]) -> Dict[str, float]:
    """Momentos teoricos, quando existem. Retorna NaN quando indefinidos."""
    fr = get(key).frozen(params)
    with np.errstate(all="ignore"):
        m, v, s, k = fr.stats(moments="mvsk")
    return {
        "media": float(m),
        "variancia": float(v),
        "desvio": float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan"),
        "assimetria": float(s),
        "curtose_excesso": float(k),
    }
