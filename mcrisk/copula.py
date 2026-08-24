"""
Dependencia por copula: Gaussiana e t de Student.

Por que existe, sendo que ja ha Iman-Conover
--------------------------------------------
Iman-Conover impoe uma correlacao de POSTO reordenando as amostras. Preserva as
marginais exatamente e nao supoe forma de dependencia -- e por isso e o padrao
deste projeto. Mas ele so controla um numero por par: o rho de Spearman. Duas
estruturas com o mesmo rho e comportamentos de cauda radicalmente diferentes sao
indistinguiveis para ele.

Essa distincao e o problema central da modelagem de risco. Sob copula Gaussiana,
a dependencia de cauda e ZERO para qualquer rho < 1: eventos extremos conjuntos
ficam assintoticamente independentes. Sob copula t com v graus de liberdade, a
dependencia de cauda e positiva mesmo com rho = 0. Modelar carteira de credito,
sinistros ou custos de projeto com Gaussiana quando o fenomeno tem cauda conjunta
subestima sistematicamente a perda agregada -- foi a critica central a modelagem
de CDOs antes de 2008 (Embrechts et al., 2002; Donnelly & Embrechts, 2010).

O coeficiente de dependencia de cauda superior da copula t e

    lambda = 2 * t_{v+1}( -sqrt((v+1)(1-rho) / (1+rho)) )

que tende a zero quando v -> infinito, recuperando a Gaussiana. Esse limite e
verificado nos testes.

O que este modulo NAO faz
-------------------------
Nao estima a copula a partir de dados nem escolhe v por criterio de informacao.
O usuario informa a matriz e os graus de liberdade; a ferramenta aplica. Ajustar
copula com poucas observacoes e um exercicio ruim, e oferecer o botao daria a
impressao contraria.

Referencias
-----------
  - Sklar, A. (1959). "Fonctions de repartition a n dimensions et leurs marges".
    Publ. Inst. Statist. Univ. Paris 8:229-231.
  - Embrechts, P., McNeil, A. & Straumann, D. (2002). "Correlation and dependence
    in risk management: properties and pitfalls". In: Risk Management: Value at
    Risk and Beyond, Cambridge University Press, 176-223.
  - Demarta, S. & McNeil, A.J. (2005). "The t Copula and Related Copulas".
    International Statistical Review 73(1):111-129.
  - McNeil, A.J., Frey, R. & Embrechts, P. (2015). Quantitative Risk Management,
    2a ed., Princeton University Press, cap. 7.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .correlation import nearest_psd_correlation, spearman_to_pearson_normal

VALID_COPULAS = ("gaussian", "t")

# Preenchido ao final do modulo, depois de ARQUIMEDIANAS existir.
COPULAS_DISPONIVEIS: tuple[str, ...] = ()

# Abaixo de 2 graus de liberdade a t nao tem variancia; abaixo de 1, nem media.
# A copula continua definida, mas a leitura de "correlacao" perde sentido, e o
# usuario provavelmente digitou errado.
DF_MINIMO = 2.0


def _cholesky_psd(P: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cholesky com reparo PSD. Devolve (L, houve_reparo)."""
    P = np.asarray(P, dtype=float)
    try:
        return np.linalg.cholesky(P), False
    except np.linalg.LinAlgError:
        P2 = nearest_psd_correlation(P)
        # jitter minimo: Higham devolve PSD, e Cholesky exige DEFINIDA positiva
        eps = 1e-10
        for _ in range(8):
            try:
                return np.linalg.cholesky(P2 + eps * np.eye(P2.shape[0])), True
            except np.linalg.LinAlgError:
                eps *= 10
        raise


def gaussian_copula_u(
    P: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    """Amostras (n x k) no cubo unitario com dependencia Gaussiana.

    `P` e a matriz de correlacao LINEAR da normal latente. Quem tem alvo em
    Spearman deve converter antes com `spearman_to_pearson_normal`.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    k = P.shape[0]
    L, reparo = _cholesky_psd(P)
    Z = rng.standard_normal((n, k)) @ L.T
    return stats.norm.cdf(Z), reparo


def t_copula_u(
    P: np.ndarray, df: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    """Amostras (n x k) no cubo unitario com dependencia t de Student.

    Construcao padrao: Z ~ N(0, P), W ~ chi2_v / v independentes, e
    T = Z / sqrt(W). A divisao pelo MESMO W em todas as coordenadas e o que
    cria a dependencia de cauda -- um choque comum de escala que arrasta todas
    as marginais para o extremo ao mesmo tempo.
    """
    if not np.isfinite(df) or df < DF_MINIMO:
        raise ValueError(
            f"graus de liberdade da copula t devem ser >= {DF_MINIMO:g}; recebido {df!r}"
        )
    P = np.atleast_2d(np.asarray(P, dtype=float))
    k = P.shape[0]
    L, reparo = _cholesky_psd(P)
    Z = rng.standard_normal((n, k)) @ L.T
    W = rng.chisquare(df, size=n) / df
    T = Z / np.sqrt(W)[:, None]
    return stats.t.cdf(T, df=df), reparo


# A conversao Spearman -> Pearson por 2*sin(pi*rho_s/6) e EXATA para a copula
# Gaussiana e apenas aproximada para a t: o rho de Spearman da t depende tambem
# de v, sem forma fechada elementar. O desvio e pequeno (da ordem de 0,02 para
# v=3, rho=0,5) mas existe, e a tabela de correlacao obtida no relatorio e que
# deve ser lida como resultado -- nao a pedida.
def copula_u(
    kind: str,
    C: np.ndarray,
    n: int,
    rng: np.random.Generator,
    df: float = 5.0,
    spearman_adjust: bool = True,
) -> tuple[np.ndarray, bool]:
    """Interface unica. `C` e alvo de Spearman se `spearman_adjust`, senao Pearson.

    Para gaussian e t, `C` e usada inteira: cada par tem o seu rho. Para as
    arquimedianas (clayton, gumbel, frank), que sao permutaveis, so a MEDIA
    dos rho fora da diagonal e utilizavel - a heterogeneidade da matriz e
    descartada por construcao da familia, e quem chama precisa saber disso
    (`dispersao_fora_da_diagonal` mede o quanto se perde).
    """
    if kind not in COPULAS_DISPONIVEIS:
        raise ValueError(
            f"copula invalida: {kind!r}; use uma de {COPULAS_DISPONIVEIS}"
        )
    P = np.atleast_2d(np.asarray(C, dtype=float))
    d = P.shape[0]

    if kind in ARQUIMEDIANAS:
        rho = rho_medio_fora_da_diagonal(P)
        theta = theta_from_spearman(kind, rho)
        return archimedean_copula_u(kind, theta, n, d, rng), False

    if spearman_adjust:
        P = np.asarray(spearman_to_pearson_normal(P), dtype=float)
        np.fill_diagonal(P, 1.0)
    if kind == "gaussian":
        return gaussian_copula_u(P, n, rng)
    return t_copula_u(P, df, n, rng)


def tail_dependence_t(rho: float, df: float) -> float:
    """Coeficiente de dependencia de cauda da copula t bivariada.

    lambda = 2 * t_{v+1}( -sqrt((v+1)(1-rho)/(1+rho)) ).
    Vale para cauda superior e inferior (a copula t e radialmente simetrica).
    """
    rho = float(np.clip(rho, -0.999999, 0.999999))
    df = float(df)
    arg = -np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(arg, df=df + 1.0))


def tail_dependence_gaussian(rho: float) -> float:
    """Zero para todo rho < 1. Existe para deixar o contraste explicito no codigo."""
    return 1.0 if float(rho) >= 1.0 else 0.0


def empirical_tail_dependence(u: np.ndarray, v: np.ndarray, q: float = 0.99) -> float:
    """Estimador nao parametrico de lambda superior: P(V > q | U > q).

    Estimador ruidoso por construcao -- usa so a fracao (1-q) da amostra. Serve
    para conferir ordem de grandeza e o SINAL do contraste entre copulas, nao
    para medir lambda com precisao.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    acima = u > q
    if acima.sum() == 0:
        return float("nan")
    return float((v[acima] > q).mean())


# ===========================================================================
# Copulas arquimedianas: Clayton, Gumbel, Frank
# ===========================================================================
#
# A t resolve dependencia de cauda, mas e RADIALMENTE SIMETRICA: impoe a mesma
# forca de dependencia na cauda superior e na inferior. Muito risco real nao e
# assim. Carteiras de credito quebram juntas na queda e se descorrelacionam na
# alta; custos de projeto estouram juntos e economizam separados. Para esses
# fenomenos, cauda simetrica erra o lado que importa.
#
#   familia   cauda inferior            cauda superior           tau de Kendall
#   -------   -----------------------   ----------------------   --------------
#   Clayton   2^(-1/theta)  > 0         0                        theta/(theta+2)
#   Gumbel    0                         2 - 2^(1/theta)  > 0     1 - 1/theta
#   Frank     0                         0                        Debye (numerico)
#
# Frank nao tem dependencia de cauda em nenhum lado, como a Gaussiana. Existe
# aqui por outro motivo: e a unica das tres que admite dependencia NEGATIVA, e
# a unica com simetria radial exata. Serve de controle - se trocar Frank por
# Gaussiana muda o resultado, o que mudou nao foi a cauda.
#
# LIMITACAO ESTRUTURAL, nao de implementacao: as tres sao PERMUTAVEIS. Um unico
# parametro governa TODOS os pares. Nao existe "A e B muito dependentes, C
# pouco" - ou todos os pares tem a mesma dependencia, ou a familia nao serve.
# Gaussiana e t aceitam matriz completa; estas nao. Com k > 2 variaveis e
# dependencias heterogeneas, a escolha honesta continua sendo Gaussiana ou t.
#
# Amostragem por Marshall-Olkin (1988): sorteia-se uma variavel latente V
# ("frailty") e, dado V, as marginais ficam independentes. E exato, nao
# aproximado, e roda em O(n*d).
#
# Referencias
# -----------
#   - Marshall, A.W. & Olkin, I. (1988). "Families of Multivariate
#     Distributions". JASA 83(403):834-841.
#   - Nelsen, R.B. (2006). "An Introduction to Copulas", 2a ed., Springer.
#   - Joe, H. (2014). "Dependence Modeling with Copulas". CRC Press.
#   - Genest, C. & Rivest, L.-P. (1993). "Statistical Inference Procedures for
#     Bivariate Archimedean Copulas". JASA 88(423):1034-1043.

ARQUIMEDIANAS = ("clayton", "gumbel", "frank")

# Limites praticos. Fora deles a familia degenera numericamente: Clayton e
# Frank tendem a independencia perto de zero, e todas saturam em dependencia
# quase perfeita no extremo alto, onde a amostragem perde precisao em float64.
LIMITES_THETA = {
    "clayton": (1e-6, 50.0),
    "gumbel": (1.0, 50.0),
    "frank": (1e-6, 50.0),
}


def _valida_theta(kind: str, theta: float) -> float:
    if kind not in ARQUIMEDIANAS:
        raise ValueError(f"familia invalida: {kind!r}; use uma de {ARQUIMEDIANAS}")
    theta = float(theta)
    lo, hi = LIMITES_THETA[kind]
    if not np.isfinite(theta):
        raise ValueError(f"theta nao finito para {kind}: {theta!r}")
    if theta < lo or theta > hi:
        raise ValueError(
            f"theta = {theta:g} fora da faixa suportada para {kind}: [{lo:g}, {hi:g}]"
        )
    return theta


def _stable_positiva(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Estavel positiva com transformada de Laplace exp(-t^alpha), alpha em (0,1].

    Algoritmo de Chambers, Mallows & Stuck (1976). Com alpha = 1 a lei degenera
    em 1, que e o caso de independencia da Gumbel (theta = 1).
    """
    if alpha >= 1.0:
        return np.ones(n)
    u = rng.uniform(0.0, np.pi, size=n)
    w = rng.exponential(1.0, size=n)
    a = np.sin(alpha * u) / np.power(np.sin(u), 1.0 / alpha)
    b = np.power(np.sin((1.0 - alpha) * u) / w, (1.0 - alpha) / alpha)
    return a * b


def archimedean_copula_u(
    kind: str, theta: float, n: int, d: int, rng: np.random.Generator
) -> np.ndarray:
    """Amostra (n, d) no cubo unitario com dependencia arquimediana permutavel."""
    theta = _valida_theta(kind, theta)
    if n < 1:
        raise ValueError(f"n deve ser >= 1, recebido {n}")
    if d < 1:
        raise ValueError(f"d deve ser >= 1, recebido {d}")
    if d == 1:
        return rng.uniform(size=(n, 1))

    e = rng.exponential(1.0, size=(n, d))

    if kind == "clayton":
        v = rng.gamma(shape=1.0 / theta, scale=1.0, size=(n, 1))
        u = np.power(1.0 + e / v, -1.0 / theta)
    elif kind == "gumbel":
        if abs(theta - 1.0) < 1e-12:
            return rng.uniform(size=(n, d))
        v = _stable_positiva(1.0 / theta, n, rng).reshape(n, 1)
        u = np.exp(-np.power(e / v, 1.0 / theta))
    else:  # frank
        p = -np.expm1(-theta)  # 1 - exp(-theta), estavel para theta pequeno
        if p <= 0.0 or p >= 1.0:
            return rng.uniform(size=(n, d))
        v = stats.logser.rvs(p, size=n, random_state=rng).astype(float).reshape(n, 1)
        u = -np.log1p(-p * np.exp(-e / v)) / theta

    return np.clip(u, 1e-12, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# Calibracao: tau de Kendall <-> theta
# ---------------------------------------------------------------------------


def _debye1(x: float) -> float:
    """D_1(x) = (1/x) * integral_0^x t/(e^t - 1) dt."""
    from scipy import integrate

    if abs(x) < 1e-12:
        return 1.0
    val, _ = integrate.quad(lambda t: t / np.expm1(t), 0.0, x, limit=200)
    return float(val / x)


def tau_from_theta(kind: str, theta: float) -> float:
    """Tau de Kendall implicado pelo parametro da familia."""
    theta = _valida_theta(kind, theta)
    if kind == "clayton":
        return float(theta / (theta + 2.0))
    if kind == "gumbel":
        return float(1.0 - 1.0 / theta)
    return float(1.0 - 4.0 / theta * (1.0 - _debye1(theta)))


def theta_from_tau(kind: str, tau: float) -> float:
    """Inverso de `tau_from_theta`.

    Clayton e Gumbel tem forma fechada. Frank exige inverter a funcao de Debye,
    feito por bissecao - monotona e bem-comportada, converge sempre.
    """
    if kind not in ARQUIMEDIANAS:
        raise ValueError(f"familia invalida: {kind!r}; use uma de {ARQUIMEDIANAS}")
    tau = float(tau)
    if not -1.0 < tau < 1.0:
        raise ValueError(f"tau deve estar em (-1, 1), recebido {tau}")
    lo, hi = LIMITES_THETA[kind]
    if tau <= 0.0:
        if kind in ("clayton", "gumbel"):
            raise ValueError(
                f"a copula {kind} so representa dependencia POSITIVA; tau = "
                f"{tau:g} nao e atingivel. Use frank para dependencia negativa, "
                f"ou gaussian/t para uma matriz com sinais mistos."
            )
        raise ValueError(
            "tau <= 0 na frank exige theta negativo, que a amostragem por "
            "frailty deste modulo nao cobre; use gaussian ou t"
        )
    if kind == "clayton":
        theta = 2.0 * tau / (1.0 - tau)
    elif kind == "gumbel":
        theta = 1.0 / (1.0 - tau)
    else:
        from scipy import optimize

        tau_lo, tau_hi = tau_from_theta("frank", lo), tau_from_theta("frank", hi)
        if not tau_lo <= tau <= tau_hi:
            raise ValueError(
                f"tau = {tau:g} fora do alcance da frank nesta faixa de theta "
                f"[{tau_lo:.4f}, {tau_hi:.4f}]"
            )
        theta = float(
            optimize.brentq(lambda t: tau_from_theta("frank", t) - tau, lo, hi)
        )
    if theta > hi:
        raise ValueError(
            f"tau = {tau:g} exigiria theta = {theta:g}, acima do limite de "
            f"{hi:g} para {kind}. Dependencia tao forte nao e distinguivel de "
            f"comonotonia com precisao de float64."
        )
    return float(max(lo, min(hi, theta)))


def tail_dependence_archimedean(kind: str, theta: float) -> tuple[float, float]:
    """(lambda inferior, lambda superior) em forma fechada."""
    theta = _valida_theta(kind, theta)
    if kind == "clayton":
        return float(2.0 ** (-1.0 / theta)), 0.0
    if kind == "gumbel":
        return 0.0, float(2.0 - 2.0 ** (1.0 / theta))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Ajuste a dados: densidades bivariadas e maxima verossimilhanca
# ---------------------------------------------------------------------------


def pseudo_observacoes(x: np.ndarray) -> np.ndarray:
    """Transforma colunas em postos escalados para (0,1).

    Divide por (n+1), nao por n, para que nenhum ponto caia exatamente em 1 -
    onde as densidades de copula divergem. E a convencao padrao na literatura
    de inferencia semiparametrica de copulas (Genest & Rivest, 1993).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.ndim != 2:
        raise ValueError("x deve ser 2D (n_observacoes x n_variaveis)")
    n = x.shape[0]
    if n < 5:
        raise ValueError(f"{n} observacoes: poucas demais para ajustar copula")
    return stats.rankdata(x, axis=0) / (n + 1.0)


def log_densidade_bivariada(kind: str, u: np.ndarray, v: np.ndarray, theta: float):
    """log c(u, v; theta) para as tres familias arquimedianas."""
    theta = _valida_theta(kind, theta)
    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1.0 - 1e-12)

    if kind == "clayton":
        s = np.power(u, -theta) + np.power(v, -theta) - 1.0
        return (
            np.log1p(theta)
            - (theta + 1.0) * (np.log(u) + np.log(v))
            - (2.0 + 1.0 / theta) * np.log(s)
        )
    if kind == "gumbel":
        x, y = -np.log(u), -np.log(v)
        w = np.power(np.power(x, theta) + np.power(y, theta), 1.0 / theta)
        return (
            -w
            + (theta - 1.0) * (np.log(x) + np.log(y))
            - (np.log(u) + np.log(v))
            + (1.0 - 2.0 * theta) * np.log(w)
            + np.log(w + theta - 1.0)
        )
    # frank
    em = -np.expm1(-theta)  # 1 - e^-theta
    a = -np.expm1(-theta * u)
    b = -np.expm1(-theta * v)
    denom = em - a * b
    return (
        np.log(theta) + np.log(em) - theta * (u + v) - 2.0 * np.log(denom)
    )


class ResultadoCopula:
    """Ajuste de uma familia de copula a um par de series."""

    __slots__ = ("familia", "theta", "loglik", "aic", "bic", "tau", "lambda_inf",
                 "lambda_sup", "n", "avisos")

    def __init__(self, familia, theta, loglik, n, avisos=None):
        self.familia = familia
        self.theta = float(theta)
        self.loglik = float(loglik)
        self.n = int(n)
        self.aic = float(2.0 * 1 - 2.0 * loglik)
        self.bic = float(np.log(n) * 1 - 2.0 * loglik)
        self.tau = tau_from_theta(familia, theta)
        self.lambda_inf, self.lambda_sup = tail_dependence_archimedean(familia, theta)
        self.avisos = list(avisos or [])

    def as_record(self):
        return {
            "familia": self.familia,
            "theta": self.theta,
            "tau_kendall": self.tau,
            "lambda_inferior": self.lambda_inf,
            "lambda_superior": self.lambda_sup,
            "logL": self.loglik,
            "AIC": self.aic,
            "BIC": self.bic,
        }

    def __repr__(self):
        return (
            f"ResultadoCopula({self.familia}, theta={self.theta:.4g}, "
            f"tau={self.tau:.4f}, AIC={self.aic:.2f})"
        )


def fit_copula(
    x: np.ndarray,
    y: np.ndarray | None = None,
    familias: tuple[str, ...] = ARQUIMEDIANAS,
) -> list[ResultadoCopula]:
    """Ajusta familias arquimedianas a um par de series, ordenadas por AIC.

    Metodo semiparametrico em dois passos ("pseudo-maxima verossimilhanca"):
    as marginais viram postos - sem supor forma nenhuma para elas - e so o
    parametro de dependencia e estimado por maxima verossimilhanca. Ganha-se
    robustez: erro de especificacao na marginal nao contamina a estimativa da
    dependencia.

    O QUE A COMPARACAO POR AIC NAO DIZ. Ela ordena as candidatas testadas; nao
    afirma que a vencedora descreva bem os dados. Todas as tres podem estar
    erradas, e a diferenca entre elas concentra-se na cauda - a regiao com
    menos observacoes, onde a verossimilhanca tem menos a dizer. Trate o
    resultado como triagem, e olhe o lambda empirico antes de decidir.
    """
    from scipy import optimize

    if y is None:
        dados = np.asarray(x, dtype=float)
        if dados.ndim != 2 or dados.shape[1] != 2:
            raise ValueError("passe duas colunas, ou x e y separadamente")
    else:
        dados = np.column_stack([np.asarray(x, float), np.asarray(y, float)])

    finito = np.all(np.isfinite(dados), axis=1)
    avisos_gerais = []
    if not finito.all():
        avisos_gerais.append(
            f"{int((~finito).sum())} observacoes descartadas por nao serem finitas."
        )
    dados = dados[finito]
    n = dados.shape[0]
    if n < 20:
        avisos_gerais.append(
            f"So {n} observacoes. A estimativa do parametro de dependencia e "
            f"instavel e a comparacao entre familias, que se decide na cauda, "
            f"nao tem informacao suficiente para ser levada a serio."
        )

    U = pseudo_observacoes(dados)
    u, v = U[:, 0], U[:, 1]
    tau_amostral = float(stats.kendalltau(u, v).statistic)

    out: list[ResultadoCopula] = []
    for fam in familias:
        if fam not in ARQUIMEDIANAS:
            raise ValueError(f"familia invalida: {fam!r}; use de {ARQUIMEDIANAS}")
        lo, hi = LIMITES_THETA[fam]
        if tau_amostral <= 0 and fam in ("clayton", "gumbel"):
            avisos_gerais.append(
                f"tau amostral = {tau_amostral:.3f} (nao positivo): {fam} so "
                f"representa dependencia positiva e foi descartada."
            )
            continue

        def neg_ll(t, _fam=fam):
            try:
                val = -float(np.sum(log_densidade_bivariada(_fam, u, v, t)))
            except (ValueError, FloatingPointError):
                return np.inf
            return val if np.isfinite(val) else np.inf

        res = optimize.minimize_scalar(
            neg_ll, bounds=(lo + 1e-9, hi), method="bounded",
            options={"xatol": 1e-8},
        )
        if not res.success or not np.isfinite(res.fun):
            avisos_gerais.append(f"{fam}: a maximizacao nao convergiu; descartada.")
            continue
        out.append(ResultadoCopula(fam, res.x, -res.fun, n, avisos_gerais))

    if not out:
        raise ValueError(
            "nenhuma familia arquimediana pode ser ajustada a estes dados "
            f"(tau amostral = {tau_amostral:.3f})"
        )
    out.sort(key=lambda r: r.aic)
    return out


# ---------------------------------------------------------------------------
# Calibracao por Spearman: ligar as arquimedianas a matriz da interface
# ---------------------------------------------------------------------------
#
# O resto do pacote fala em rho de Spearman - e o que a grade da interface
# pede e o que o Iman-Conover mira. Para Gaussiana e t ha formula fechada
# ligando Spearman ao parametro. Para as arquimedianas nao ha forma fechada
# elementar, e inventar uma seria pior que nao ter.
#
# A saida e calibrar por medicao: simula-se a familia numa grade de theta com
# SEMENTE FIXA, mede-se o Spearman de cada ponto, e inverte-se a curva por
# interpolacao monotona. O resultado e deterministico (mesma entrada, mesmo
# theta) e o erro e mensuravel - esta no BENCHMARK.md, nao numa suposicao.
#
# O preco esta registrado: theta calibrado por interpolacao tem erro de
# calibracao proprio, somado ao erro de amostragem da simulacao do usuario. A
# tabela de correlacao OBTIDA continua sendo o numero a ler.

_GRADE_N = 60_000
_GRADE_SEED = 20260824
_grade_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _grade_spearman(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(thetas, rho_s medidos) crescentes, com semente fixa. Calculada uma vez."""
    if kind in _grade_cache:
        return _grade_cache[kind]
    lo, hi = LIMITES_THETA[kind]
    if kind == "gumbel":
        thetas = np.concatenate([[1.0], np.geomspace(1.02, hi, 39)])
    else:
        thetas = np.geomspace(max(lo, 1e-3), hi, 40)
    rng = np.random.default_rng(_GRADE_SEED)
    rhos = np.empty_like(thetas)
    for i, th in enumerate(thetas):
        U = archimedean_copula_u(kind, float(th), _GRADE_N, 2, rng)
        rhos[i] = float(stats.spearmanr(U[:, 0], U[:, 1]).statistic)
    # A curva e teoricamente monotona; ruido de amostragem pode inverter dois
    # pontos vizinhos. `maximum.accumulate` restaura a monotonia sem mexer na
    # forma, o que e requisito para a interpolacao inversa fazer sentido.
    rhos = np.maximum.accumulate(rhos)
    _grade_cache[kind] = (thetas, rhos)
    return thetas, rhos


def spearman_from_theta(kind: str, theta: float) -> float:
    """Rho de Spearman implicado por theta, por interpolacao da grade medida."""
    theta = _valida_theta(kind, theta)
    thetas, rhos = _grade_spearman(kind)
    return float(np.interp(theta, thetas, rhos))


def theta_from_spearman(kind: str, rho_s: float) -> float:
    """Theta que produz o rho de Spearman pedido, por interpolacao inversa."""
    if kind not in ARQUIMEDIANAS:
        raise ValueError(f"familia invalida: {kind!r}; use uma de {ARQUIMEDIANAS}")
    rho_s = float(rho_s)
    if not -1.0 < rho_s < 1.0:
        raise ValueError(f"rho de Spearman deve estar em (-1, 1), recebido {rho_s}")
    thetas, rhos = _grade_spearman(kind)
    if rho_s <= rhos[0]:
        if rho_s <= 0.0:
            raise ValueError(
                f"as copulas arquimedianas deste modulo so representam "
                f"dependencia POSITIVA; rho = {rho_s:g} nao e atingivel. Use "
                f"gaussian ou t para correlacao negativa ou de sinais mistos."
            )
        return float(thetas[0])
    if rho_s >= rhos[-1]:
        raise ValueError(
            f"rho = {rho_s:g} exige dependencia acima do que a familia {kind} "
            f"alcanca com theta <= {LIMITES_THETA[kind][1]:g} "
            f"(maximo medido: {rhos[-1]:.4f})"
        )
    return float(np.interp(rho_s, rhos, thetas))


def rho_medio_fora_da_diagonal(C: np.ndarray) -> float:
    """Media dos rho fora da diagonal - o unico numero que uma permutavel usa."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("a matriz de correlacao precisa ser quadrada")
    if C.shape[0] < 2:
        return 0.0
    fora = ~np.eye(C.shape[0], dtype=bool)
    return float(C[fora].mean())


def dispersao_fora_da_diagonal(C: np.ndarray) -> float:
    """Amplitude dos rho fora da diagonal, para medir o quanto se perde."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    if C.shape[0] < 2:
        return 0.0
    fora = ~np.eye(C.shape[0], dtype=bool)
    vals = C[fora]
    return float(vals.max() - vals.min())


COPULAS_DISPONIVEIS = VALID_COPULAS + ARQUIMEDIANAS
