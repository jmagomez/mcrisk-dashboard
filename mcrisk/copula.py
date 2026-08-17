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
    """Interface unica. `C` e alvo de Spearman se `spearman_adjust`, senao Pearson."""
    if kind not in VALID_COPULAS:
        raise ValueError(f"copula invalida: {kind!r}; use uma de {VALID_COPULAS}")
    P = np.atleast_2d(np.asarray(C, dtype=float))
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
