"""
Correlacao entre entradas pelo metodo de Iman-Conover.

O metodo induz uma correlacao de POSTO (Spearman) alvo entre colunas de
uma amostra, reordenando os valores de cada coluna. Como so reordena,
as marginais sao preservadas EXATAMENTE - essa e a propriedade que torna
o metodo "distribution-free".

Algoritmo (Iman & Conover, 1982):
  1. Monta matriz de escores M (n x k) em que cada coluna e uma permutacao
     aleatoria dos escores de van der Waerden a_i = Phi^-1(i/(n+1)).
  2. E = corr(M); Cholesky E = F F'.
  3. Alvo C = P P' (Cholesky).
  4. M* = M (F^-1)' P'  ->  corr(M*) ~= C.
  5. Reordena cada coluna da amostra real para que seus postos coincidam
     com os postos de M*.

PONTO CRITICO (implementado como opcao, com teste empirico no repositorio):
o passo 2-4 controla a correlacao de PEARSON dos escores normais. Para
escores normais vale a relacao rho_S = (6/pi) * arcsin(rho_P / 2), de modo
que mirar rho_P = alvo produz um Spearman final levemente ABAIXO do alvo.
A correcao inversa e rho_P = 2 * sin(pi * rho_S / 6). O parametro
`spearman_adjust` liga/desliga essa correcao; o teste
`tests/test_correlation.py` mede qual das duas fica mais proxima do alvo
em vez de assumir.

Referencias:
  - Iman, R.L. & Conover, W.J. (1982). "A distribution-free approach to
    inducing rank correlation among input variables". Communications in
    Statistics - Simulation and Computation 11(3):311-334.
    https://www.tandfonline.com/doi/abs/10.1080/03610918208812265
  - Higham, N.J. (2002). "Computing the nearest correlation matrix - a problem
    from finance". IMA Journal of Numerical Analysis 22(3):329-343.
  - Kruskal, W.H. (1958). "Ordinal Measures of Association". JASA 53:814-861.
    (relacao arcsin entre Pearson e Spearman no caso normal bivariado)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Leitura de uma matriz preenchida pela metade
# ---------------------------------------------------------------------------


def mirror_triangle(
    C: np.ndarray, tol: float = 1e-9
) -> Tuple[np.ndarray, List[Tuple[int, int, float, float]]]:
    """Torna simetrica uma matriz preenchida em apenas um dos triangulos.

    POR QUE ISSO NAO E `(C + C.T) / 2`
    ----------------------------------
    Se a interface pede "preencha apenas um triangulo" e o usuario digita
    0,8 acima da diagonal deixando 0 abaixo, a media dos dois lados devolve
    0,4. A simulacao roda, nao reclama de nada, e usa metade da correlacao
    pedida. E o pior tipo de defeito: nao quebra, so mente.

    Regra aplicada a cada par (i, j):
      - so um lado preenchido  -> espelha o lado preenchido;
      - os dois iguais         -> usa o valor;
      - os dois preenchidos e diferentes -> AMBIGUO: usa o triangulo
        superior e registra o conflito para o chamador avisar;
      - nenhum preenchido      -> 0.

    Um zero explicito e indistinguivel de "nao preenchi" numa grade
    numerica. A escolha aqui e assumir que zero significa "nao preenchi",
    que e o caso comum; quem quiser fixar zero de verdade pode preencher os
    dois lados com zero, e nenhum conflito sera reportado.

    Retorna
    -------
    (matriz simetrica com diagonal 1, lista de conflitos (i, j, acima, abaixo))
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("a matriz precisa ser quadrada")
    n = C.shape[0]
    out = np.eye(n, dtype=float)
    conflitos: List[Tuple[int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            acima, abaixo = float(C[i, j]), float(C[j, i])
            vazio_acima = not np.isfinite(acima) or abs(acima) <= tol
            vazio_abaixo = not np.isfinite(abaixo) or abs(abaixo) <= tol
            if vazio_acima and vazio_abaixo:
                valor = 0.0
            elif vazio_abaixo:
                valor = acima
            elif vazio_acima:
                valor = abaixo
            elif abs(acima - abaixo) <= tol:
                valor = acima
            else:
                valor = acima
                conflitos.append((i, j, acima, abaixo))
            out[i, j] = out[j, i] = valor
    return out, conflitos


# ---------------------------------------------------------------------------
# Diagnostico e reparo da matriz alvo
# ---------------------------------------------------------------------------


def check_correlation_matrix(C: np.ndarray) -> list[str]:
    """Valida simetria, diagonal unitaria, faixa [-1,1] e positividade."""
    C = np.asarray(C, dtype=float)
    errs: list[str] = []
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        return ["matriz de correlacao deve ser quadrada"]
    if not np.allclose(C, C.T, atol=1e-8):
        errs.append("matriz nao e simetrica")
    if not np.allclose(np.diag(C), 1.0, atol=1e-8):
        errs.append("diagonal deve ser 1.0")
    if np.any(np.abs(C) > 1.0 + 1e-12):
        errs.append("ha coeficientes fora de [-1, 1]")
    if not errs:
        eig = np.linalg.eigvalsh(C)
        if eig.min() < -1e-10:
            errs.append(
                f"matriz nao e positiva semidefinida (menor autovalor = "
                f"{eig.min():.4g}); as correlacoes especificadas sao mutuamente "
                f"incompativeis"
            )
    return errs


def nearest_psd_correlation(
    C: np.ndarray, eps: float = 1e-10, max_iter: int = 200
) -> np.ndarray:
    """Matriz de correlacao positiva semidefinida mais proxima.

    Implementacao do metodo de projecoes alternadas de Higham (2002),
    simplificado. Devolve matriz simetrica, diagonal 1, PSD.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    Y = (C + C.T) / 2.0
    dS = np.zeros_like(Y)
    for _ in range(max_iter):
        R = Y - dS
        # Projecao no cone PSD
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        w = np.clip(w, eps, None)
        X = (V * w) @ V.T
        dS = X - R
        # Projecao no conjunto de diagonal unitaria
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
        if np.linalg.eigvalsh(Y).min() > eps:
            break
    Y = (Y + Y.T) / 2.0
    np.fill_diagonal(Y, 1.0)
    return Y


def spearman_to_pearson_normal(rho_s: np.ndarray | float) -> np.ndarray | float:
    """rho_P = 2 * sin(pi * rho_S / 6), valido para a normal bivariada."""
    return 2.0 * np.sin(np.pi * np.asarray(rho_s, dtype=float) / 6.0)


def pearson_to_spearman_normal(rho_p: np.ndarray | float) -> np.ndarray | float:
    """rho_S = (6/pi) * arcsin(rho_P / 2), valido para a normal bivariada."""
    return (6.0 / np.pi) * np.arcsin(np.asarray(rho_p, dtype=float) / 2.0)


# ---------------------------------------------------------------------------
# Iman-Conover
# ---------------------------------------------------------------------------


def van_der_waerden_scores(n: int) -> np.ndarray:
    """a_i = Phi^-1(i / (n+1)), i = 1..n."""
    i = np.arange(1, n + 1, dtype=float)
    return stats.norm.ppf(i / (n + 1.0))


def iman_conover(
    X: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator | None = None,
    spearman_adjust: bool = True,
    repair_psd: bool = True,
) -> np.ndarray:
    """Reordena as colunas de X para induzir a correlacao de posto alvo.

    Parametros
    ----------
    X : (n, k) amostra ja transformada pelas marginais.
    target : (k, k) matriz de correlacao de SPEARMAN desejada.
    spearman_adjust : aplica rho_P = 2 sin(pi rho_S / 6) antes da fatoracao.
    repair_psd : se a matriz alvo nao for PSD, projeta na PSD mais proxima
        em vez de falhar. Emite o resultado reparado silenciosamente - o
        chamador deve avisar o usuario (a UI faz isso).

    Retorna
    -------
    (n, k) com as MESMAS marginais de X, apenas reordenadas por coluna.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    C = np.asarray(target, dtype=float)
    if C.shape != (k, k):
        raise ValueError(f"target deve ser {k}x{k}, recebido {C.shape}")
    rng = np.random.default_rng() if rng is None else rng

    if k == 1:
        return X.copy()

    C_work = spearman_to_pearson_normal(C) if spearman_adjust else C.copy()
    np.fill_diagonal(C_work, 1.0)

    if np.linalg.eigvalsh(C_work).min() <= 0:
        if not repair_psd:
            raise np.linalg.LinAlgError(
                "matriz alvo nao e positiva definida; correlacoes incompativeis"
            )
        C_work = nearest_psd_correlation(C_work)

    # 1. escores embaralhados
    scores = van_der_waerden_scores(n)
    M = np.empty((n, k), dtype=float)
    for j in range(k):
        M[:, j] = scores[rng.permutation(n)]

    # 2. descorrelaciona os escores
    E = np.corrcoef(M, rowvar=False)
    # E pode ficar quase singular por acaso; regulariza minimamente
    E = E + np.eye(k) * 1e-12
    F = np.linalg.cholesky(E)
    P = np.linalg.cholesky(C_work)

    # 3. impoe a estrutura alvo
    M_star = M @ np.linalg.inv(F).T @ P.T

    # 4. transfere a ordenacao para X, coluna a coluna
    out = np.empty_like(X)
    for j in range(k):
        target_rank = np.argsort(np.argsort(M_star[:, j]))
        sorted_col = np.sort(X[:, j])
        out[:, j] = sorted_col[target_rank]
    return out


def achieved_spearman(X: np.ndarray) -> np.ndarray:
    """Correlacao de Spearman efetivamente obtida na amostra."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 1:
        return np.ones((1, 1))
    rho = stats.spearmanr(X).statistic
    return np.atleast_2d(np.asarray(rho, dtype=float))


def correlation_error(
    X: np.ndarray, target: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """(erro maximo absoluto, erro medio absoluto, matriz de diferencas)."""
    got = achieved_spearman(X)
    diff = got - np.asarray(target, dtype=float)
    off = ~np.eye(diff.shape[0], dtype=bool)
    return float(np.abs(diff[off]).max()), float(np.abs(diff[off]).mean()), diff
