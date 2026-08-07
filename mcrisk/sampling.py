"""
Esquemas de amostragem sobre o cubo unitario [0,1]^k.

Todos os metodos produzem uma matriz U (n x k) que depois e transformada
pelas funcoes quantil das marginais. Separar "gerar U" de "aplicar ppf"
e o que permite trocar Monte Carlo por Latin Hypercube sem tocar nas
distribuicoes.

Metodos:
  - "mc"  : Monte Carlo simples (amostragem aleatoria independente)
  - "lhs" : Latin Hypercube Sampling com jitter dentro do estrato
  - "lhs_median" : LHS com o ponto medio do estrato (deterministico dado
                   o embaralhamento; reduz variancia, mas subestima a
                   variabilidade dentro do estrato)

Referencias:
  - McKay, M.D., Beckman, R.J. & Conover, W.J. (1979). "A Comparison of Three
    Methods for Selecting Values of Input Variables in the Analysis of Output
    from a Computer Code". Technometrics 21(2):239-245.
    https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489755
  - Stein, M. (1987). "Large Sample Properties of Simulations Using Latin
    Hypercube Sampling". Technometrics 29(2):143-151.
    https://www.tandfonline.com/doi/abs/10.1080/00401706.1987.10488205
  - Helton, J.C. & Davis, F.J. (2003). "Latin hypercube sampling and the
    propagation of uncertainty in analyses of complex systems". Reliability
    Engineering & System Safety 81(1):23-69.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Method = Literal["mc", "lhs", "lhs_median"]

VALID_METHODS = ("mc", "lhs", "lhs_median")

# Evita u exatamente 0 ou 1, que levam ppf a +-inf em suportes ilimitados.
_EPS = np.finfo(float).eps


def unit_samples(
    n: int,
    k: int,
    method: Method = "lhs",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Gera matriz (n x k) de valores em (0,1)."""
    if n < 1:
        raise ValueError("n deve ser >= 1")
    if k < 1:
        raise ValueError("k deve ser >= 1")
    if method not in VALID_METHODS:
        raise ValueError(f"metodo invalido: {method}; use um de {VALID_METHODS}")
    rng = np.random.default_rng() if rng is None else rng

    if method == "mc":
        u = rng.random((n, k))
    else:
        # Um estrato por observacao, em cada dimensao independentemente.
        base = np.empty((n, k), dtype=float)
        strata = np.arange(n, dtype=float)
        for j in range(k):
            perm = rng.permutation(n)
            jitter = 0.5 if method == "lhs_median" else rng.random(n)
            base[:, j] = (strata[perm] + jitter) / n
        u = base

    # Recorta para o interior aberto de (0,1).
    return np.clip(u, _EPS, 1.0 - _EPS)


def effective_iterations_note(method: Method) -> str:
    """Texto honesto sobre o que muda no calculo do erro com LHS.

    Este e um ponto que costuma ser ignorado: sob LHS as observacoes NAO
    sao independentes, entao o erro padrao classico s/sqrt(n) nao e um
    estimador valido da incerteza de Monte Carlo. Ver `stats.replicate_error`.
    """
    if method == "mc":
        return (
            "Monte Carlo simples: as iteracoes sao i.i.d., portanto o erro "
            "padrao classico s/sqrt(n) e valido."
        )
    return (
        "Latin Hypercube: as iteracoes NAO sao independentes. O erro padrao "
        "s/sqrt(n) e apenas indicativo e tende a SUPERESTIMAR o erro real da "
        "media (Stein, 1987). Use replicacoes independentes para quantificar "
        "o erro corretamente."
    )
