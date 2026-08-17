"""
mcrisk - motor de simulacao de Monte Carlo para analise quantitativa de risco.

Modulos:
    distributions : registro de distribuicoes e transformada inversa
    sampling      : Monte Carlo simples e Latin Hypercube Sampling
    correlation   : correlacao de posto por Iman-Conover, reparo PSD
    copula        : dependencia por copula Gaussiana e t (dependencia de cauda)
    formula       : avaliacao segura de formulas de saida (sem eval cru)
    engine        : orquestracao da simulacao
    summary       : estatisticas de saida e erro de simulacao
    sensitivity   : indices de sensibilidade e ordenacao para tornado
    fitting       : ajuste de distribuicoes a dados com AIC/BIC e bootstrap
    scenarios     : analise condicional (sem re-simular) e estresse (re-simulando)

A metodologia e as limitacoes estao documentadas em METHODOLOGY.md e
LIMITATIONS.md, na raiz do repositorio.
"""

__version__ = "0.2.0"

from . import (  # noqa: F401
    copula,
    correlation,
    distributions,
    engine,
    fitting,
    formula,
    sampling,
    scenarios,
    sensitivity,
    summary,
)

__all__ = [
    "copula",
    "correlation",
    "distributions",
    "engine",
    "fitting",
    "formula",
    "sampling",
    "scenarios",
    "sensitivity",
    "summary",
]
