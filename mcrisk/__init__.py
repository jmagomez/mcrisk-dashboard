"""
mcrisk - motor de simulacao de Monte Carlo para analise quantitativa de risco.

Modulos:
    distributions : registro de distribuicoes e transformada inversa
    sampling      : Monte Carlo simples e Latin Hypercube Sampling
    correlation   : correlacao de posto por Iman-Conover, reparo PSD
    copula        : cinco copulas (Gaussiana, t, Clayton, Gumbel, Frank),
                    dependencia de cauda em forma fechada, ajuste a dados
    formula       : avaliacao segura de formulas de saida (sem eval cru)
    engine        : orquestracao da simulacao
    summary       : estatisticas de saida, medidas de risco de queda e erro
                    de simulacao
    convergence   : criterio de parada com tolerancia e nivel de confianca
    sensitivity   : quatro metodos de sensibilidade e ordenacao para tornado
    fitting       : ajuste de distribuicoes a dados com AIC/BIC e bootstrap,
                    propagacao da incerteza de parametro e media de modelos
    scenarios     : analise condicional (sem re-simular), estresse
                    (re-simulando) e significancia das entradas

A metodologia e as limitacoes estao documentadas em METHODOLOGY.md e
LIMITATIONS.md, e as medicoes de qualidade, velocidade e confiabilidade em
BENCHMARK.md, na raiz do repositorio.
"""

__version__ = "0.3.0"

from . import (  # noqa: F401
    convergence,
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
    "convergence",
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
