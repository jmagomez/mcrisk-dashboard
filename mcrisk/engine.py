"""
Orquestrador da simulacao: modelo -> amostragem -> correlacao -> saida.

Fluxo:
  1. Gera U (n x k) pelo esquema escolhido (MC ou LHS).
  2. Aplica a ppf de cada marginal, coluna a coluna.
  3. Se ha matriz de correlacao, aplica Iman-Conover (reordena as colunas,
     preservando as marginais exatamente).
  4. Avalia a formula de saida de forma vetorizada.

Reprodutibilidade: toda a aleatoriedade vem de um unico
`numpy.random.Generator` construido a partir de `seed`. Mesma seed +
mesma especificacao = mesmos resultados, bit a bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from . import correlation as corr_mod
from . import distributions as dists
from . import sampling
from .formula import Formula


@dataclass
class Variable:
    """Uma entrada incerta do modelo."""

    name: str  # identificador usado na formula
    label: str  # rotulo exibido
    dist_key: str  # chave do REGISTRY, ou "discrete_custom"/"empirical"
    params: Dict[str, float] = field(default_factory=dict)
    values: List[float] | None = None  # discrete_custom
    probs: List[float] | None = None  # discrete_custom
    data: List[float] | None = None  # empirical

    def validate(self) -> List[str]:
        errs: List[str] = []
        if not self.name:
            errs.append("nome vazio")
        if self.dist_key == "discrete_custom":
            if not self.values or not self.probs:
                errs.append("informe valores e probabilidades")
            elif len(self.values) != len(self.probs):
                errs.append("valores e probabilidades com tamanhos diferentes")
            elif any(p < 0 for p in self.probs):
                errs.append("probabilidades negativas")
            elif sum(self.probs) <= 0:
                errs.append("soma das probabilidades deve ser > 0")
        elif self.dist_key == "empirical":
            if not self.data or len(self.data) < 2:
                errs.append("serie empirica precisa de ao menos 2 observacoes")
        else:
            try:
                errs.extend(dists.get(self.dist_key).validate(self.params))
            except KeyError as e:
                errs.append(str(e))
        return errs

    def ppf(self, u: np.ndarray) -> np.ndarray:
        if self.dist_key == "discrete_custom":
            return dists.discrete_ppf(u, self.values or [], self.probs or [])
        if self.dist_key == "empirical":
            return dists.empirical_ppf(u, self.data or [])
        return dists.get(self.dist_key).ppf(u, self.params)


@dataclass
class SimulationSpec:
    variables: List[Variable]
    formula: str
    iterations: int = 10_000
    method: str = "lhs"
    seed: int | None = 12345
    correlation: np.ndarray | None = None  # Spearman alvo (k x k)
    spearman_adjust: bool = True

    def validate(self) -> List[str]:
        errs: List[str] = []
        if not self.variables:
            errs.append("defina ao menos uma variavel de entrada")
        names = [v.name for v in self.variables]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            errs.append(f"nomes de variaveis duplicados: {', '.join(sorted(dup))}")
        for v in self.variables:
            for e in v.validate():
                errs.append(f"[{v.label or v.name}] {e}")
        if self.iterations < 2:
            errs.append("numero de iteracoes deve ser >= 2")
        if self.method not in sampling.VALID_METHODS:
            errs.append(f"metodo de amostragem invalido: {self.method}")
        try:
            Formula(self.formula, names)
        except Exception as e:
            errs.append(f"[formula] {e}")
        if self.correlation is not None:
            k = len(self.variables)
            C = np.asarray(self.correlation, dtype=float)
            if C.shape != (k, k):
                errs.append(f"matriz de correlacao deve ser {k}x{k}")
            else:
                errs.extend(f"[correlacao] {e}" for e in corr_mod.check_correlation_matrix(C))
        return errs


@dataclass
class SimulationResult:
    inputs: np.ndarray  # (n, k) amostras das entradas
    output: np.ndarray  # (n,) saida
    names: List[str]
    labels: List[str]
    spec: SimulationSpec
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.output.size)


def run(spec: SimulationSpec, strict: bool = True) -> SimulationResult:
    """Executa uma simulacao completa."""
    errs = spec.validate()
    hard = [e for e in errs if not e.startswith("[correlacao] matriz nao e positiva")]
    if hard and strict:
        raise ValueError("Especificacao invalida:\n- " + "\n- ".join(hard))

    notes: List[str] = []
    rng = np.random.default_rng(spec.seed)
    names = [v.name for v in spec.variables]
    labels = [v.label or v.name for v in spec.variables]
    k, n = len(spec.variables), int(spec.iterations)

    # 1-2. amostragem e transformada inversa
    U = sampling.unit_samples(n, k, method=spec.method, rng=rng)  # type: ignore[arg-type]
    X = np.empty((n, k), dtype=float)
    for j, v in enumerate(spec.variables):
        X[:, j] = v.ppf(U[:, j])

    non_finite = int((~np.isfinite(X)).sum())
    if non_finite:
        notes.append(
            f"{non_finite} valores nao finitos gerados nas entradas (suporte "
            f"ilimitado + quantis extremos). Eles sao propagados e depois "
            f"descartados nas estatisticas."
        )

    # 3. correlacao
    if spec.correlation is not None and k > 1:
        C = np.asarray(spec.correlation, dtype=float)
        if np.linalg.eigvalsh(
            corr_mod.spearman_to_pearson_normal(C) if spec.spearman_adjust else C
        ).min() <= 0:
            notes.append(
                "A matriz de correlacao informada nao e positiva definida: as "
                "correlacoes especificadas sao mutuamente incompativeis. Foi "
                "usada a matriz positiva semidefinida mais proxima (Higham, "
                "2002), portanto as correlacoes efetivas DIFEREM das pedidas. "
                "Confira a tabela de correlacao obtida."
            )
        X = corr_mod.iman_conover(
            X, C, rng=rng, spearman_adjust=spec.spearman_adjust, repair_psd=True
        )

    # 4. saida
    f = Formula(spec.formula, names)
    y = f.evaluate({names[j]: X[:, j] for j in range(k)})

    bad_y = int((~np.isfinite(y)).sum())
    if bad_y:
        notes.append(
            f"{bad_y} de {n} iteracoes ({100*bad_y/n:.2f}%) produziram saida nao "
            f"finita (divisao por zero, log de numero nao positivo, overflow). "
            f"Elas sao excluidas das estatisticas, o que enviesa os resultados "
            f"se a fracao nao for desprezivel."
        )

    unused = [nm for nm in names if nm not in f.used_names]
    if unused:
        notes.append(
            "Variaveis definidas mas nao usadas na formula: "
            + ", ".join(unused)
            + ". Elas nao influenciam a saida e aparecerao com sensibilidade nula."
        )

    notes.append(sampling.effective_iterations_note(spec.method))  # type: ignore[arg-type]

    return SimulationResult(
        inputs=X, output=y, names=names, labels=labels, spec=spec, notes=notes
    )


def run_replicates(
    spec: SimulationSpec, replicates: int = 10
) -> tuple[SimulationResult, List[np.ndarray]]:
    """Roda R replicacoes independentes (seeds distintas).

    Necessario para quantificar honestamente o erro de simulacao sob LHS,
    onde s/sqrt(n) nao e um estimador valido. Devolve a primeira replicacao
    completa (para os graficos) e a lista de saidas de todas elas.
    """
    outs: List[np.ndarray] = []
    first: SimulationResult | None = None
    base = spec.seed if spec.seed is not None else 0
    for r in range(int(replicates)):
        sub = SimulationSpec(
            variables=spec.variables,
            formula=spec.formula,
            iterations=spec.iterations,
            method=spec.method,
            seed=base + r * 7919,  # primo, para afastar os fluxos
            correlation=spec.correlation,
            spearman_adjust=spec.spearman_adjust,
        )
        res = run(sub)
        outs.append(res.output)
        if first is None:
            first = res
    assert first is not None
    return first, outs


def to_dataframe(result: SimulationResult, output_name: str = "saida"):
    """DataFrame com entradas e saida, para exportacao."""
    import pandas as pd

    data: Dict[str, Any] = {
        result.labels[j]: result.inputs[:, j] for j in range(len(result.names))
    }
    data[output_name] = result.output
    return pd.DataFrame(data)
