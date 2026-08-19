"""
Desempenho e escala.

Estes testes nao competem com um benchmark de verdade: maquina de CI e ruidosa e
qualquer limite absoluto de tempo vira falha intermitente. O que eles protegem e
outra coisa, mais util e mais estavel: a FORMA como o custo cresce.

Uma regressao de vetorizacao -- um laco em Python que substitui uma operacao de
array -- nao muda o resultado nem o teste estatistico, mas troca crescimento
linear por quadratico. E o tipo de defeito que so aparece quando o usuario roda
com um milhao de iteracoes e desiste antes de terminar.

Os limites de tempo aqui sao folgados de proposito (fator 4 sobre o esperado).
Servem para pegar catastrofe, nao para medir microssegundos.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mcrisk.engine import SimulationSpec, Variable, run
from mcrisk.sampling import unit_samples
from mcrisk.summary import describe, value_at_risk

PESADO = pytest.mark.slow


def _cronometrar(fn, *a, **kw) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return time.perf_counter() - t0, out


def _modelo(n: int, k: int = 3, **kw) -> SimulationSpec:
    vs = [Variable(f"v{j}", f"V{j}", "normal", {"mu": 10.0, "sigma": 2.0})
          for j in range(k)]
    base = dict(variables=vs, formula=" + ".join(f"v{j}" for j in range(k)),
                iterations=n, method="lhs", seed=1)
    base.update(kw)
    return SimulationSpec(**base)  # type: ignore[arg-type]


def test_custo_cresce_de_forma_aproximadamente_linear_nas_iteracoes():
    """Dobrar N nao pode mais que triplicar o tempo."""
    run(_modelo(20_000))  # aquece import e cache
    t1, _ = _cronometrar(run, _modelo(100_000))
    t2, _ = _cronometrar(run, _modelo(200_000))
    assert t2 < 3.0 * max(t1, 1e-3), (
        f"tempo passou de {t1:.3f}s para {t2:.3f}s ao dobrar N -- "
        f"sinal de laco nao vetorizado"
    )


def test_custo_cresce_de_forma_aproximadamente_linear_nas_variaveis():
    t1, _ = _cronometrar(run, _modelo(50_000, k=2))
    t2, _ = _cronometrar(run, _modelo(50_000, k=8))
    assert t2 < 8.0 * max(t1, 1e-3), f"{t1:.3f}s -> {t2:.3f}s de 2 para 8 variaveis"


def test_lhs_nao_e_desproporcionalmente_mais_caro_que_mc():
    t_mc, _ = _cronometrar(unit_samples, 200_000, 5, "mc", np.random.default_rng(1))
    t_lhs, _ = _cronometrar(unit_samples, 200_000, 5, "lhs", np.random.default_rng(1))
    assert t_lhs < 12.0 * max(t_mc, 1e-4), (
        f"LHS {t_lhs:.3f}s contra MC {t_mc:.3f}s -- estratificacao virou gargalo"
    )


def test_memoria_da_amostra_nao_excede_o_previsto():
    """Uma amostra (n x k) de float64 nao pode custar mais que ~1,3x n*k*8 bytes."""
    n, k = 300_000, 4
    r = run(_modelo(n, k=k))
    esperado = n * k * 8
    assert r.inputs.nbytes <= 1.3 * esperado
    assert r.inputs.dtype == np.float64
    assert r.output.nbytes <= 1.3 * n * 8


def test_estatisticas_nao_copiam_a_amostra_varias_vezes():
    """describe + VaR sobre 1M de pontos tem de ser rapido; ordenacao unica."""
    y = np.random.default_rng(0).normal(size=1_000_000)
    t, _ = _cronometrar(describe, y)
    assert t < 6.0, f"describe levou {t:.2f}s em 1M de pontos"
    t2, _ = _cronometrar(value_at_risk, y, 0.99)
    assert t2 < 3.0, f"VaR levou {t2:.2f}s em 1M de pontos"


@PESADO
def test_um_milhao_de_iteracoes_termina_e_da_resultado_correto():
    """Escala de uso real. Marcado como lento: rode com -m slow."""
    t, r = _cronometrar(run, _modelo(1_000_000, k=3))
    assert r.output.size == 1_000_000
    assert np.isfinite(r.output).all()
    # media = 30, dp = sqrt(3)*2
    assert r.output.mean() == pytest.approx(30.0, abs=0.05)
    assert r.output.std(ddof=1) == pytest.approx(np.sqrt(3) * 2.0, rel=0.02)
    assert t < 120.0, f"1M de iteracoes levou {t:.1f}s"


@PESADO
def test_iman_conover_em_escala_grande():
    n, k = 500_000, 6
    C = np.full((k, k), 0.35)
    np.fill_diagonal(C, 1.0)
    t, r = _cronometrar(run, _modelo(n, k=k, correlation=C))
    assert r.output.size == n
    assert t < 180.0, f"Iman-Conover com {n}x{k} levou {t:.1f}s"


def test_amostragem_de_um_milhao_de_pontos_em_tempo_razoavel():
    t, u = _cronometrar(unit_samples, 1_000_000, 3, "lhs", np.random.default_rng(2))
    assert u.shape == (1_000_000, 3)
    assert t < 30.0, f"unit_samples levou {t:.1f}s"
