"""
Copulas (Gaussiana e t) e analise de cenarios.

O teste que importa aqui e o do CONTRASTE. Copula Gaussiana e copula t com a
mesma correlacao produzem medias parecidas, desvios parecidos e ate VaR a 95%
parecido -- e e exatamente por isso que a diferenca passa despercebida em
revisao. Ela aparece onde interessa: na frequencia de extremos SIMULTANEOS.

Se algum dia esses testes pararem de distinguir as duas, a copula t virou
enfeite e o motor voltou a subestimar risco de cauda sem avisar.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mcrisk import copula, scenarios
from mcrisk.correlation import achieved_spearman
from mcrisk.engine import SimulationSpec, Variable, run

RHO = 0.5
P2 = np.array([[1.0, RHO], [RHO, 1.0]])


# ---------------------------------------------------------------------------
# Propriedades das copulas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["gaussian", "t"])
def test_marginais_da_copula_sao_uniformes(kind):
    """Copula e, por definicao, distribuicao com marginais U(0,1)."""
    rng = np.random.default_rng(1)
    u = (copula.gaussian_copula_u(P2, 50_000, rng)[0] if kind == "gaussian"
         else copula.t_copula_u(P2, 5.0, 50_000, rng)[0])
    for j in range(2):
        p = stats.kstest(u[:, j], "uniform").pvalue
        assert p > 0.001, f"marginal {j} da copula {kind} nao e uniforme (p={p:.4g})"


@pytest.mark.parametrize("kind", ["gaussian", "t"])
def test_copula_atinge_a_correlacao_pedida(kind):
    rng = np.random.default_rng(2)
    u, _ = copula.copula_u(kind, P2, 100_000, rng, df=8.0, spearman_adjust=True)
    obtido = achieved_spearman(u)[0, 1]
    # a conversao e exata na Gaussiana e aproximada na t
    tol = 0.02 if kind == "gaussian" else 0.05
    assert obtido == pytest.approx(RHO, abs=tol), f"{kind}: obtido {obtido:.4f}"


def test_dependencia_de_cauda_distingue_t_de_gaussiana():
    """O ponto inteiro do modulo, em um teste.

    Mesma correlacao, mesma marginal, mesmo numero de iteracoes. A copula t tem
    de produzir sensivelmente mais pares extremos simultaneos.
    """
    rng = np.random.default_rng(3)
    n = 300_000
    ug, _ = copula.gaussian_copula_u(P2, n, rng)
    ut, _ = copula.t_copula_u(P2, 3.0, n, rng)
    lg = copula.empirical_tail_dependence(ug[:, 0], ug[:, 1], q=0.99)
    lt = copula.empirical_tail_dependence(ut[:, 0], ut[:, 1], q=0.99)
    assert lt > 2.0 * lg, f"t={lt:.4f} nao supera folgadamente gaussiana={lg:.4f}"


def test_dependencia_de_cauda_empirica_se_aproxima_da_teorica():
    rng = np.random.default_rng(4)
    df = 4.0
    teorico = copula.tail_dependence_t(RHO, df)
    ut, _ = copula.t_copula_u(P2, df, 400_000, rng)
    emp = copula.empirical_tail_dependence(ut[:, 0], ut[:, 1], q=0.995)
    assert emp == pytest.approx(teorico, abs=0.06), f"emp={emp:.4f} teorico={teorico:.4f}"


def test_t_converge_para_gaussiana_quando_gl_cresce():
    """Limite assintotico: lambda -> 0 conforme v -> infinito."""
    valores = [copula.tail_dependence_t(RHO, v) for v in (3, 10, 30, 100, 1000)]
    assert all(a > b for a, b in zip(valores, valores[1:])), valores
    assert valores[-1] < 1e-3
    assert copula.tail_dependence_gaussian(RHO) == 0.0


def test_dependencia_de_cauda_positiva_mesmo_com_correlacao_zero():
    """Resultado contraintuitivo e central: rho=0 nao implica independencia na t."""
    assert copula.tail_dependence_t(0.0, 3.0) > 0.05
    assert copula.tail_dependence_gaussian(0.0) == 0.0


def test_copula_recusa_graus_de_liberdade_invalidos():
    rng = np.random.default_rng(5)
    for df in (0.0, 1.0, -3.0, float("nan")):
        with pytest.raises(ValueError):
            copula.t_copula_u(P2, df, 100, rng)


def test_copula_repara_matriz_nao_psd_e_avisa():
    C = np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]])
    rng = np.random.default_rng(6)
    u, reparo = copula.gaussian_copula_u(C, 5_000, rng)
    assert reparo is True
    assert u.shape == (5_000, 3)
    assert np.isfinite(u).all()


def test_copula_invalida_e_recusada():
    with pytest.raises(ValueError, match="copula invalida"):
        copula.copula_u("clayton", P2, 100, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Integracao com o motor
# ---------------------------------------------------------------------------

def _duas_normais(dep: str, **kw) -> SimulationSpec:
    vs = [Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
          Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0})]
    base = dict(variables=vs, formula="a + b", iterations=200_000, seed=11,
                correlation=P2, dependence=dep)
    base.update(kw)
    return SimulationSpec(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("dep", ["iman_conover", "gaussian", "t"])
def test_motor_preserva_as_marginais_em_todos_os_esquemas(dep):
    r = run(_duas_normais(dep, copula_df=6.0))
    for j in range(2):
        p = stats.kstest(r.inputs[:, j], "norm").pvalue
        assert p > 0.001, f"{dep}: marginal {j} deixou de ser normal (p={p:.4g})"


@pytest.mark.parametrize("dep", ["gaussian", "t"])
def test_motor_atinge_a_correlacao_pedida_com_copula(dep):
    r = run(_duas_normais(dep, copula_df=8.0))
    obtido = achieved_spearman(r.inputs)[0, 1]
    assert obtido == pytest.approx(RHO, abs=0.05), f"{dep}: {obtido:.4f}"


def test_copula_t_engorda_a_cauda_da_soma():
    """Onde a escolha da copula muda a decisao: o percentil extremo da saida."""
    g = run(_duas_normais("gaussian"))
    t = run(_duas_normais("t", copula_df=3.0))
    q_g = float(np.quantile(g.output, 0.999))
    q_t = float(np.quantile(t.output, 0.999))
    assert q_t > q_g, f"P99,9 sob t ({q_t:.3f}) deveria superar o gaussiano ({q_g:.3f})"


def test_motor_avisa_que_lhs_nao_vale_sob_copula():
    r = run(_duas_normais("gaussian", method="lhs"))
    assert any("LHS" in n for n in r.notes), (
        "usuario que escolheu LHS precisa saber que a estratificacao nao se aplica"
    )


def test_copula_t_avisa_sobre_a_conversao_aproximada():
    r = run(_duas_normais("t", copula_df=3.0, spearman_adjust=True))
    assert any("exata apenas" in n for n in r.notes)


def test_copula_sem_matriz_de_correlacao_e_recusada():
    vs = [Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0})]
    with pytest.raises(ValueError, match="copula exige"):
        run(SimulationSpec(variables=vs, formula="a", iterations=100,
                           dependence="gaussian"))


def test_esquema_de_dependencia_invalido_e_recusado():
    with pytest.raises(ValueError, match="dependencia invalido"):
        run(_duas_normais("clayton"))


def test_gl_invalido_na_spec_e_recusado():
    with pytest.raises(ValueError, match="graus de liberdade"):
        run(_duas_normais("t", copula_df=1.0))


def test_copula_e_reprodutivel():
    a, b = run(_duas_normais("t", iterations=5_000)), run(_duas_normais("t", iterations=5_000))
    assert np.array_equal(a.output, b.output)


def test_iman_conover_continua_sendo_o_padrao():
    """Mudanca de padrao silenciosa alteraria todos os resultados ja publicados."""
    spec = SimulationSpec(
        variables=[Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0})],
        formula="a", iterations=10,
    )
    assert spec.dependence == "iman_conover"


# ---------------------------------------------------------------------------
# Cenarios
# ---------------------------------------------------------------------------

def _modelo_custo() -> SimulationSpec:
    return SimulationSpec(
        variables=[Variable("c", "Custo", "normal", {"mu": 100.0, "sigma": 20.0}),
                   Variable("q", "Qtd", "triangular",
                            {"minimo": 8.0, "moda": 10.0, "maximo": 15.0})],
        formula="c * q", iterations=60_000, seed=7,
    )


def test_condicional_filtra_sem_re_simular():
    spec = _modelo_custo()
    r = run(spec)
    cen = scenarios.conditional(r, lambda v: v["c"] > 120.0, "custo alto")
    assert 0 < cen.n < r.n
    assert cen.fracao == pytest.approx(cen.n / r.n)
    # E[c | c > 120] > E[c], logo a saida condicional tem media maior
    assert cen.resumo["media"] > float(np.mean(r.output[np.isfinite(r.output)]))


def test_condicional_avisa_quando_a_amostra_e_pequena():
    r = run(_modelo_custo())
    cen = scenarios.conditional(r, lambda v: v["c"] > 175.0, "cauda")
    assert cen.n < scenarios.MIN_ITERACOES_CONDICIONAL
    assert any("poucos pontos" in a or "iteracoes no cenario" in a for a in cen.avisos)


def test_condicional_sem_nenhuma_iteracao_nao_inventa_estatistica():
    r = run(_modelo_custo())
    cen = scenarios.conditional(r, lambda v: v["c"] > 1e6, "impossivel")
    assert cen.n == 0
    assert cen.resumo == {}
    assert cen.avisos


def test_condicional_recusa_mascara_de_forma_errada():
    r = run(_modelo_custo())
    with pytest.raises(ValueError, match="mascara"):
        scenarios.conditional(r, lambda v: np.array([True, False]), "ruim")


def test_estresse_desloca_a_media_no_valor_previsto():
    """E[c*q] = E[c]*E[q] com independencia. Deslocar mu de c em 30 desloca em 30*E[q]."""
    spec = _modelo_custo()
    base = run(spec)
    st = scenarios.stress(spec, {"c": {"mu": 130.0}}, "custo +30", base=base)
    e_q = (8.0 + 10.0 + 15.0) / 3.0
    assert st.delta["media"] == pytest.approx(30.0 * e_q, rel=0.02)
    assert st.delta_pct["media"] == pytest.approx(30.0, rel=0.05)


def test_estresse_nao_modifica_a_especificacao_original():
    spec = _modelo_custo()
    antes = dict(spec.variables[0].params)
    scenarios.stress(spec, {"c": {"mu": 999.0}}, "extremo")
    assert spec.variables[0].params == antes, (
        "cenario que muda o objeto original contamina a base de comparacao"
    )


def test_estresse_permite_trocar_a_familia_da_distribuicao():
    spec = _modelo_custo()
    st = scenarios.stress(
        spec, {"c": {"dist_key": "lognormal_real", "media": 100.0, "desvio": 40.0}},
        "custo assimetrico")
    assert np.isfinite(st.resumo_estressado["media"])
    assert st.resumo_estressado["media"] == pytest.approx(st.resumo_base["media"], rel=0.05)


def test_estresse_recusa_variavel_inexistente():
    with pytest.raises(KeyError, match="inexistentes"):
        scenarios.stress(_modelo_custo(), {"nao_existe": {"mu": 1.0}})


def test_estresse_carrega_a_ressalva_de_leitura():
    st = scenarios.stress(_modelo_custo(), {"c": {"mu": 110.0}})
    assert any("probabilidade" in a for a in st.avisos), (
        "numero de estresse lido como percentil da distribuicao base e erro comum"
    )


def test_tabela_de_estresse_reune_varios_cenarios():
    spec = _modelo_custo()
    base = run(spec)
    cenarios_ = [scenarios.stress(spec, {"c": {"mu": m}}, f"mu={m}", base=base)
                 for m in (110.0, 130.0, 150.0)]
    df = scenarios.tabela_estresse(cenarios_, "media")
    assert list(df["cenario"]) == ["mu=110.0", "mu=130.0", "mu=150.0"]
    assert df["delta"].is_monotonic_increasing
    assert (df["base"].nunique() == 1), "a base tem de ser a mesma nos tres"


def test_estresse_preserva_o_esquema_de_dependencia():
    spec = _duas_normais("t", iterations=5_000, copula_df=4.0)
    novo = scenarios.aplicar_overrides(spec, {"a": {"mu": 1.0}})
    assert novo.dependence == "t" and novo.copula_df == 4.0
    assert novo.correlation is spec.correlation
