"""
Verificacao do orquestrador, das estatisticas de saida e da sensibilidade.

Onde possivel, os testes usam modelos com solucao ANALITICA conhecida, de
forma que o valor esperado nao vem da propria simulacao.
"""

import numpy as np
import pytest

from mcrisk import sensitivity, summary
from mcrisk.engine import SimulationSpec, Variable, run, run_replicates, to_dataframe


def _spec_soma_de_normais(n=50_000, method="lhs", seed=1, corr=None):
    """X ~ N(10,2), Y ~ N(5,1), saida X+Y.

    Solucao analitica: X+Y ~ N(15, sqrt(4+1)) quando independentes.
    """
    return SimulationSpec(
        variables=[
            Variable("x", "X", "normal", {"mu": 10.0, "sigma": 2.0}),
            Variable("y", "Y", "normal", {"mu": 5.0, "sigma": 1.0}),
        ],
        formula="x + y",
        iterations=n,
        method=method,
        seed=seed,
        correlation=corr,
    )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def test_soma_de_normais_bate_com_a_solucao_analitica():
    r = run(_spec_soma_de_normais())
    d = summary.describe(r.output)
    assert abs(d["media"] - 15.0) < 0.05
    assert abs(d["desvio"] - np.sqrt(5.0)) < 0.05


def test_correlacao_muda_a_variancia_na_direcao_prevista():
    """Var(X+Y) = Var X + Var Y + 2 Cov. Com correlacao positiva, deve subir."""
    base = summary.describe(run(_spec_soma_de_normais()).output)["desvio"]
    C = np.array([[1.0, 0.8], [0.8, 1.0]])
    corr_pos = summary.describe(run(_spec_soma_de_normais(corr=C)).output)["desvio"]
    Cn = np.array([[1.0, -0.8], [-0.8, 1.0]])
    corr_neg = summary.describe(run(_spec_soma_de_normais(corr=Cn)).output)["desvio"]
    assert corr_pos > base > corr_neg
    # valor analitico aproximado para rho de Pearson ~ 0.8
    assert abs(corr_pos - np.sqrt(4 + 1 + 2 * 0.8 * 2 * 1)) < 0.1


def test_reprodutibilidade_bit_a_bit():
    a = run(_spec_soma_de_normais(n=5000, seed=99)).output
    b = run(_spec_soma_de_normais(n=5000, seed=99)).output
    assert np.array_equal(a, b)


def test_seeds_diferentes_dao_resultados_diferentes():
    a = run(_spec_soma_de_normais(n=5000, seed=1)).output
    b = run(_spec_soma_de_normais(n=5000, seed=2)).output
    assert not np.array_equal(a, b)


def test_especificacao_invalida_e_recusada():
    spec = SimulationSpec(
        variables=[Variable("x", "X", "normal", {"mu": 0.0, "sigma": -1.0})],
        formula="x",
    )
    assert spec.validate()
    with pytest.raises(ValueError):
        run(spec)


def test_nomes_duplicados_sao_detectados():
    spec = SimulationSpec(
        variables=[
            Variable("x", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("x", "B", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="x",
    )
    assert any("duplicados" in e for e in spec.validate())


def test_variavel_nao_usada_gera_aviso():
    spec = SimulationSpec(
        variables=[
            Variable("x", "X", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("z", "Z", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="x",
        iterations=1000,
    )
    r = run(spec)
    assert any("nao usadas" in n for n in r.notes)


def test_saida_nao_finita_gera_aviso():
    spec = SimulationSpec(
        variables=[Variable("x", "X", "normal", {"mu": 0.0, "sigma": 1.0})],
        formula="log(x)",  # metade das iteracoes tem x < 0
        iterations=2000,
        seed=3,
    )
    r = run(spec)
    assert any("nao finita" in n for n in r.notes)
    d = summary.describe(r.output)
    assert d["n_nao_finitos"] > 0


def test_dataframe_de_exportacao_tem_as_colunas_certas():
    r = run(_spec_soma_de_normais(n=100))
    df = to_dataframe(r, output_name="resultado")
    assert list(df.columns) == ["X", "Y", "resultado"]
    assert len(df) == 100


def test_replicacoes_sao_independentes_entre_si():
    spec = _spec_soma_de_normais(n=2000)
    _, outs = run_replicates(spec, replicates=5)
    assert len(outs) == 5
    for i in range(1, 5):
        assert not np.array_equal(outs[0], outs[i])


# --------------------------------------------------------------------------
# Estatisticas
# --------------------------------------------------------------------------


def test_percentis_de_uma_normal_batem_com_a_teoria():
    from scipy import stats as st

    r = run(_spec_soma_de_normais(n=100_000))
    d = summary.describe(r.output)
    for p in (5, 50, 95):
        teorico = st.norm.ppf(p / 100, loc=15.0, scale=np.sqrt(5.0))
        assert abs(d[f"P{p}"] - teorico) < 0.06, f"P{p}"


def test_ic_da_media_cobre_o_valor_verdadeiro_na_frequencia_certa():
    """Teste de cobertura: ~95% dos ICs de 95% devem conter a media real."""
    cobertura = 0
    reps = 200
    for r in range(reps):
        out = run(_spec_soma_de_normais(n=500, method="mc", seed=1000 + r)).output
        lo, hi = summary.mean_ci(out, level=0.95)
        cobertura += int(lo <= 15.0 <= hi)
    taxa = cobertura / reps
    assert 0.90 < taxa < 0.99, f"cobertura observada = {taxa:.3f}"


def test_ic_de_quantil_cobre_o_valor_verdadeiro():
    from scipy import stats as st

    verdadeiro = st.norm.ppf(0.95, loc=15.0, scale=np.sqrt(5.0))
    cobertura = 0
    reps = 200
    for r in range(reps):
        out = run(_spec_soma_de_normais(n=1000, method="mc", seed=2000 + r)).output
        lo, hi = summary.quantile_ci(out, 0.95, level=0.95)
        cobertura += int(lo <= verdadeiro <= hi)
    assert cobertura / reps > 0.88


def test_cvar_e_mais_extremo_que_var():
    r = run(_spec_soma_de_normais(n=50_000))
    var = summary.value_at_risk(r.output, 0.05)
    cvar = summary.conditional_value_at_risk(r.output, 0.05)
    assert cvar < var


def test_var_respeita_a_orientacao_de_perda():
    r = run(_spec_soma_de_normais(n=20_000))
    baixo = summary.value_at_risk(r.output, 0.05, loss_is_low=True)
    alto = summary.value_at_risk(r.output, 0.05, loss_is_low=False)
    assert baixo < alto


def test_erro_padrao_diminui_com_a_raiz_de_n():
    a = summary.mc_standard_error(
        run(_spec_soma_de_normais(n=2_000, method="mc", seed=5)).output
    )
    b = summary.mc_standard_error(
        run(_spec_soma_de_normais(n=8_000, method="mc", seed=5)).output
    )
    assert abs(a / b - 2.0) < 0.35  # 4x iteracoes -> ~2x menos erro


def test_lhs_tem_menos_erro_real_que_mc_com_mesmo_n():
    """Confirma o ganho do LHS no nivel da simulacao completa."""
    erros = {"mc": [], "lhs": []}
    for m in ("mc", "lhs"):
        for r in range(60):
            out = run(_spec_soma_de_normais(n=400, method=m, seed=3000 + r)).output
            erros[m].append(abs(np.mean(out) - 15.0))
    assert np.mean(erros["lhs"]) < np.mean(erros["mc"])


def test_replicate_summary_produz_ic_valido():
    spec = _spec_soma_de_normais(n=2000)
    _, outs = run_replicates(spec, replicates=15)
    res = summary.replicate_summary([float(np.mean(o)) for o in outs])
    assert res["replicacoes"] == 15
    assert res["ic_inf"] < 15.0 < res["ic_sup"]
    assert res["erro_padrao"] > 0


def test_probabilidades_sao_complementares():
    r = run(_spec_soma_de_normais(n=10_000))
    assert np.isclose(
        summary.prob_below(r.output, 15.0) + summary.prob_above(r.output, 15.0), 1.0
    )


def test_iterations_for_precision_escala_com_o_quadrado():
    r = run(_spec_soma_de_normais(n=10_000))
    n1 = summary.iterations_for_precision(r.output, 0.10)
    n2 = summary.iterations_for_precision(r.output, 0.05)
    assert abs(n2 / n1 - 4.0) < 0.01


def test_convergence_path_termina_na_media_final():
    r = run(_spec_soma_de_normais(n=5000))
    idx, med = summary.convergence_path(r.output)
    assert idx[-1] == 5000
    assert abs(med[-1] - np.mean(r.output)) < 1e-9


# --------------------------------------------------------------------------
# Sensibilidade
# --------------------------------------------------------------------------


def test_sensibilidade_identifica_a_variavel_dominante():
    """Modelo linear com contribuicoes de variancia conhecidas."""
    spec = SimulationSpec(
        variables=[
            Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("c", "C", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="10*a + 5*b + 1*c",
        iterations=20_000,
        seed=8,
    )
    r = run(spec)
    s = sensitivity.analyze(r.inputs, r.output, r.labels)
    ordem = sensitivity.tornado_order(s)
    assert [s.names[i] for i in ordem] == ["A", "B", "C"]
    # contribuicoes teoricas: 100/126, 25/126, 1/126
    assert abs(s.contribution[0] - 100 / 126) < 0.02
    assert abs(s.contribution[1] - 25 / 126) < 0.02
    assert s.rank_r2 > 0.95


def test_sinal_da_sensibilidade_e_correto():
    spec = SimulationSpec(
        variables=[
            Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="a - 2*b",
        iterations=10_000,
        seed=9,
    )
    r = run(spec)
    s = sensitivity.analyze(r.inputs, r.output, r.labels)
    assert s.spearman[0] > 0 and s.spearman[1] < 0
    assert s.srrc[0] > 0 and s.srrc[1] < 0


def test_modelo_nao_monotono_dispara_aviso_de_r2_baixo():
    """O caso em que o grafico tornado engana - e o codigo avisa."""
    spec = SimulationSpec(
        variables=[
            Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="a * b",  # simetrica: sensibilidade monotona ~ 0
        iterations=20_000,
        seed=10,
    )
    r = run(spec)
    s = sensitivity.analyze(r.inputs, r.output, r.labels)
    assert s.rank_r2 < 0.5
    assert any("Sobol" in w for w in s.warnings)


def test_variavel_constante_nao_quebra_a_analise():
    spec = SimulationSpec(
        variables=[
            Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("k", "K", "discrete_custom", values=[5.0], probs=[1.0]),
        ],
        formula="a + k",
        iterations=5_000,
        seed=11,
    )
    r = run(spec)
    s = sensitivity.analyze(r.inputs, r.output, r.labels)
    assert s.spearman[1] == 0.0
    assert any("constantes" in w for w in s.warnings)


def test_entradas_colineares_geram_aviso():
    C = np.array([[1.0, 0.995], [0.995, 1.0]])
    spec = SimulationSpec(
        variables=[
            Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="a + b",
        iterations=10_000,
        seed=12,
        correlation=C,
    )
    r = run(spec)
    s = sensitivity.analyze(r.inputs, r.output, r.labels)
    assert any("VIF" in w for w in s.warnings)


def test_entradas_pouco_correlacionadas_nao_geram_aviso_de_vif():
    """Guarda contra aviso disparando a esmo em modelos normais."""
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    spec = SimulationSpec(
        variables=[
            Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
            Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0}),
        ],
        formula="a + b",
        iterations=10_000,
        seed=13,
        correlation=C,
    )
    r = run(spec)
    s = sensitivity.analyze(r.inputs, r.output, r.labels)
    assert not any("VIF" in w for w in s.warnings)
