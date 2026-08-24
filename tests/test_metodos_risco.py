"""
Os metodos de risco acrescentados na paridade com @RISK e ModelRisk.

Onde existe solucao fechada, o valor esperado NAO vem da simulacao. Onde nao
existe, o teste verifica uma PROPRIEDADE que a implementacao correta tem e a
errada nao - preferencialmente uma que distinga este metodo dos outros, para
que o teste falhe se alguem trocar um pelo outro.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mcrisk import convergence as conv
from mcrisk import copula, fitting, scenarios, sensitivity, summary
from mcrisk.engine import SimulationSpec, Variable, run


def _tres_normais(formula: str, n: int = 40_000, seed: int = 1, **kw):
    vs = [
        Variable(c, c.upper(), "normal", {"mu": 0.0, "sigma": 1.0})
        for c in ("a", "b", "c")
    ]
    return run(SimulationSpec(variables=vs, formula=formula, iterations=n,
                              seed=seed, **kw))


# ---------------------------------------------------------------------------
# Contribuicao para a variancia: ancora analitica exata
# ---------------------------------------------------------------------------


def test_contribuicao_para_variancia_recupera_as_fracoes_teoricas():
    """y = 3a + b com a, b ~ N(0,1) independentes: 90% e 10% da variancia."""
    rng = np.random.default_rng(1)
    n = 60_000
    X = rng.normal(size=(n, 3))
    y = 3 * X[:, 0] + X[:, 1]
    cv = sensitivity.contribution_to_variance(X, y, ["a", "b", "c"])
    assert cv.fracao[0] == pytest.approx(0.9, abs=0.01)
    assert cv.fracao[1] == pytest.approx(0.1, abs=0.01)
    assert cv.fracao[2] == pytest.approx(0.0, abs=0.005)
    assert cv.r2_total == pytest.approx(1.0, abs=1e-6)
    assert cv.nao_explicada == pytest.approx(0.0, abs=1e-6)


def test_as_fracoes_somam_exatamente_o_r2_total():
    """Invariante do metodo sequencial: os incrementos somam o R2 do modelo."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20_000, 4))
    y = X[:, 0] + 0.5 * X[:, 1] - 2 * X[:, 2] + rng.normal(scale=3, size=20_000)
    cv = sensitivity.contribution_to_variance(X, y, list("abcd"))
    assert cv.fracao.sum() == pytest.approx(cv.r2_total, abs=1e-9)
    assert 0.0 <= cv.r2_total <= 1.0


def test_variancia_nao_explicada_aparece_em_modelo_nao_linear():
    """Com y = a*b nenhuma regressao linear explica nada, e o teste exige que
    a implementacao ADMITA isso em vez de distribuir 100% entre as entradas."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20_000, 2))
    y = X[:, 0] * X[:, 1]
    cv = sensitivity.contribution_to_variance(X, y, ["a", "b"])
    assert cv.nao_explicada > 0.9
    assert any("NAO e explicada" in w for w in cv.warnings)


def test_ordem_de_entrada_e_registrada():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10_000, 3))
    y = 5 * X[:, 2] + X[:, 0]
    cv = sensitivity.contribution_to_variance(X, y, list("abc"))
    assert cv.ordem_entrada[0] == 2, "a mais explicativa entra primeiro"
    recs = {r["variavel"]: r["passo"] for r in cv.as_records()}
    assert recs["c"] == 1


# ---------------------------------------------------------------------------
# Change in Output Statistic: o metodo que enxerga o que os outros nao veem
# ---------------------------------------------------------------------------


def test_metodo_condicional_enxerga_relacao_em_u_que_os_de_posto_perdem():
    """O ponto inteiro de ter um quarto metodo.

    Com y = a^2 a correlacao de posto e o SRRC ficam proximos de zero, porque
    a relacao nao e monotona - e `a` domina o modelo. Se este teste passasse
    para os dois grupos, o metodo condicional nao estaria acrescentando nada.
    """
    rng = np.random.default_rng(5)
    n = 40_000
    X = rng.normal(size=(n, 3))
    y = X[:, 0] ** 2
    nomes = list("abc")

    posto = sensitivity.analyze(X, y, nomes)
    assert abs(posto.spearman[0]) < 0.05, "de posto e cego para relacao em U"
    assert abs(posto.srrc[0]) < 0.05

    cond = sensitivity.change_in_output_statistic(X, y, nomes, bins=10)
    assert cond.swing[0] > 10 * max(cond.swing[1], cond.swing[2]), (
        "o metodo condicional precisa apontar `a` como dominante"
    )
    assert cond.ordem()[0] == 0


def test_swing_da_media_bate_com_o_calculo_direto_por_faixas():
    """Ancoragem no proprio criterio: as faixas sao equiprovaveis por CONTAGEM."""
    rng = np.random.default_rng(6)
    n, bins = 10_000, 5
    X = rng.normal(size=(n, 1))
    y = 2.0 * X[:, 0]
    cond = sensitivity.change_in_output_statistic(X, y, ["a"], bins=bins)
    ordem = np.argsort(X[:, 0], kind="stable")
    esperado = [float(np.mean(y[g])) for g in np.array_split(ordem, bins)]
    assert cond.valores[0] == pytest.approx(esperado, abs=1e-9)
    assert cond.swing[0] == pytest.approx(max(esperado) - min(esperado), abs=1e-9)
    assert cond.n_por_faixa.sum() == n


def test_faixas_sao_equiprovaveis_e_nao_de_largura_igual():
    """Numa entrada muito assimetrica, faixas por largura esvaziariam a cauda."""
    rng = np.random.default_rng(7)
    X = rng.lognormal(0.0, 1.5, size=(6_000, 1))
    cond = sensitivity.change_in_output_statistic(X, X[:, 0], ["a"], bins=10)
    assert cond.n_por_faixa.max() - cond.n_por_faixa.min() <= 1


def test_faixa_pequena_demais_gera_aviso_em_vez_de_grafico_bonito():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(120, 2))
    cond = sensitivity.change_in_output_statistic(
        X, X[:, 0], ["a", "b"], bins=20, min_por_faixa=30
    )
    assert any("instavel" in w for w in cond.warnings)


@pytest.mark.parametrize("stat", list(sensitivity.STATS_CONDICIONAIS))
def test_toda_estatistica_condicional_declarada_funciona(stat):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(4_000, 2))
    cond = sensitivity.change_in_output_statistic(
        X, X[:, 0] + X[:, 1], ["a", "b"], stat=stat, bins=5
    )
    assert np.isfinite(cond.valores).all()
    assert np.isfinite(cond.base)


def test_estatistica_condicional_invalida_e_recusada():
    X = np.random.default_rng(0).normal(size=(500, 1))
    with pytest.raises(ValueError, match="estatistica invalida"):
        sensitivity.change_in_output_statistic(X, X[:, 0], ["a"], stat="mediana_movel")


def test_bins_demais_para_a_amostra_e_recusado():
    X = np.random.default_rng(0).normal(size=(10, 1))
    with pytest.raises(ValueError, match="nao dao para"):
        sensitivity.change_in_output_statistic(X, X[:, 0], ["a"], bins=10)


# ---------------------------------------------------------------------------
# Significancia de cenario
# ---------------------------------------------------------------------------


def test_significancia_ordena_pelas_entradas_que_levam_ao_cenario():
    r = _tres_normais("5*a + b + 0*c", n=40_000, seed=11)
    sig = scenarios.scenario_significance(r, percentil=90.0, cauda="superior")
    assert sig.ordem()[0] == 0
    assert abs(sig.significancia[0]) >= sig.limiar
    assert abs(sig.significancia[2]) < 0.05, "peso zero nao pode ser significativo"
    assert abs(sig.significancia[0]) > abs(sig.significancia[1]) > abs(
        sig.significancia[2]
    )
    assert sig.n == pytest.approx(0.10 * r.n, rel=0.02)


def test_significancia_troca_de_sinal_com_a_cauda():
    """Cenario de saida ALTA puxa a mediana de `a` para cima; o de saida BAIXA,
    para baixo. Sinal errado aqui inverteria a leitura de qualquer decisao."""
    r = _tres_normais("5*a + b", n=30_000, seed=12)
    alto = scenarios.scenario_significance(r, percentil=90.0, cauda="superior")
    baixo = scenarios.scenario_significance(r, percentil=10.0, cauda="inferior")
    assert alto.significancia[0] > 0
    assert baixo.significancia[0] < 0
    assert alto.significancia[0] == pytest.approx(-baixo.significancia[0], rel=0.15)


def test_entrada_irrelevante_fica_abaixo_do_limiar():
    r = _tres_normais("a + b + 0*c", n=30_000, seed=13)
    sig = scenarios.scenario_significance(r, percentil=95.0)
    significativas = {sig.labels[i] for i in sig.significativas()}
    assert "C" not in significativas


def test_denominador_e_o_desvio_da_amostra_inteira():
    """Usar o desvio do SUBCONJUNTO tornaria a medida circular: condicionar
    estreita a entrada influente, o denominador encolheria justamente para
    ela, e a significancia dela seria inflada."""
    r = _tres_normais("5*a + b", n=20_000, seed=14)
    sig = scenarios.scenario_significance(r, percentil=90.0)
    assert sig.desvio == pytest.approx(r.inputs.std(axis=0, ddof=1), rel=1e-9)


def test_alvo_por_funcao_e_equivalente_ao_alvo_por_percentil():
    r = _tres_normais("a + b", n=20_000, seed=15)
    corte = float(np.percentile(r.output, 90))
    por_pct = scenarios.scenario_significance(r, percentil=90.0)
    por_fn = scenarios.scenario_significance(r, alvo=lambda y: y > corte)
    assert por_fn.n == por_pct.n
    assert por_fn.significancia == pytest.approx(por_pct.significancia, abs=1e-12)


def test_cenario_vazio_avisa_em_vez_de_quebrar():
    r = _tres_normais("a + b", n=5_000, seed=16)
    sig = scenarios.scenario_significance(r, alvo=lambda y: y > 1e9)
    assert sig.n == 0
    assert any("Nenhuma iteracao" in a for a in sig.avisos)
    assert np.isnan(sig.significancia).all()


def test_cenario_estreito_avisa_sobre_instabilidade():
    r = _tres_normais("a + b", n=5_000, seed=17)
    sig = scenarios.scenario_significance(r, percentil=99.0, n_minimo=100)
    assert any("instavel" in a for a in sig.avisos)


def test_alvo_e_percentil_juntos_sao_recusados():
    r = _tres_normais("a", n=2_000, seed=18)
    with pytest.raises(ValueError, match="exatamente um"):
        scenarios.scenario_significance(r, alvo=lambda y: y > 0, percentil=90.0)
    with pytest.raises(ValueError, match="exatamente um"):
        scenarios.scenario_significance(r)


# ---------------------------------------------------------------------------
# Convergencia
# ---------------------------------------------------------------------------


def test_projecao_de_iteracoes_bate_com_a_formula_fechada():
    """n = (z*s/(tol*|media|))^2, com z = 1,959964 para 95%."""
    rng = np.random.default_rng(21)
    y = rng.normal(100.0, 20.0, 200_000)
    obtido = conv.iteracoes_para_tolerancia(y, tolerancia=0.01, confianca=0.95)
    z = float(stats.norm.ppf(0.975))
    esperado = (z * y.std(ddof=1) / (0.01 * abs(y.mean()))) ** 2
    assert obtido == pytest.approx(np.ceil(esperado), rel=1e-9)


def test_meia_largura_da_media_e_o_erro_padrao_vezes_z():
    rng = np.random.default_rng(22)
    y = rng.normal(50.0, 5.0, 20_000)
    est = conv.check(y, "media", tolerancia=0.01, confianca=0.95)
    z = float(stats.norm.ppf(0.975))
    assert est.meia_largura == pytest.approx(
        z * y.std(ddof=1) / np.sqrt(y.size), rel=1e-12
    )
    assert est.valor == pytest.approx(float(y.mean()), rel=1e-12)


def test_tolerancia_mais_apertada_exige_mais_iteracoes():
    rng = np.random.default_rng(23)
    y = rng.normal(100.0, 20.0, 60_000)
    frouxa = conv.monitor(y, ("media",), tolerancia=0.05, passo=500)
    apertada = conv.monitor(y, ("media",), tolerancia=0.01, passo=500)
    assert frouxa.convergiu_em["media"] is not None
    assert apertada.convergiu_em["media"] is not None
    assert apertada.convergiu_em["media"] > frouxa.convergiu_em["media"]


def test_confianca_maior_exige_mais_iteracoes():
    rng = np.random.default_rng(24)
    y = rng.normal(100.0, 20.0, 60_000)
    a = conv.monitor(y, ("media",), tolerancia=0.01, confianca=0.80, passo=500)
    b = conv.monitor(y, ("media",), tolerancia=0.01, confianca=0.99, passo=500)
    assert b.convergiu_em["media"] > a.convergiu_em["media"]


def test_percentil_extremo_converge_mais_devagar_que_a_media():
    """Percentis de cauda dependem de poucas observacoes; qualquer implementacao
    que os declare convergidos junto com a media esta usando o IC errado."""
    rng = np.random.default_rng(25)
    y = rng.normal(100.0, 20.0, 200_000)
    r = conv.monitor(y, ("media", "p95"), tolerancia=0.005, passo=1_000)
    assert r.convergiu_em["p95"] > r.convergiu_em["media"]


def test_media_proxima_de_zero_avisa_em_vez_de_nunca_convergir():
    rng = np.random.default_rng(26)
    y = rng.normal(0.0, 1.0, 20_000)
    r = conv.monitor(y, ("media",), tolerancia=0.01, passo=1_000)
    assert r.convergiu_em["media"] is None
    assert any("Nao convergiu" in a for a in r.avisos)


def test_amostra_de_lhs_recebe_aviso_sobre_independencia():
    rng = np.random.default_rng(27)
    y = rng.normal(10.0, 2.0, 5_000)
    r = conv.monitor(y, ("media",), passo=500, metodo_amostragem="lhs")
    assert any("nao sao independentes" in a.lower() for a in r.avisos)
    r_mc = conv.monitor(y, ("media",), passo=500, metodo_amostragem="mc")
    assert not any("independentes" in a.lower() for a in r_mc.avisos)


@pytest.mark.parametrize("kw", [{"tolerancia": 0.0}, {"tolerancia": 1.5},
                                {"confianca": 0.0}, {"confianca": 1.0}])
def test_parametros_invalidos_de_convergencia_sao_recusados(kw):
    y = np.random.default_rng(0).normal(size=1_000)
    with pytest.raises(ValueError):
        conv.check(y, "media", **kw)


def test_amostra_curta_demais_e_recusada():
    with pytest.raises(ValueError, match="poucas demais"):
        conv.monitor(np.arange(10.0))


# ---------------------------------------------------------------------------
# Estatisticas novas: todas com valor teorico conhecido para a normal
# ---------------------------------------------------------------------------


def test_desvio_absoluto_medio_da_normal():
    """E[|X - mu|] = sigma * sqrt(2/pi)."""
    y = np.random.default_rng(31).normal(10.0, 3.0, 400_000)
    assert summary.mean_absolute_deviation(y) == pytest.approx(
        3.0 * np.sqrt(2.0 / np.pi), rel=0.01
    )


def test_semi_desvio_da_normal_e_sigma_sobre_raiz_de_dois():
    """Metade da massa fica abaixo da media e contribui metade da variancia."""
    y = np.random.default_rng(32).normal(0.0, 4.0, 400_000)
    assert summary.semi_std(y) == pytest.approx(4.0 / np.sqrt(2.0), rel=0.02)
    assert summary.semi_variance(y) == pytest.approx(16.0 / 2.0, rel=0.03)


def test_semi_variancia_com_limiar_explicito():
    y = np.random.default_rng(33).normal(0.0, 1.0, 200_000)
    assert summary.semi_variance(y, threshold=-10.0) == pytest.approx(0.0, abs=1e-6)
    assert summary.semi_variance(y, threshold=10.0) > summary.semi_variance(y)


def test_as_duas_convencoes_de_curtose_diferem_por_exatamente_tres():
    y = np.random.default_rng(34).standard_t(8, 200_000)
    d = summary.describe(y)
    assert summary.kurtosis_pearson(y) == pytest.approx(
        d["curtose_excesso"] + 3.0, rel=1e-9
    )


def test_curtose_de_pearson_da_normal_e_tres():
    y = np.random.default_rng(35).normal(size=400_000)
    assert summary.kurtosis_pearson(y) == pytest.approx(3.0, abs=0.06)


def test_moda_de_dados_discretos_e_o_valor_mais_frequente():
    y = np.array([1.0] * 10 + [2.0] * 50 + [3.0] * 7)
    assert summary.mode(y) == 2.0


def test_moda_de_continua_cai_perto_do_pico():
    y = np.random.default_rng(36).normal(7.0, 1.0, 200_000)
    assert summary.mode(y) == pytest.approx(7.0, abs=0.15)


def test_percentil_descendente_e_o_complemento_do_ascendente():
    y = np.random.default_rng(37).normal(size=100_000)
    for q in (1, 5, 25, 50, 95):
        assert summary.percentile_descending(y, q) == pytest.approx(
            float(np.percentile(y, 100 - q)), rel=1e-12
        )


def test_amplitude_cresce_com_o_numero_de_iteracoes():
    """Documenta a instabilidade, em vez de deixar o usuario descobrir."""
    rng = np.random.default_rng(38)
    pequena = summary.value_range(rng.normal(size=1_000))
    grande = summary.value_range(rng.normal(size=500_000))
    assert grande > pequena


def test_razao_desvio_sobre_dam_separa_cauda_pesada_de_normal():
    rng = np.random.default_rng(39)
    normal = rng.normal(size=200_000)
    pesada = rng.standard_t(3, 200_000)
    r_normal = normal.std(ddof=1) / summary.mean_absolute_deviation(normal)
    r_pesada = pesada.std(ddof=1) / summary.mean_absolute_deviation(pesada)
    assert r_normal == pytest.approx(np.sqrt(np.pi / 2.0), rel=0.02)
    assert r_pesada > r_normal * 1.2


# ---------------------------------------------------------------------------
# Copulas arquimedianas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fam", copula.ARQUIMEDIANAS)
def test_tau_de_kendall_obtido_bate_com_o_alvo(fam):
    theta = copula.theta_from_tau(fam, 0.5)
    u = copula.archimedean_copula_u(fam, theta, 100_000, 2,
                                    np.random.default_rng(41))
    obtido = float(stats.kendalltau(u[:, 0], u[:, 1]).statistic)
    assert obtido == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("fam,tau", [(f, t) for f in copula.ARQUIMEDIANAS
                                     for t in (0.2, 0.5, 0.75)])
def test_conversao_tau_theta_e_inversa_de_si_mesma(fam, tau):
    theta = copula.theta_from_tau(fam, tau)
    assert copula.tau_from_theta(fam, theta) == pytest.approx(tau, abs=1e-6)


def test_formulas_fechadas_de_tau():
    """Clayton: theta/(theta+2). Gumbel: 1 - 1/theta."""
    assert copula.tau_from_theta("clayton", 2.0) == pytest.approx(0.5)
    assert copula.tau_from_theta("gumbel", 2.0) == pytest.approx(0.5)
    assert copula.tau_from_theta("clayton", 6.0) == pytest.approx(0.75)
    assert copula.tau_from_theta("gumbel", 4.0) == pytest.approx(0.75)


def test_clayton_tem_cauda_inferior_e_gumbel_superior():
    """O contraste e o motivo de as duas existirem. Um teste que so checasse
    lambda > 0 passaria mesmo se as familias fossem trocadas."""
    li_c, ls_c = copula.tail_dependence_archimedean("clayton", 2.0)
    li_g, ls_g = copula.tail_dependence_archimedean("gumbel", 2.0)
    assert li_c == pytest.approx(2.0 ** -0.5) and ls_c == 0.0
    assert ls_g == pytest.approx(2.0 - 2.0**0.5) and li_g == 0.0


def test_frank_nao_tem_dependencia_de_cauda_em_lado_nenhum():
    assert copula.tail_dependence_archimedean("frank", 5.0) == (0.0, 0.0)


def test_assimetria_de_cauda_aparece_na_amostra():
    """Com o MESMO tau, Clayton concentra extremos embaixo e Gumbel em cima."""
    rng = np.random.default_rng(42)
    n = 200_000
    uc = copula.archimedean_copula_u(
        "clayton", copula.theta_from_tau("clayton", 0.6), n, 2, rng
    )
    ug = copula.archimedean_copula_u(
        "gumbel", copula.theta_from_tau("gumbel", 0.6), n, 2, rng
    )
    inf_c = copula.empirical_tail_dependence(1 - uc[:, 0], 1 - uc[:, 1], 0.99)
    sup_c = copula.empirical_tail_dependence(uc[:, 0], uc[:, 1], 0.99)
    inf_g = copula.empirical_tail_dependence(1 - ug[:, 0], 1 - ug[:, 1], 0.99)
    sup_g = copula.empirical_tail_dependence(ug[:, 0], ug[:, 1], 0.99)
    assert inf_c > sup_c + 0.3, "Clayton: cauda inferior domina"
    assert sup_g > inf_g + 0.3, "Gumbel: cauda superior domina"


def test_gumbel_com_theta_um_e_independencia():
    u = copula.archimedean_copula_u("gumbel", 1.0, 50_000, 2,
                                    np.random.default_rng(43))
    assert float(stats.kendalltau(u[:, 0], u[:, 1]).statistic) == pytest.approx(
        0.0, abs=0.01
    )


@pytest.mark.parametrize("fam", copula.ARQUIMEDIANAS)
@pytest.mark.parametrize("d", [2, 3, 5])
def test_amostra_arquimediana_e_uniforme_em_cada_margem(fam, d):
    """Propriedade que define uma copula: toda marginal e U(0,1)."""
    u = copula.archimedean_copula_u(fam, 2.0 if fam != "frank" else 5.0,
                                    20_000, d, np.random.default_rng(44))
    assert u.shape == (20_000, d)
    for j in range(d):
        assert stats.kstest(u[:, j], "uniform").pvalue > 0.001


@pytest.mark.parametrize("fam", copula.ARQUIMEDIANAS)
def test_calibragem_por_spearman_atinge_o_alvo(fam):
    alvo = 0.5
    theta = copula.theta_from_spearman(fam, alvo)
    u = copula.archimedean_copula_u(fam, theta, 100_000, 2,
                                    np.random.default_rng(45))
    obtido = float(stats.spearmanr(u[:, 0], u[:, 1]).statistic)
    assert obtido == pytest.approx(alvo, abs=0.02)


def test_arquimedianas_recusam_dependencia_negativa_com_mensagem_util():
    for fam in ("clayton", "gumbel"):
        with pytest.raises(ValueError, match="POSITIVA"):
            copula.theta_from_spearman(fam, -0.3)
        with pytest.raises(ValueError, match="POSITIVA"):
            copula.theta_from_tau(fam, -0.3)


def test_dependencia_forte_demais_e_recusada_em_vez_de_saturar_em_silencio():
    with pytest.raises(ValueError, match="alcanca"):
        copula.theta_from_spearman("clayton", 0.999999)


def test_theta_fora_da_faixa_e_recusado():
    with pytest.raises(ValueError, match="fora da faixa"):
        copula.archimedean_copula_u("gumbel", 0.5, 10, 2, np.random.default_rng(0))


@pytest.mark.parametrize("fam", copula.ARQUIMEDIANAS)
def test_ajuste_de_copula_recupera_a_familia_geradora(fam):
    """Teste mais exigente do modulo: gerar de uma familia e exigir que a
    comparacao por AIC escolha ELA, nao apenas que rode."""
    theta = copula.theta_from_tau(fam, 0.45)
    u = copula.archimedean_copula_u(fam, theta, 4_000, 2,
                                    np.random.default_rng(46))
    ajustes = copula.fit_copula(u[:, 0], u[:, 1])
    assert ajustes[0].familia == fam
    assert ajustes[0].tau == pytest.approx(0.45, abs=0.05)
    assert ajustes[0].theta == pytest.approx(theta, rel=0.20)


def test_ajuste_de_copula_ordena_por_aic_e_reporta_a_cauda():
    u = copula.archimedean_copula_u("clayton", 2.0, 3_000, 2,
                                    np.random.default_rng(47))
    ajustes = copula.fit_copula(u[:, 0], u[:, 1])
    aics = [a.aic for a in ajustes]
    assert aics == sorted(aics)
    rec = ajustes[0].as_record()
    assert {"lambda_inferior", "lambda_superior", "tau_kendall"} <= set(rec)


def test_pseudo_observacoes_ficam_no_aberto_zero_um():
    x = np.random.default_rng(48).normal(size=(200, 2))
    u = copula.pseudo_observacoes(x)
    assert (u > 0).all() and (u < 1).all()
    assert u.shape == (200, 2)


def test_ajuste_de_copula_com_dependencia_negativa_recusa_as_positivas():
    rng = np.random.default_rng(49)
    a = rng.normal(size=3_000)
    b = -a + rng.normal(scale=0.3, size=3_000)
    with pytest.raises(ValueError, match="nenhuma familia"):
        copula.fit_copula(a, b, familias=("clayton", "gumbel"))


def test_media_fora_da_diagonal_e_a_dispersao_medem_o_que_se_perde():
    C = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
    assert copula.rho_medio_fora_da_diagonal(C) == pytest.approx(0.5)
    assert copula.dispersao_fora_da_diagonal(C) == pytest.approx(0.6)


def test_motor_avisa_que_a_arquimediana_achata_a_matriz():
    vs = [Variable(c, c.upper(), "normal", {"mu": 0.0, "sigma": 1.0})
          for c in ("a", "b", "c")]
    C = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
    r = run(SimulationSpec(variables=vs, formula="a+b+c", iterations=8_000,
                           seed=51, correlation=C, dependence="clayton"))
    texto = " ".join(r.notes)
    assert "PERMUTAVEIS" in texto
    assert "DESCARTADA" in texto


# ---------------------------------------------------------------------------
# Incerteza de parametro e de modelo
# ---------------------------------------------------------------------------


def test_erro_padrao_do_bootstrap_bate_com_a_teoria_para_a_normal():
    """ep(mu) = sigma/sqrt(n). E a checagem mais direta de que o bootstrap
    esta reamostrando certo."""
    x = stats.norm(50.0, 8.0).rvs(400, random_state=61)
    inc = fitting.parameter_uncertainty(x, "normal", replicas=300,
                                        rng=np.random.default_rng(62))
    ep = inc.erro_padrao()
    assert ep[0] == pytest.approx(float(np.std(x, ddof=1)) / np.sqrt(400), rel=0.20)
    assert ep[1] == pytest.approx(
        float(np.std(x, ddof=1)) / np.sqrt(2 * 400), rel=0.30
    )


def test_intervalo_do_bootstrap_contem_a_estimativa_pontual():
    x = stats.gamma(3.0, scale=2.0).rvs(300, random_state=63)
    inc = fitting.parameter_uncertainty(x, "gamma", replicas=200,
                                        rng=np.random.default_rng(64))
    ic = inc.intervalo(0.95)
    for j, p in enumerate(inc.params_pontuais):
        assert ic[j, 0] <= p <= ic[j, 1]


def test_incerteza_de_parametro_alarga_a_preditiva():
    """A propriedade que justifica a funcao existir: propagar a incerteza dos
    parametros NAO pode estreitar o resultado."""
    x = stats.norm(0.0, 1.0).rvs(40, random_state=65)
    inc = fitting.parameter_uncertainty(x, "normal", replicas=800,
                                        rng=np.random.default_rng(66))
    n = 300_000
    pontual = stats.norm(*inc.params_pontuais).rvs(n, random_state=67)
    preditiva = fitting.simular_com_incerteza(inc, n, np.random.default_rng(68))
    assert preditiva.std(ddof=1) > pontual.std(ddof=1)
    assert np.percentile(preditiva, 99.5) > np.percentile(pontual, 99.5)


def test_bootstrap_cru_estreitaria_a_preditiva_em_vez_de_alargar():
    """Regressao do defeito que a correcao de vies existe para consertar.

    Reamostrar as replicas do MLE sem refletir propaga o vies para baixo do
    estimador de escala, e ele CANCELA o alargamento que a incerteza de
    locacao deveria produzir. Medido aqui: a preditiva crua fica mais ESTREITA
    que a pontual, que e o oposto do que a funcao promete.
    """
    x = stats.norm(0.0, 1.0).rvs(40, random_state=65)
    inc = fitting.parameter_uncertainty(x, "normal", replicas=800,
                                        rng=np.random.default_rng(66))
    n = 300_000
    pontual = stats.norm(*inc.params_pontuais).rvs(n, random_state=67)
    crua = fitting.simular_com_incerteza(
        inc, n, np.random.default_rng(68), corrigir_vies=False
    )
    corrigida = fitting.simular_com_incerteza(inc, n, np.random.default_rng(68))
    assert crua.var(ddof=1) < pontual.var(ddof=1) * 1.01
    assert corrigida.var(ddof=1) > crua.var(ddof=1) * 1.05


def test_preditiva_corrigida_se_aproxima_da_solucao_bayesiana_exata():
    """Ancora fechada: com priori de Jeffreys, a preditiva da normal e uma t
    de Student com n-1 graus de liberdade, escala s*sqrt(1+1/n). Variancia
    exata = s^2 * (1 + 1/n) * (n-1)/(n-3).
    """
    n_dados = 40
    x = stats.norm(0.0, 1.0).rvs(n_dados, random_state=65)
    inc = fitting.parameter_uncertainty(x, "normal", replicas=1_000,
                                        rng=np.random.default_rng(66))
    amostra = fitting.simular_com_incerteza(inc, 300_000,
                                            np.random.default_rng(68))
    s2 = float(np.var(x, ddof=1))
    exata = s2 * (1 + 1 / n_dados) * (n_dados - 1) / (n_dados - 3)
    pontual = float(np.var(x, ddof=0))
    obtida = float(amostra.var(ddof=1))
    # Nao chega ao valor exato -- o bootstrap e nao parametrico e a reflexao
    # corrige so o vies de primeira ordem. O que o teste exige e que feche a
    # maior parte da distancia: nada menos que 70% dela.
    fechado = (obtida - pontual) / (exata - pontual)
    assert 0.70 <= fechado <= 1.30, f"fechou {fechado:.0%} da distancia"


def test_o_alargamento_encolhe_quando_ha_mais_dados():
    """Efeito proporcional a 1/n: com amostra grande, ignorar a incerteza de
    parametro passa a ser defensavel - e o teste mede isso em vez de afirmar."""
    def excesso(n_dados: int) -> float:
        x = stats.norm(0.0, 1.0).rvs(n_dados, random_state=69)
        inc = fitting.parameter_uncertainty(x, "normal", replicas=300,
                                            rng=np.random.default_rng(70))
        n = 120_000
        pontual = stats.norm(*inc.params_pontuais).rvs(n, random_state=71)
        pred = fitting.simular_com_incerteza(inc, n, np.random.default_rng(72))
        return float(pred.std(ddof=1) / pontual.std(ddof=1))

    assert excesso(25) > excesso(1_000)


def test_bootstrap_com_poucas_replicas_avisa():
    x = stats.norm(0.0, 1.0).rvs(100, random_state=73)
    inc = fitting.parameter_uncertainty(x, "normal", replicas=50,
                                        rng=np.random.default_rng(74))
    assert any("instaveis" in a for a in inc.avisos)


def test_bootstrap_com_dados_de_menos_e_recusado():
    with pytest.raises(ValueError, match="poucas demais"):
        fitting.parameter_uncertainty(np.arange(5.0), "normal")


def test_media_de_modelos_usa_os_pesos_de_akaike():
    x = stats.lognorm(0.6, scale=np.exp(3)).rvs(150, random_state=75)
    res = fitting.fit_many(x, ["normal", "lognormal", "gamma", "weibull"])
    ma = fitting.model_average(res)
    assert ma.pesos.sum() == pytest.approx(1.0)
    assert (ma.pesos > 0).all()
    assert ma.nomes[0] == res[0].name
    esperado = fitting.akaike_weights(res)
    assert ma.peso_do_primeiro == pytest.approx(float(esperado.max()), abs=0.02)


def test_media_de_modelos_avisa_quando_a_vencedora_domina():
    x = stats.norm(100.0, 5.0).rvs(3_000, random_state=76)
    res = fitting.fit_many(x, ["normal", "logistic", "gumbel_r"])
    ma = fitting.model_average(res)
    assert any("praticamente indistinguivel" in a for a in ma.avisos)


def test_mistura_e_sorteio_de_familia_nao_media_de_quantis():
    """Media de quantis produziria uma curva com cauda MAIS LEVE que a mais
    pesada das candidatas. A mistura nao pode ter essa propriedade."""
    x = stats.lognorm(0.8, scale=np.exp(2)).rvs(200, random_state=77)
    res = fitting.fit_many(x, ["normal", "lognormal", "gamma"])
    ma = fitting.model_average(res, peso_minimo=1e-6)
    amostra = fitting.simular_media_de_modelos(ma, 200_000,
                                               np.random.default_rng(78))
    caudas = [
        float(np.percentile(
            fitting.FITTABLE[c].rvs(*p, size=200_000, random_state=79), 99.5
        ))
        for c, p in zip(ma.chaves, ma.params)
    ]
    p995 = float(np.percentile(amostra, 99.5))
    assert p995 <= max(caudas) * 1.05
    assert p995 >= min(caudas) * 0.95


def test_media_de_modelos_sem_ajustes_e_recusada():
    with pytest.raises(ValueError, match="nenhum ajuste"):
        fitting.model_average([])
