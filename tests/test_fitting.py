"""
Verificacao do ajuste de distribuicoes.

Dois grupos de teste:
  1. RECUPERACAO DE PARAMETROS: gerando dados de uma distribuicao conhecida,
     o MLE deve recuperar os parametros e o AIC deve preferir o modelo certo.
  2. CALIBRACAO DO TESTE DE ADERENCIA: o ponto metodologico do modulo. O
     p-valor por bootstrap deve ser aproximadamente uniforme sob H0,
     enquanto o teste ingenuo (parametros estimados tratados como conhecidos)
     deve ser conservador demais - aceitando ajustes que deveria rejeitar.
"""

import numpy as np
import pytest
from scipy import stats

from mcrisk import fitting as f


# --------------------------------------------------------------------------
# Recuperacao de parametros
# --------------------------------------------------------------------------


def test_recupera_parametros_da_normal():
    x = stats.norm(loc=50.0, scale=8.0).rvs(5000, random_state=1)
    r = f.fit_distribution(x, "normal")
    assert abs(r.params[0] - 50.0) < 0.5
    assert abs(r.params[1] - 8.0) < 0.5


def test_recupera_parametros_da_lognormal():
    verdadeiro = stats.lognorm(s=0.5, scale=np.exp(2.0))
    x = verdadeiro.rvs(8000, random_state=2)
    r = f.fit_distribution(x, "lognormal")
    ajustado = f.FITTABLE["lognormal"](*r.params)
    assert abs(ajustado.median() - verdadeiro.median()) / verdadeiro.median() < 0.05


def test_recupera_parametros_da_weibull():
    x = stats.weibull_min(1.8, scale=10.0).rvs(8000, random_state=3)
    r = f.fit_distribution(x, "weibull")
    assert abs(r.params[0] - 1.8) < 0.1


@pytest.mark.parametrize(
    "gerador,esperada",
    [
        (stats.norm(loc=0, scale=1), "normal"),
        (stats.expon(scale=3), "exponential"),
        (stats.gumbel_r(loc=0, scale=2), "gumbel_r"),
    ],
)
def test_aic_prefere_a_distribuicao_correta(gerador, esperada):
    x = gerador.rvs(6000, random_state=4)
    resultados = f.fit_many(x)
    top3 = [r.name for r in resultados[:3]]
    assert esperada in top3, f"esperada {esperada} fora do top 3: {top3}"


def test_fit_many_ordena_por_aicc():
    x = stats.norm(loc=0, scale=1).rvs(2000, random_state=5)
    rs = f.fit_many(x)
    aiccs = [r.aicc for r in rs if np.isfinite(r.aicc)]
    assert aiccs == sorted(aiccs)


def test_fit_many_exclui_suporte_positivo_com_dados_negativos():
    """Ajustar lognormal a dados negativos nao faz sentido e deve ser evitado."""
    x = stats.norm(loc=0, scale=1).rvs(2000, random_state=6)
    nomes = {r.name for r in f.fit_many(x)}
    assert "lognormal" not in nomes and "gamma" not in nomes


def test_pesos_de_akaike_somam_um():
    x = stats.norm(loc=0, scale=1).rvs(1000, random_state=7)
    rs = f.fit_many(x)
    w = f.akaike_weights(rs)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w[0] == w.max()  # o primeiro da lista tem o maior peso


def test_aicc_penaliza_mais_que_aic_em_amostra_pequena():
    x = stats.norm(loc=0, scale=1).rvs(30, random_state=8)
    r = f.fit_distribution(x, "normal")
    assert r.aicc > r.aic


def test_bic_penaliza_mais_que_aic_em_amostra_grande():
    x = stats.norm(loc=0, scale=1).rvs(5000, random_state=9)
    r = f.fit_distribution(x, "normal")
    assert r.bic > r.aic


def test_amostra_pequena_demais_e_recusada():
    with pytest.raises(ValueError):
        f.fit_distribution([1.0, 2.0], "normal")


def test_falha_de_ajuste_nao_levanta_excecao():
    """Dados degenerados devem devolver um resultado marcado, nao explodir."""
    r = f.fit_distribution(np.ones(50), "normal")
    assert isinstance(r, f.FitResult)


# --------------------------------------------------------------------------
# Estatisticas de aderencia
# --------------------------------------------------------------------------


def test_ks_bate_com_a_implementacao_do_scipy():
    x = stats.norm(loc=2, scale=3).rvs(500, random_state=10)
    frozen = stats.norm(loc=2, scale=3)
    nosso = f.ks_statistic(x, frozen)
    scipy_stat = stats.kstest(x, frozen.cdf).statistic
    assert abs(nosso - scipy_stat) < 1e-12


def test_ad_bate_com_a_implementacao_do_scipy_para_normal():
    """scipy.stats.anderson usa parametros estimados; comparamos a formula."""
    x = stats.norm(loc=0, scale=1).rvs(500, random_state=11)
    mu, sd = np.mean(x), np.std(x, ddof=1)
    nosso = f.anderson_darling_statistic(x, stats.norm(loc=mu, scale=sd))
    ref = stats.anderson(x, dist="norm").statistic
    assert abs(nosso - ref) < 1e-8


def test_ad_e_maior_quando_o_ajuste_e_pior():
    x = stats.expon(scale=2).rvs(2000, random_state=12)
    bom = f.anderson_darling_statistic(x, stats.expon(scale=2))
    ruim = f.anderson_darling_statistic(x, stats.norm(loc=2, scale=2))
    assert ruim > bom


def test_ks_e_maior_quando_o_ajuste_e_pior():
    x = stats.expon(scale=2).rvs(2000, random_state=13)
    bom = f.ks_statistic(x, stats.expon(scale=2))
    ruim = f.ks_statistic(x, stats.norm(loc=2, scale=2))
    assert ruim > bom


# --------------------------------------------------------------------------
# O ponto metodologico: calibracao do p-valor
# --------------------------------------------------------------------------


def test_pvalor_bootstrap_e_aproximadamente_uniforme_sob_h0():
    """Se H0 e verdadeira, o p-valor deve se distribuir ~U(0,1).

    Este e o criterio que define um teste calibrado. Rodamos M conjuntos de
    dados realmente vindos de uma normal, ajustamos uma normal a cada um e
    verificamos a distribuicao dos p-valores.
    """
    M, n, B = 60, 120, 250
    ps = []
    for m in range(M):
        x = stats.norm(loc=0, scale=1).rvs(n, random_state=1000 + m)
        r = f.fit_distribution(
            x, "normal", bootstrap=B, rng=np.random.default_rng(m)
        )
        if r.ks_pvalue_bootstrap is not None:
            ps.append(r.ks_pvalue_bootstrap)
    ps = np.asarray(ps)
    assert ps.size > 50
    # A media de uma U(0,1) e 0.5; com M=60 o erro padrao e ~0.037.
    assert 0.38 < ps.mean() < 0.62, f"media dos p-valores = {ps.mean():.3f}"
    # Taxa de rejeicao a 5% deve ficar proxima de 5%, nao de 0%.
    assert (ps < 0.05).mean() < 0.20


def test_teste_ingenuo_e_conservador_demais_quando_parametros_sao_estimados():
    """Demonstra por que o p-valor assintotico do K-S nao e reportado.

    Tratar os parametros estimados como se fossem conhecidos infla o
    p-valor: a media fica bem acima de 0.5, e o teste quase nunca rejeita.
    """
    M, n = 120, 120
    ingenuos = []
    for m in range(M):
        x = stats.norm(loc=0, scale=1).rvs(n, random_state=2000 + m)
        mu, sd = np.mean(x), np.std(x, ddof=1)
        ingenuos.append(stats.kstest(x, stats.norm(loc=mu, scale=sd).cdf).pvalue)
    ingenuos = np.asarray(ingenuos)
    assert ingenuos.mean() > 0.75, (
        f"esperava-se p-valor ingenuo inflado; media = {ingenuos.mean():.3f}"
    )
    # Praticamente nunca rejeita a 5%, mesmo sendo o teste de tamanho nominal 5%.
    assert (ingenuos < 0.05).mean() < 0.02


def test_bootstrap_rejeita_ajuste_claramente_errado():
    """Poder do teste: dados exponenciais nao devem passar por normais."""
    x = stats.expon(scale=2).rvs(200, random_state=14)
    r = f.fit_distribution(x, "normal", bootstrap=200, rng=np.random.default_rng(0))
    assert r.ks_pvalue_bootstrap is not None
    assert r.ks_pvalue_bootstrap < 0.05


def test_bootstrap_desligado_nao_reporta_pvalor():
    x = stats.norm().rvs(200, random_state=15)
    r = f.fit_distribution(x, "normal", bootstrap=0)
    assert r.ks_pvalue_bootstrap is None and r.ad_pvalue_bootstrap is None


def test_qq_points_tem_o_tamanho_certo_e_e_monotono():
    x = stats.norm(loc=3, scale=2).rvs(500, random_state=16)
    r = f.fit_distribution(x, "normal")
    teo, amostral = f.qq_points(x, r)
    assert teo.size == amostral.size == 500
    assert np.all(np.diff(amostral) >= 0)
    # ajuste bom => pontos proximos da diagonal
    assert np.corrcoef(teo, amostral)[0, 1] > 0.99


def test_describe_data_reporta_n_correto():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    d = f.describe_data(x)
    assert d["n"] == 3
