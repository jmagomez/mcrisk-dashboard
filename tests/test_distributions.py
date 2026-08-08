"""
Verificacao das distribuicoes.

Estrategia: comparar momentos AMOSTRAIS com momentos TEORICOS fechados.
Um erro de parametrizacao (troca de escala por taxa, sigma log por sigma
real, etc.) e exatamente o tipo de bug que passa despercebido numa
inspecao visual e aparece aqui.
"""

import numpy as np
import pytest
from scipy import stats

from mcrisk import distributions as d
from mcrisk.sampling import unit_samples

RNG = np.random.default_rng(20240807)
N = 200_000

# Parametros de teste para cada distribuicao do registro.
CASES = {
    "normal": {"mu": 3.0, "sigma": 2.0},
    "lognormal_log": {"mu_log": 0.5, "sigma_log": 0.4},
    "lognormal_real": {"media": 100.0, "desvio": 25.0},
    "triangular": {"minimo": 1.0, "moda": 4.0, "maximo": 10.0},
    "pert": {"minimo": 1.0, "moda": 4.0, "maximo": 10.0, "lam": 4.0},
    "uniform": {"a": -2.0, "b": 5.0},
    "beta": {"alpha": 2.0, "beta_": 5.0, "minimo": 0.0, "maximo": 10.0},
    "gamma": {"forma": 3.0, "escala": 2.0, "loc": 0.0},
    "exponential": {"escala": 4.0, "loc": 0.0},
    "weibull": {"forma": 1.8, "escala": 3.0, "loc": 0.0},
    "student_t": {"gl": 8.0, "loc": 1.0, "escala": 2.0},
    "logistic": {"loc": 2.0, "escala": 1.5},
    "gumbel_r": {"loc": 0.0, "escala": 2.0},
    "pareto": {"b": 4.5, "escala": 1.0},
    "genpareto": {"xi": 0.15, "loc": 0.0, "escala": 2.0},
    "bernoulli": {"p": 0.3},
    "binomial": {"n": 20.0, "p": 0.35},
    "poisson": {"lam": 4.0},
    "negbinom": {"n": 6.0, "p": 0.4},
    "geometric": {"p": 0.25},
    "discrete_uniform": {"a": 1.0, "b": 10.0},
}


def test_todas_as_distribuicoes_tem_caso_de_teste():
    """Se alguem adicionar uma distribuicao sem teste, este teste falha."""
    faltando = set(d.REGISTRY) - set(CASES)
    assert not faltando, f"distribuicoes sem caso de teste: {sorted(faltando)}"


@pytest.mark.parametrize("key", sorted(CASES))
def test_media_amostral_bate_com_a_teorica(key):
    params = CASES[key]
    teo = d.theoretical_moments(key, params)
    u = unit_samples(N, 1, method="lhs", rng=np.random.default_rng(1))[:, 0]
    x = d.get(key).ppf(u, params)
    assert np.all(np.isfinite(x)), f"{key} gerou valores nao finitos"

    # Tolerancia baseada no erro padrao teorico; LHS reduz a variancia,
    # entao 5 erros padrao e folgado o suficiente para nao dar flake.
    se = np.sqrt(teo["variancia"] / N)
    tol = max(5 * se, 1e-3 * max(abs(teo["media"]), 1.0))
    assert abs(np.mean(x) - teo["media"]) < tol, (
        f"{key}: media amostral {np.mean(x):.6g} vs teorica {teo['media']:.6g}"
    )


@pytest.mark.parametrize("key", sorted(CASES))
def test_desvio_amostral_bate_com_o_teorico(key):
    params = CASES[key]
    teo = d.theoretical_moments(key, params)
    u = unit_samples(N, 1, method="lhs", rng=np.random.default_rng(2))[:, 0]
    x = d.get(key).ppf(u, params)
    assert np.isclose(np.std(x, ddof=1), teo["desvio"], rtol=0.02), (
        f"{key}: desvio amostral {np.std(x, ddof=1):.6g} vs teorico "
        f"{teo['desvio']:.6g}"
    )


def test_lognormal_duas_parametrizacoes_sao_consistentes():
    """Ponto classico de erro: mu_log != media de X."""
    media, desvio = 100.0, 25.0
    fr = d.get("lognormal_real").frozen({"media": media, "desvio": desvio})
    assert np.isclose(fr.mean(), media, rtol=1e-10)
    assert np.isclose(fr.std(), desvio, rtol=1e-10)

    # A conversao inversa deve reproduzir a mesma distribuicao.
    sigma_log = np.sqrt(np.log(1 + (desvio / media) ** 2))
    mu_log = np.log(media) - 0.5 * sigma_log**2
    fr2 = d.get("lognormal_log").frozen({"mu_log": mu_log, "sigma_log": sigma_log})
    assert np.isclose(fr.ppf(0.9), fr2.ppf(0.9), rtol=1e-10)


def test_pert_media_segue_a_formula_classica():
    """mu = (min + lambda*moda + max) / (lambda + 2)."""
    a, m, b, lam = 2.0, 5.0, 11.0, 4.0
    esperado = (a + lam * m + b) / (lam + 2)
    fr = d.get("pert").frozen({"minimo": a, "moda": m, "maximo": b, "lam": lam})
    assert np.isclose(fr.mean(), esperado, rtol=1e-12)


def test_pert_tem_menos_massa_nas_caudas_que_triangular():
    """Afirmacao feita na documentacao; aqui ela e verificada."""
    a, m, b = 0.0, 3.0, 10.0
    pert = d.get("pert").frozen({"minimo": a, "moda": m, "maximo": b, "lam": 4.0})
    tri = d.get("triangular").frozen({"minimo": a, "moda": m, "maximo": b})
    assert pert.std() < tri.std()


def test_triangular_respeita_limites():
    fr = d.get("triangular").frozen({"minimo": -5.0, "moda": 0.0, "maximo": 2.0})
    u = np.linspace(1e-9, 1 - 1e-9, 10_000)
    x = fr.ppf(u)
    assert x.min() >= -5.0 - 1e-9 and x.max() <= 2.0 + 1e-9


def test_validacao_rejeita_parametros_invalidos():
    assert d.get("normal").validate({"mu": 0.0, "sigma": -1.0})
    assert d.get("uniform").validate({"a": 5.0, "b": 1.0})
    assert d.get("triangular").validate({"minimo": 0.0, "moda": 9.0, "maximo": 5.0})
    assert d.get("bernoulli").validate({"p": 1.5})
    assert d.get("normal").validate({"mu": 0.0, "sigma": None})
    assert d.get("normal").validate({"mu": np.inf, "sigma": 1.0})
    # caso valido nao produz erro
    assert d.get("normal").validate({"mu": 0.0, "sigma": 1.0}) == []


def test_frozen_levanta_erro_em_parametros_invalidos():
    with pytest.raises(ValueError):
        d.get("normal").frozen({"mu": 0.0, "sigma": 0.0})


def test_discrete_ppf_reproduz_as_probabilidades():
    valores = [10.0, 20.0, 30.0]
    probs = [0.2, 0.5, 0.3]
    u = unit_samples(300_000, 1, method="lhs", rng=np.random.default_rng(3))[:, 0]
    x = d.discrete_ppf(u, valores, probs)
    for v, p in zip(valores, probs):
        assert np.isclose(np.mean(x == v), p, atol=0.005)


def test_discrete_ppf_normaliza_probabilidades_nao_somando_um():
    x = d.discrete_ppf(np.linspace(0.001, 0.999, 1000), [0.0, 1.0], [2.0, 2.0])
    assert np.isclose(np.mean(x), 0.5, atol=0.01)


def test_discrete_ppf_rejeita_entradas_invalidas():
    with pytest.raises(ValueError):
        d.discrete_ppf(np.array([0.5]), [1.0, 2.0], [0.5])
    with pytest.raises(ValueError):
        d.discrete_ppf(np.array([0.5]), [1.0, 2.0], [-0.5, 1.5])


def test_empirical_ppf_nao_extrapola():
    """Limitacao declarada: a empirica nunca ultrapassa o observado."""
    dados = np.array([1.0, 2.0, 3.0, 10.0])
    x = d.empirical_ppf(np.linspace(1e-9, 1 - 1e-9, 5000), dados)
    assert x.min() >= dados.min() and x.max() <= dados.max()
    assert set(np.unique(x)).issubset(set(dados))


def test_ppf_e_monotona_nao_decrescente():
    """Requisito para que LHS e Iman-Conover preservem a estrutura de postos."""
    u = np.linspace(1e-6, 1 - 1e-6, 5000)
    for key, params in CASES.items():
        x = d.get(key).ppf(u, params)
        assert np.all(np.diff(x) >= -1e-9), f"{key}: ppf nao monotona"


def test_pareto_com_indice_baixo_tem_variancia_infinita():
    """Guarda contra a suposicao de que toda media amostral converge."""
    teo = d.theoretical_moments("pareto", {"b": 1.5, "escala": 1.0})
    assert not np.isfinite(teo["variancia"])
    teo1 = d.theoretical_moments("pareto", {"b": 0.8, "escala": 1.0})
    assert not np.isfinite(teo1["media"])
