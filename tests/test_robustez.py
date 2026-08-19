"""
Robustez: parametros degenerados, matrizes impossiveis, formulas patologicas e
dados malformados.

O criterio destes testes nao e "nao quebra". E mais exigente: ou o motor produz
resultado correto, ou recusa com mensagem que diz o que fazer. O que nao pode
acontecer e a terceira via -- devolver numero silenciosamente errado.

Varredura sobre o registro inteiro de distribuicoes: qualquer familia nova
passa a ser exercitada automaticamente, sem que ninguem precise lembrar de
acrescentar um teste.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcrisk import distributions as dists
from mcrisk.correlation import check_correlation_matrix, nearest_psd_correlation
from mcrisk.engine import SimulationSpec, Variable, run
from mcrisk.formula import Formula, FormulaError

# Parametros validos minimos para cada familia do registro.
PARAMS_VALIDOS = {
    "normal": {"mu": 10.0, "sigma": 2.0},
    "lognormal_log": {"mu_log": 0.5, "sigma_log": 0.4},
    "lognormal_real": {"media": 10.0, "desvio": 3.0},
    "triangular": {"minimo": 1.0, "moda": 2.0, "maximo": 5.0},
    "pert": {"minimo": 1.0, "moda": 2.0, "maximo": 5.0, "lam": 4.0},
    "uniform": {"a": 0.0, "b": 4.0},
    "beta": {"alpha": 2.0, "beta_": 3.0, "minimo": 0.0, "maximo": 1.0},
    "gamma": {"forma": 2.0, "escala": 3.0, "loc": 0.0},
    "exponential": {"escala": 2.0, "loc": 0.0},
    "weibull": {"forma": 1.5, "escala": 2.0, "loc": 0.0},
    "student_t": {"gl": 6.0, "loc": 0.0, "escala": 1.0},
    "logistic": {"loc": 0.0, "escala": 1.0},
    "gumbel_r": {"loc": 0.0, "escala": 1.0},
    "pareto": {"b": 3.0, "escala": 1.0},
    "genpareto": {"xi": 0.2, "loc": 0.0, "escala": 1.0},
    "bernoulli": {"p": 0.4},
    "binomial": {"n": 10.0, "p": 0.3},
    "poisson": {"lam": 3.0},
    "negbinom": {"n": 5.0, "p": 0.4},
    "geometric": {"p": 0.3},
    "discrete_uniform": {"a": 1.0, "b": 6.0},
}


def test_o_dicionario_de_teste_cobre_o_registro_inteiro():
    """Guarda-costas: familia nova sem parametros aqui faz este teste falhar."""
    assert set(PARAMS_VALIDOS) == set(dists.REGISTRY), (
        "distribuicoes sem cobertura: "
        f"{sorted(set(dists.REGISTRY) ^ set(PARAMS_VALIDOS))}"
    )


@pytest.mark.parametrize("chave", sorted(PARAMS_VALIDOS))
def test_toda_distribuicao_gera_amostra_finita(chave):
    v = Variable("x", "X", chave, PARAMS_VALIDOS[chave])
    assert v.validate() == []
    r = run(SimulationSpec(variables=[v], formula="x", iterations=4_000, seed=5))
    finitos = np.isfinite(r.output)
    assert finitos.mean() > 0.99, f"{chave}: {(~finitos).sum()} valores nao finitos"


@pytest.mark.parametrize("chave", sorted(PARAMS_VALIDOS))
def test_ppf_e_monotona_nao_decrescente(chave):
    """Propriedade que define uma funcao quantil. Vale para continua e discreta."""
    spec = dists.get(chave)
    u = np.linspace(1e-6, 1 - 1e-6, 2_000)
    x = spec.ppf(u, PARAMS_VALIDOS[chave])
    assert np.all(np.diff(x) >= -1e-9), f"{chave}: ppf nao monotona"


@pytest.mark.parametrize("chave", sorted(PARAMS_VALIDOS))
def test_ppf_respeita_o_suporte_declarado(chave):
    spec = dists.get(chave)
    p = PARAMS_VALIDOS[chave]
    x = spec.ppf(np.linspace(1e-9, 1 - 1e-9, 5_000), p)
    x = x[np.isfinite(x)]
    if chave in ("triangular", "pert"):
        assert x.min() >= p["minimo"] - 1e-9 and x.max() <= p["maximo"] + 1e-9
    if chave in ("uniform", "discrete_uniform"):
        assert x.min() >= p["a"] - 1e-9 and x.max() <= p["b"] + 1e-9
    if chave in ("lognormal_log", "lognormal_real", "gamma", "exponential",
                 "weibull", "pareto", "poisson", "geometric", "negbinom",
                 "binomial", "bernoulli", "genpareto"):
        assert x.min() >= -1e-9, f"{chave} produziu valor negativo: {x.min()}"


# ---------------------------------------------------------------------------
# Parametros degenerados
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chave,params,motivo", [
    ("normal", {"mu": 0.0, "sigma": 0.0}, "desvio nulo"),
    ("normal", {"mu": 0.0, "sigma": -1.0}, "desvio negativo"),
    ("uniform", {"a": 5.0, "b": 5.0}, "intervalo degenerado"),
    ("uniform", {"a": 5.0, "b": 1.0}, "limites invertidos"),
    ("triangular", {"minimo": 5.0, "moda": 1.0, "maximo": 9.0}, "moda fora"),
    ("triangular", {"minimo": 1.0, "moda": 2.0, "maximo": 1.0}, "maximo < minimo"),
    ("beta", {"alpha": 0.0, "beta_": 2.0, "minimo": 0.0, "maximo": 1.0}, "alpha nulo"),
    ("bernoulli", {"p": 1.5}, "probabilidade > 1"),
    ("bernoulli", {"p": -0.1}, "probabilidade < 0"),
    ("poisson", {"lam": -2.0}, "taxa negativa"),
    ("student_t", {"gl": 0.0, "loc": 0.0, "escala": 1.0}, "gl nulo"),
])
def test_parametros_degenerados_sao_recusados_com_mensagem(chave, params, motivo):
    erros = dists.get(chave).validate(params)
    assert erros, f"{chave} com {motivo} deveria ser recusado"
    assert all(isinstance(e, str) and e.strip() for e in erros)


def test_moda_no_extremo_do_triangular_e_aceita_e_produz_amostra_valida():
    """Caso-limite legitimo: distribuicao triangular degenerada em rampa."""
    for params in ({"minimo": 0.0, "moda": 0.0, "maximo": 10.0},
                   {"minimo": 0.0, "moda": 10.0, "maximo": 10.0}):
        v = Variable("x", "X", "triangular", params)
        assert v.validate() == []
        r = run(SimulationSpec(variables=[v], formula="x", iterations=3_000, seed=2))
        assert np.isfinite(r.output).all()
        assert 0.0 <= r.output.min() and r.output.max() <= 10.0


def test_pert_com_lambda_zero_vira_uniforme_no_intervalo():
    v = Variable("x", "X", "pert", {"minimo": 0.0, "moda": 5.0, "maximo": 10.0, "lam": 0.0})
    if v.validate():
        pytest.skip("implementacao recusa lambda=0; comportamento aceitavel")
    r = run(SimulationSpec(variables=[v], formula="x", iterations=50_000, seed=3))
    assert r.output.mean() == pytest.approx(5.0, abs=0.15)


# ---------------------------------------------------------------------------
# Matrizes de correlacao impossiveis
# ---------------------------------------------------------------------------

def test_matriz_nao_psd_e_detectada_e_reparada():
    """Tres variaveis todas com correlacao -0,9 entre si: impossivel."""
    C = np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]])
    assert np.linalg.eigvalsh(C).min() < 0
    assert check_correlation_matrix(C), "matriz impossivel tem de gerar diagnostico"
    R = nearest_psd_correlation(C)
    assert np.linalg.eigvalsh(R).min() >= -1e-10
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T)


def test_simulacao_com_matriz_impossivel_avisa_em_vez_de_calar():
    C = np.array([[1.0, -0.9, -0.9], [-0.9, 1.0, -0.9], [-0.9, -0.9, 1.0]])
    vs = [Variable(f"v{i}", f"V{i}", "normal", {"mu": 0.0, "sigma": 1.0}) for i in range(3)]
    r = run(SimulationSpec(variables=vs, formula="v0+v1+v2", iterations=5_000,
                           seed=4, correlation=C))
    assert any("positiva" in n.lower() or "diferem" in n.lower() for n in r.notes), (
        "reparo silencioso e o pior resultado possivel: os numeros mudam e "
        "ninguem fica sabendo"
    )


@pytest.mark.parametrize("C,motivo", [
    (np.array([[1.0, 1.5], [1.5, 1.0]]), "correlacao fora de [-1,1]"),
    (np.array([[2.0, 0.3], [0.3, 1.0]]), "diagonal diferente de 1"),
    (np.array([[1.0, 0.3], [0.5, 1.0]]), "matriz nao simetrica"),
])
def test_matrizes_malformadas_sao_recusadas(C, motivo):
    assert check_correlation_matrix(C), motivo


def test_matriz_identidade_nao_altera_as_marginais():
    vs = [Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
          Variable("b", "B", "normal", {"mu": 0.0, "sigma": 1.0})]
    kw = dict(variables=vs, formula="a+b", iterations=20_000, seed=9)
    livre = run(SimulationSpec(**kw))
    ident = run(SimulationSpec(**kw, correlation=np.eye(2)))
    assert ident.output.std(ddof=1) == pytest.approx(livre.output.std(ddof=1), rel=0.05)


# ---------------------------------------------------------------------------
# Formulas patologicas
# ---------------------------------------------------------------------------

def test_divisao_por_zero_nao_derruba_e_e_contabilizada():
    v = Variable("x", "X", "discrete_uniform", {"a": 0.0, "b": 3.0})
    r = run(SimulationSpec(variables=[v], formula="1 / x", iterations=5_000, seed=6))
    assert any("nao finita" in n for n in r.notes), (
        "iteracoes perdidas mudam a distribuicao dos resultados e precisam "
        "aparecer, nao serem descartadas em silencio"
    )
    assert np.isfinite(r.output).sum() > 0


def test_log_de_valor_nao_positivo_e_tratado():
    v = Variable("x", "X", "normal", {"mu": 0.0, "sigma": 5.0})
    r = run(SimulationSpec(variables=[v], formula="log(x)", iterations=5_000, seed=8))
    assert np.isfinite(r.output).sum() > 0
    assert any("nao finita" in n for n in r.notes)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "exec('x=1')",
    "lambda: 1",
    "[c for c in ()]",
    "globals()",
])
def test_formula_recusa_construcoes_perigosas(expr):
    with pytest.raises((FormulaError, Exception)):
        Formula(expr, ["x"])


def test_formula_recusa_nome_desconhecido():
    with pytest.raises(FormulaError):
        Formula("x + inexistente", ["x"])


def test_variavel_definida_e_nao_usada_gera_aviso():
    vs = [Variable("a", "A", "normal", {"mu": 1.0, "sigma": 1.0}),
          Variable("b", "B", "normal", {"mu": 1.0, "sigma": 1.0})]
    r = run(SimulationSpec(variables=vs, formula="a", iterations=1_000, seed=1))
    assert any("nao usadas" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Dados malformados na reamostragem empirica
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data,ok", [
    ([1.0], False),
    ([], False),
    ([1.0, 2.0], True),
    ([5.0, 5.0, 5.0], True),
])
def test_serie_empirica_exige_amostra_minima(data, ok):
    v = Variable("x", "X", "empirical", {}, data=data)
    assert (v.validate() == []) is ok


def test_serie_empirica_constante_produz_saida_constante():
    v = Variable("x", "X", "empirical", {}, data=[7.0] * 10)
    r = run(SimulationSpec(variables=[v], formula="x", iterations=1_000, seed=1))
    assert np.allclose(r.output, 7.0)


@pytest.mark.parametrize("valores,probs,ok", [
    ([1.0, 2.0], [0.5, 0.5], True),
    ([1.0, 2.0], [0.5], False),
    ([1.0, 2.0], [-0.1, 1.1], False),
    ([1.0, 2.0], [0.0, 0.0], False),
    ([1.0, 2.0], [2.0, 6.0], True),  # pesos nao normalizados sao aceitaveis
])
def test_discreta_customizada_valida_entradas(valores, probs, ok):
    v = Variable("x", "X", "discrete_custom", {}, values=valores, probs=probs)
    assert (v.validate() == []) is ok


def test_discreta_customizada_respeita_as_probabilidades():
    v = Variable("x", "X", "discrete_custom", {}, values=[0.0, 1.0], probs=[0.25, 0.75])
    r = run(SimulationSpec(variables=[v], formula="x", iterations=100_000,
                           method="mc", seed=1))
    assert r.output.mean() == pytest.approx(0.75, abs=0.01)


# ---------------------------------------------------------------------------
# Especificacoes invalidas
# ---------------------------------------------------------------------------

def test_nomes_duplicados_sao_recusados():
    vs = [Variable("x", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
          Variable("x", "B", "normal", {"mu": 0.0, "sigma": 1.0})]
    with pytest.raises(ValueError, match="duplicados"):
        run(SimulationSpec(variables=vs, formula="x", iterations=100))


def test_zero_variaveis_e_recusado():
    with pytest.raises(ValueError):
        run(SimulationSpec(variables=[], formula="1", iterations=100))


@pytest.mark.parametrize("n", [0, 1, -5])
def test_iteracoes_insuficientes_sao_recusadas(n):
    v = Variable("x", "X", "normal", {"mu": 0.0, "sigma": 1.0})
    with pytest.raises(ValueError):
        run(SimulationSpec(variables=[v], formula="x", iterations=n))


def test_metodo_de_amostragem_invalido_e_recusado():
    v = Variable("x", "X", "normal", {"mu": 0.0, "sigma": 1.0})
    with pytest.raises(ValueError, match="metodo"):
        run(SimulationSpec(variables=[v], formula="x", iterations=100, method="sobol"))


# ---------------------------------------------------------------------------
# Varredura aleatoria
# ---------------------------------------------------------------------------

def test_varredura_aleatoria_de_modelos_nao_produz_falha_inesperada():
    """80 modelos sorteados: ou rodam, ou recusam com ValueError explicativo.

    O que este teste procura nao e um caso especifico -- e a combinacao que
    ninguem pensou em escrever a mao: quatro familias diferentes, formula
    montada por sorteio, correlacao aleatoria.
    """
    rng = np.random.default_rng(2026)
    chaves = sorted(PARAMS_VALIDOS)
    ops = ["+", "-", "*"]
    for i in range(80):
        k = int(rng.integers(1, 4))
        escolhidas = rng.choice(chaves, size=k, replace=False)
        vs = [Variable(f"v{j}", f"V{j}", c, dict(PARAMS_VALIDOS[c]))
              for j, c in enumerate(escolhidas)]
        formula = f"v{0}"
        for j in range(1, k):
            formula += f" {ops[int(rng.integers(0, len(ops)))]} v{j}"
        C = None
        if k > 1 and rng.random() < 0.5:
            rho = float(rng.uniform(-0.6, 0.6))
            C = np.full((k, k), rho)
            np.fill_diagonal(C, 1.0)
        spec = SimulationSpec(variables=vs, formula=formula, iterations=1_500,
                              seed=int(rng.integers(1, 10**6)), correlation=C)
        try:
            r = run(spec)
        except ValueError as e:
            assert str(e).strip(), f"modelo {i}: recusa sem mensagem"
            continue
        assert r.output.shape == (1_500,)
        assert np.isfinite(r.output).mean() > 0.5, f"modelo {i}: {formula}"
