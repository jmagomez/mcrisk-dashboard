"""
Validacao estatistica do motor contra resultados fechados e propriedades teoricas.

A pergunta destes testes nao e "o codigo roda?", e sim "o numero que sai e o
numero certo?". Cada caso confronta a simulacao com algo que se sabe de forma
independente: uma media analitica, uma taxa de convergencia, uma cobertura
nominal, um limite assintotico.

Testes de Monte Carlo sao estocasticos por natureza. Todos aqui usam seed fixa e
tolerancias derivadas do proprio erro-padrao teorico, nao chutadas -- uma
tolerancia folgada demais aceita motor quebrado, e uma apertada demais gera
falha intermitente que ensina a equipe a ignorar o CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mcrisk import summary
from mcrisk.correlation import achieved_spearman, iman_conover
from mcrisk.engine import SimulationSpec, Variable, run, run_replicates


def _spec(**kw) -> SimulationSpec:
    base = dict(
        variables=[Variable("x", "X", "normal", {"mu": 0.0, "sigma": 1.0})],
        formula="x",
        iterations=20_000,
        method="mc",
        seed=20260809,
    )
    base.update(kw)
    return SimulationSpec(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Convergencia: o erro tem de cair com 1/sqrt(N)
# ---------------------------------------------------------------------------

def test_erro_padrao_cai_com_raiz_de_n():
    """Quadruplicar N tem de reduzir o erro-padrao pela metade, +-10%."""
    ses = []
    for n in (25_000, 100_000):
        r = run(_spec(iterations=n))
        ses.append(summary.mc_standard_error(r.output))
    razao = ses[0] / ses[1]
    assert razao == pytest.approx(2.0, rel=0.10), f"razao observada {razao:.3f}"


def test_erro_padrao_bate_com_sigma_sobre_raiz_n():
    n = 100_000
    r = run(_spec(iterations=n))
    esperado = 1.0 / np.sqrt(n)  # sigma = 1 na normal padrao
    assert summary.mc_standard_error(r.output) == pytest.approx(esperado, rel=0.02)


def test_iteracoes_para_precisao_e_coerente_com_o_erro_observado():
    """A funcao que dimensiona a amostra tem de acertar na pratica."""
    piloto = run(_spec(iterations=10_000))
    alvo = 0.005
    n_nec = summary.iterations_for_precision(piloto.output, alvo)
    obtido = summary.mc_standard_error(run(_spec(iterations=int(n_nec))).output)
    assert obtido <= alvo * 1.15, f"pedido {alvo}, obtido {obtido:.5f} com n={n_nec}"


# ---------------------------------------------------------------------------
# 2. Solucoes analiticas
# ---------------------------------------------------------------------------

def test_soma_de_normais_independentes():
    """X+Y ~ N(mu1+mu2, sqrt(s1^2+s2^2)). Sem correlacao, sem termo cruzado."""
    r = run(_spec(
        variables=[Variable("a", "A", "normal", {"mu": 10.0, "sigma": 3.0}),
                   Variable("b", "B", "normal", {"mu": -4.0, "sigma": 4.0})],
        formula="a + b", iterations=200_000,
    ))
    dp_teorico = np.sqrt(9.0 + 16.0)
    assert r.output.mean() == pytest.approx(6.0, abs=4 * dp_teorico / np.sqrt(200_000))
    assert r.output.std(ddof=1) == pytest.approx(dp_teorico, rel=0.01)


def test_lognormal_reproduz_media_e_variancia_fechadas():
    mu_log, s_log = 1.0, 0.5
    r = run(_spec(
        variables=[Variable("x", "X", "lognormal_log",
                            {"mu_log": mu_log, "sigma_log": s_log})],
        formula="x", iterations=400_000,
    ))
    media = np.exp(mu_log + s_log ** 2 / 2)
    var = (np.exp(s_log ** 2) - 1) * np.exp(2 * mu_log + s_log ** 2)
    assert r.output.mean() == pytest.approx(media, rel=0.01)
    assert r.output.var(ddof=1) == pytest.approx(var, rel=0.05)


def test_var_e_cvar_da_normal_contra_formula_fechada():
    """CVaR_q da normal = mu + sigma * phi(z_q)/(1-q). Identidade exata."""
    mu, sigma, q = 0.0, 1.0, 0.95
    r = run(_spec(iterations=500_000))
    z = stats.norm.ppf(q)
    var_teorico = mu + sigma * z
    cvar_teorico = mu + sigma * stats.norm.pdf(z) / (1 - q)
    assert summary.value_at_risk(r.output, q) == pytest.approx(var_teorico, abs=0.02)
    obtido = float(np.mean(r.output[r.output >= var_teorico]))
    assert obtido == pytest.approx(cvar_teorico, rel=0.02)


def test_media_de_uniforme_e_triangular():
    r = run(_spec(
        variables=[Variable("u", "U", "uniform", {"a": 2.0, "b": 8.0}),
                   Variable("t", "T", "triangular",
                            {"minimo": 0.0, "moda": 1.0, "maximo": 5.0})],
        formula="u + t", iterations=300_000,
    ))
    assert r.output.mean() == pytest.approx(5.0 + 2.0, rel=0.005)


# ---------------------------------------------------------------------------
# 3. Cobertura dos intervalos de confianca
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nivel", [0.90, 0.95])
def test_cobertura_do_ic_da_media(nivel):
    """Repetindo o experimento, o IC tem de conter a media verdadeira na taxa nominal.

    Este e o unico teste que valida um IC de verdade. Conferir que o IC "existe"
    ou que a largura "parece razoavel" nao testa nada: o que define um IC de 95%
    e errar 5% das vezes.
    """
    rng = np.random.default_rng(4242)
    repeticoes, dentro = 300, 0
    for _ in range(repeticoes):
        y = rng.normal(0.0, 1.0, 4_000)
        lo, hi = summary.mean_ci(y, nivel)
        dentro += int(lo <= 0.0 <= hi)
    taxa = dentro / repeticoes
    erro = np.sqrt(nivel * (1 - nivel) / repeticoes)
    assert abs(taxa - nivel) < 3 * erro, f"cobertura {taxa:.3f} para nominal {nivel}"


def test_cobertura_do_ic_de_percentil():
    """IC nao parametrico do P90: cobertura empirica proxima da nominal."""
    rng = np.random.default_rng(99)
    verdadeiro = stats.norm.ppf(0.90)
    repeticoes, dentro = 300, 0
    for _ in range(repeticoes):
        y = rng.normal(0.0, 1.0, 3_000)
        lo, hi = summary.quantile_ci(y, 0.90, 0.95)
        dentro += int(lo <= verdadeiro <= hi)
    taxa = dentro / repeticoes
    assert 0.90 <= taxa <= 1.0, f"cobertura {taxa:.3f}"


# ---------------------------------------------------------------------------
# 4. Correlacao: fidelidade do Iman-Conover
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alvo", [0.0, 0.3, -0.5, 0.8, -0.9])
def test_iman_conover_atinge_o_spearman_pedido(alvo):
    rng = np.random.default_rng(7)
    n = 30_000
    X = np.column_stack([rng.normal(size=n), rng.gamma(2.0, 3.0, size=n)])
    C = np.array([[1.0, alvo], [alvo, 1.0]])
    Y = iman_conover(X, C, rng=rng, spearman_adjust=True)
    obtido = achieved_spearman(Y)[0, 1]
    assert obtido == pytest.approx(alvo, abs=0.02)


def test_iman_conover_preserva_as_marginais_exatamente():
    """A reordenacao nao pode criar, apagar nem alterar um unico valor."""
    rng = np.random.default_rng(11)
    X = np.column_stack([rng.normal(size=5_000), rng.exponential(size=5_000)])
    C = np.array([[1.0, 0.7], [0.7, 1.0]])
    Y = iman_conover(X, C, rng=rng)
    for j in range(2):
        assert np.array_equal(np.sort(X[:, j]), np.sort(Y[:, j]))


def test_iman_conover_em_cinco_dimensoes():
    rng = np.random.default_rng(13)
    n, k = 40_000, 5
    alvo = np.full((k, k), 0.4)
    np.fill_diagonal(alvo, 1.0)
    X = np.column_stack([rng.normal(size=n) for _ in range(k)])
    Y = iman_conover(X, alvo, rng=rng, spearman_adjust=True)
    obtido = achieved_spearman(Y)
    fora = np.abs(obtido - alvo)[~np.eye(k, dtype=bool)]
    assert fora.max() < 0.03, f"maior desvio {fora.max():.4f}"


# ---------------------------------------------------------------------------
# 5. Reprodutibilidade
# ---------------------------------------------------------------------------

def test_mesma_seed_produz_saida_identica_bit_a_bit():
    a, b = run(_spec(iterations=5_000)), run(_spec(iterations=5_000))
    assert np.array_equal(a.output, b.output)
    assert np.array_equal(a.inputs, b.inputs)


def test_seeds_diferentes_produzem_saidas_diferentes():
    a = run(_spec(iterations=5_000, seed=1))
    b = run(_spec(iterations=5_000, seed=2))
    assert not np.array_equal(a.output, b.output)


def test_replicacoes_usam_fluxos_distintos():
    _, outs = run_replicates(_spec(iterations=2_000), replicates=5)
    medias = [float(o.mean()) for o in outs]
    assert len(set(medias)) == 5, "replicacoes nao podem repetir a mesma sequencia"


def test_correlacao_nao_depende_da_ordem_de_chamada():
    """Duas execucoes da mesma spec correlacionada tem de coincidir."""
    C = np.array([[1.0, 0.6], [0.6, 1.0]])
    kw = dict(
        variables=[Variable("a", "A", "normal", {"mu": 0.0, "sigma": 1.0}),
                   Variable("b", "B", "lognormal_log", {"mu_log": 0.0, "sigma_log": 0.5})],
        formula="a * b", iterations=8_000, correlation=C,
    )
    assert np.array_equal(run(_spec(**kw)).output, run(_spec(**kw)).output)


# ---------------------------------------------------------------------------
# 6. Latin Hypercube: reduz variancia sem deslocar a media
# ---------------------------------------------------------------------------

def test_lhs_reduz_a_variancia_da_media_estimada():
    """Para funcao monotona, LHS tem de bater MC na dispersao entre replicacoes.

    Este e o motivo de LHS existir. Se o ganho nao aparecer, a estratificacao
    esta quebrada -- e o codigo continuaria passando em todos os outros testes.
    """
    medias = {}
    for metodo in ("mc", "lhs"):
        vals = []
        for s in range(40):
            r = run(_spec(iterations=2_000, method=metodo, seed=1000 + s,
                          variables=[Variable("x", "X", "uniform", {"a": 0.0, "b": 1.0})],
                          formula="x ** 2"))
            vals.append(float(r.output.mean()))
        medias[metodo] = np.array(vals)
    # a media continua correta nos dois: E[U^2] = 1/3
    assert medias["lhs"].mean() == pytest.approx(1 / 3, abs=0.005)
    assert medias["mc"].mean() == pytest.approx(1 / 3, abs=0.005)
    # e a dispersao entre replicacoes cai de forma substancial
    assert medias["lhs"].std(ddof=1) < 0.5 * medias["mc"].std(ddof=1)


def test_lhs_cobre_todos_os_estratos():
    """Com k=1, os n valores tem de cair um em cada faixa de largura 1/n."""
    from mcrisk.sampling import unit_samples

    n = 500
    u = unit_samples(n, 1, method="lhs", rng=np.random.default_rng(3))[:, 0]
    estratos = np.floor(u * n).astype(int)
    assert len(np.unique(estratos)) == n, "cada estrato tem de receber exatamente um ponto"
