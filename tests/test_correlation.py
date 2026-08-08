"""
Verificacao do metodo de Iman-Conover.

Inclui um teste que MEDE se a correcao arcsin (Spearman -> Pearson) melhora
a recuperacao do alvo, em vez de assumir que sim. O resultado desse teste
e o que justifica o padrao `spearman_adjust=True` no codigo.
"""

import numpy as np
import pytest
from scipy import stats

from mcrisk import correlation as c
from mcrisk import distributions as d
from mcrisk.sampling import unit_samples


def _amostra_marginais_variadas(n, rng):
    """Marginais deliberadamente diferentes entre si e nao normais."""
    u = unit_samples(n, 3, method="lhs", rng=rng)
    x = np.empty((n, 3))
    x[:, 0] = d.get("lognormal_real").ppf(u[:, 0], {"media": 100, "desvio": 40})
    x[:, 1] = d.get("pert").ppf(
        u[:, 1], {"minimo": 1, "moda": 3, "maximo": 20, "lam": 4.0}
    )
    x[:, 2] = d.get("normal").ppf(u[:, 2], {"mu": -5, "sigma": 2})
    return x


ALVO = np.array([[1.0, 0.8, -0.5], [0.8, 1.0, -0.3], [-0.5, -0.3, 1.0]])


def test_marginais_sao_preservadas_exatamente():
    """A propriedade central do metodo: so reordena, nao transforma."""
    rng = np.random.default_rng(11)
    X = _amostra_marginais_variadas(20_000, rng)
    Y = c.iman_conover(X, ALVO, rng=rng)
    for j in range(X.shape[1]):
        assert np.array_equal(np.sort(X[:, j]), np.sort(Y[:, j])), (
            f"coluna {j}: o conjunto de valores mudou"
        )


def test_correlacao_alvo_e_atingida():
    rng = np.random.default_rng(12)
    X = _amostra_marginais_variadas(50_000, rng)
    Y = c.iman_conover(X, ALVO, rng=rng, spearman_adjust=True)
    err_max, err_med, _ = c.correlation_error(Y, ALVO)
    assert err_max < 0.02, f"erro maximo {err_max:.4f}"
    assert err_med < 0.01, f"erro medio {err_med:.4f}"


def test_correcao_arcsin_melhora_a_recuperacao_do_alvo():
    """Mede as duas variantes em vez de assumir qual e melhor.

    Sem correcao, o metodo mira a correlacao de Pearson dos escores normais;
    o Spearman resultante fica sistematicamente abaixo do alvo. Com a
    correcao rho_P = 2 sin(pi rho_S / 6) o vies deve praticamente sumir.
    """
    n, reps = 20_000, 5
    err_com, err_sem = [], []
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        X = _amostra_marginais_variadas(n, rng)
        Y1 = c.iman_conover(X, ALVO, rng=np.random.default_rng(500 + r),
                            spearman_adjust=True)
        Y0 = c.iman_conover(X, ALVO, rng=np.random.default_rng(500 + r),
                            spearman_adjust=False)
        err_com.append(c.correlation_error(Y1, ALVO)[1])
        err_sem.append(c.correlation_error(Y0, ALVO)[1])
    m_com, m_sem = float(np.mean(err_com)), float(np.mean(err_sem))
    assert m_com < m_sem, (
        f"a correcao arcsin nao ajudou: com={m_com:.5f} sem={m_sem:.5f}. "
        f"Se este teste falhar, o padrao spearman_adjust=True deve ser revisto."
    )
    # O vies sem correcao deve ser materialmente maior, nao apenas ruido.
    assert m_sem > 2 * m_com


def test_relacoes_arcsin_sao_inversas_uma_da_outra():
    r = np.linspace(-0.99, 0.99, 41)
    assert np.allclose(c.pearson_to_spearman_normal(c.spearman_to_pearson_normal(r)), r)


def test_correlacao_zero_produz_independencia_aproximada():
    rng = np.random.default_rng(13)
    X = _amostra_marginais_variadas(30_000, rng)
    I = np.eye(3)
    Y = c.iman_conover(X, I, rng=rng)
    err_max, _, _ = c.correlation_error(Y, I)
    assert err_max < 0.02


def test_matriz_nao_psd_e_detectada():
    """Correlacoes mutuamente incompativeis devem ser sinalizadas."""
    impossivel = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    errs = c.check_correlation_matrix(impossivel)
    assert any("positiva semidefinida" in e for e in errs)


def test_matriz_nao_psd_e_reparada_e_fica_proxima():
    impossivel = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    reparada = c.nearest_psd_correlation(impossivel)
    assert c.check_correlation_matrix(reparada) == []
    assert np.allclose(np.diag(reparada), 1.0)
    # deve ser a mais proxima possivel, nao a identidade
    assert np.linalg.norm(reparada - impossivel) < np.linalg.norm(
        np.eye(3) - impossivel
    )


def test_iman_conover_com_matriz_nao_psd_nao_quebra():
    rng = np.random.default_rng(14)
    X = _amostra_marginais_variadas(5_000, rng)
    impossivel = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    Y = c.iman_conover(X, impossivel, rng=rng, repair_psd=True)
    assert np.all(np.isfinite(Y))
    with pytest.raises(np.linalg.LinAlgError):
        c.iman_conover(X, impossivel, rng=rng, repair_psd=False)


def test_matriz_de_validacao_rejeita_formatos_errados():
    assert c.check_correlation_matrix(np.array([[1.0, 0.5]]))
    assert c.check_correlation_matrix(np.array([[1.0, 0.5], [0.4, 1.0]]))
    assert c.check_correlation_matrix(np.array([[2.0, 0.0], [0.0, 1.0]]))
    assert c.check_correlation_matrix(np.eye(3)) == []


def test_uma_unica_variavel_e_no_op():
    X = np.random.default_rng(0).normal(size=(100, 1))
    Y = c.iman_conover(X, np.array([[1.0]]))
    assert np.array_equal(X, Y)


def test_correlacao_negativa_forte():
    rng = np.random.default_rng(15)
    u = unit_samples(30_000, 2, method="lhs", rng=rng)
    X = np.column_stack([
        d.get("gamma").ppf(u[:, 0], {"forma": 2, "escala": 3, "loc": 0}),
        d.get("weibull").ppf(u[:, 1], {"forma": 1.5, "escala": 4, "loc": 0}),
    ])
    alvo = np.array([[1.0, -0.85], [-0.85, 1.0]])
    Y = c.iman_conover(X, alvo, rng=rng)
    obtido = stats.spearmanr(Y[:, 0], Y[:, 1]).statistic
    assert abs(obtido - (-0.85)) < 0.02
