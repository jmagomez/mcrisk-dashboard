"""
Verificacao dos esquemas de amostragem.

O teste central mede se LHS realmente reduz a variancia do estimador da
media em relacao ao Monte Carlo simples - que e a unica razao para
preferi-lo. Se a implementacao estiver errada (por exemplo, permutando
todas as colunas com a MESMA permutacao), o ganho desaparece.
"""

import numpy as np
import pytest

from mcrisk import distributions as d
from mcrisk.sampling import VALID_METHODS, effective_iterations_note, unit_samples


@pytest.mark.parametrize("method", VALID_METHODS)
def test_saida_esta_no_intervalo_aberto(method):
    u = unit_samples(5000, 4, method=method, rng=np.random.default_rng(0))
    assert u.shape == (5000, 4)
    assert np.all(u > 0.0) and np.all(u < 1.0)


def test_lhs_preenche_todos_os_estratos():
    """Propriedade definidora do LHS: um ponto por estrato, por dimensao."""
    n = 1000
    u = unit_samples(n, 3, method="lhs", rng=np.random.default_rng(1))
    for j in range(3):
        estratos = np.floor(u[:, j] * n).astype(int)
        assert len(np.unique(estratos)) == n, (
            f"dimensao {j}: {len(np.unique(estratos))} estratos ocupados de {n}"
        )


def test_mc_nao_preenche_todos_os_estratos():
    """Contraste: com MC simples ha colisoes de estrato (problema do aniversario)."""
    n = 1000
    u = unit_samples(n, 1, method="mc", rng=np.random.default_rng(2))
    estratos = np.floor(u[:, 0] * n).astype(int)
    assert len(np.unique(estratos)) < n


def test_colunas_do_lhs_sao_permutadas_independentemente():
    """Guarda contra o bug de reutilizar a mesma permutacao em todas as colunas."""
    u = unit_samples(4000, 2, method="lhs", rng=np.random.default_rng(3))
    rho = np.corrcoef(u[:, 0], u[:, 1])[0, 1]
    assert abs(rho) < 0.06, f"colunas do LHS correlacionadas: rho={rho:.4f}"


def test_lhs_reduz_a_variancia_do_estimador_da_media():
    """O ganho de eficiencia do LHS, medido e nao assumido."""
    n, reps = 500, 300
    params = {"minimo": 1.0, "moda": 4.0, "maximo": 10.0, "lam": 4.0}
    medias = {"mc": [], "lhs": []}
    for m in ("mc", "lhs"):
        for r in range(reps):
            u = unit_samples(n, 1, method=m, rng=np.random.default_rng(1000 + r))[:, 0]
            medias[m].append(np.mean(d.get("pert").ppf(u, params)))
    var_mc = float(np.var(medias["mc"], ddof=1))
    var_lhs = float(np.var(medias["lhs"], ddof=1))
    assert var_lhs < var_mc, f"var_lhs={var_lhs:.6g} nao e menor que var_mc={var_mc:.6g}"
    # Para uma funcao monotona da uniforme, a reducao esperada e substancial.
    assert var_lhs < 0.25 * var_mc, (
        f"reducao de variancia menor que a esperada: razao={var_lhs/var_mc:.3f}"
    )


def test_ambos_os_metodos_sao_nao_viesados():
    """Reducao de variancia nao pode vir as custas de vies."""
    n, reps = 2000, 200
    verdadeiro = (1.0 + 4 * 4.0 + 10.0) / 6.0  # media PERT
    params = {"minimo": 1.0, "moda": 4.0, "maximo": 10.0, "lam": 4.0}
    for m in ("mc", "lhs"):
        medias = [
            np.mean(
                d.get("pert").ppf(
                    unit_samples(n, 1, method=m, rng=np.random.default_rng(r))[:, 0],
                    params,
                )
            )
            for r in range(reps)
        ]
        erro = abs(np.mean(medias) - verdadeiro)
        se = np.std(medias, ddof=1) / np.sqrt(reps)
        assert erro < 4 * se, f"{m}: possivel vies, erro={erro:.5g}, se={se:.5g}"


def test_reprodutibilidade_com_a_mesma_seed():
    a = unit_samples(500, 3, method="lhs", rng=np.random.default_rng(42))
    b = unit_samples(500, 3, method="lhs", rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_seeds_diferentes_produzem_amostras_diferentes():
    a = unit_samples(500, 3, method="lhs", rng=np.random.default_rng(1))
    b = unit_samples(500, 3, method="lhs", rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_lhs_median_e_deterministico_dado_o_embaralhamento():
    u = unit_samples(100, 1, method="lhs_median", rng=np.random.default_rng(5))
    esperado = (np.arange(100) + 0.5) / 100
    assert np.allclose(np.sort(u[:, 0]), esperado)


def test_parametros_invalidos_sao_rejeitados():
    with pytest.raises(ValueError):
        unit_samples(0, 2)
    with pytest.raises(ValueError):
        unit_samples(10, 0)
    with pytest.raises(ValueError):
        unit_samples(10, 2, method="sobol")  # nao implementado


def test_nota_sobre_independencia_diferencia_os_metodos():
    assert "i.i.d" in effective_iterations_note("mc")
    assert "NAO sao independentes" in effective_iterations_note("lhs")
