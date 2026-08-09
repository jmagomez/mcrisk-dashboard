"""
Testes de auditoria: defeitos encontrados na revisao do app.

Cada teste aqui nasceu de uma suspeita levantada lendo o codigo, e foi
executado ANTES da correcao para confirmar que o defeito era real. Um teste
de bug que ja passa no codigo defeituoso nao vale nada.

Os quatro defeitos confirmados:

1. A matriz de correlacao preenchida em um triangulo so era "simetrizada"
   com (C + C.T)/2, o que divide por dois toda correlacao digitada. O app
   rodava, nao reclamava, e usava metade do que o usuario pediu.
2. O resultado do ajuste de distribuicoes sumia da tela em qualquer rerun,
   tornando o seletor "Inspecionar ajuste" inutilizavel.
3. Dois rotulos iguais derrubavam a aba de correlacao com uma
   StreamlitAPIException nao tratada.
4. `achieved_spearman` devolvia matriz 1x1 quando havia exatamente duas
   variaveis, porque scipy.spearmanr retorna escalar nesse caso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st_
from streamlit.testing.v1 import AppTest

from mcrisk.correlation import mirror_triangle

TIMEOUT = 300


def novo_app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def add_var(at, vid, label, dist, params):
    at.button[0].click().run()
    at.text_input(key=f"lab{vid}").set_value(label).run()
    at.selectbox(key=f"dist{vid}").set_value(dist).run()
    for nome, valor in params.items():
        at.number_input(key=f"p{vid}_{nome}").set_value(float(valor)).run()
    return at


def botao(at, trecho):
    return next(b for b in at.button if trecho in b.label)


def campo_formula(at):
    return next(t for t in at.text_input if t.label == "Expressao")


def metrica(at, rotulo):
    return next(m for m in at.metric if m.label == rotulo).value


def num(txt):
    return float(txt.replace(",", ""))


# ===========================================================================
# DEFEITO 1 - correlacao dividida por dois
# ===========================================================================

# --- nivel unitario -------------------------------------------------------


def test_espelha_triangulo_superior():
    C = np.eye(3)
    C[0, 1] = 0.8
    C[0, 2] = -0.5
    out, conflitos = mirror_triangle(C)
    assert conflitos == []
    assert out[0, 1] == out[1, 0] == pytest.approx(0.8)
    assert out[0, 2] == out[2, 0] == pytest.approx(-0.5)
    assert np.allclose(np.diag(out), 1.0)


def test_espelha_triangulo_inferior():
    C = np.eye(3)
    C[1, 0] = 0.65
    out, conflitos = mirror_triangle(C)
    assert conflitos == []
    assert out[0, 1] == out[1, 0] == pytest.approx(0.65)


def test_nao_divide_pela_metade():
    """O defeito original, em uma linha."""
    C = np.eye(2)
    C[0, 1] = 0.8
    out, _ = mirror_triangle(C)
    assert out[0, 1] == pytest.approx(0.8)
    assert out[0, 1] != pytest.approx(0.4)


def test_valores_iguais_nos_dois_lados_nao_sao_conflito():
    C = np.eye(2)
    C[0, 1] = C[1, 0] = 0.3
    out, conflitos = mirror_triangle(C)
    assert conflitos == []
    assert out[0, 1] == pytest.approx(0.3)


def test_valores_diferentes_nos_dois_lados_sao_reportados():
    C = np.eye(2)
    C[0, 1] = 0.8
    C[1, 0] = 0.2
    out, conflitos = mirror_triangle(C)
    assert len(conflitos) == 1
    i, j, acima, abaixo = conflitos[0]
    assert (i, j) == (0, 1)
    assert acima == pytest.approx(0.8) and abaixo == pytest.approx(0.2)
    # documentado: prevalece o triangulo superior
    assert out[0, 1] == pytest.approx(0.8)


def test_matriz_vazia_continua_identidade():
    out, conflitos = mirror_triangle(np.eye(4))
    assert conflitos == []
    assert np.allclose(out, np.eye(4))


def test_resultado_e_sempre_simetrico_com_diagonal_um():
    rng = np.random.default_rng(0)
    for _ in range(20):
        C = np.eye(4)
        iu = np.triu_indices(4, 1)
        C[iu] = rng.uniform(-0.9, 0.9, size=len(iu[0]))
        out, _ = mirror_triangle(C)
        assert np.allclose(out, out.T)
        assert np.allclose(np.diag(out), 1.0)


def test_matriz_nao_quadrada_e_recusada():
    with pytest.raises(ValueError):
        mirror_triangle(np.zeros((2, 3)))


# --- nivel de interface ---------------------------------------------------


def _rodar_com_correlacao(rho: float, onde: str, iteracoes: int = 30_000):
    """Monta x,y ~ N(0,1), injeta a matriz como o editor faria, e roda.

    O `st.data_editor` nao e acessivel pelo AppTest nesta versao do
    Streamlit, entao escrevemos direto em `st.session_state["corr_df"]` —
    que e exatamente o que o editor faz. O caminho de codigo exercitado
    (espelhamento + Iman-Conover) e o mesmo.
    """
    at = novo_app()
    at.number_input[0].set_value(iteracoes).run()
    add_var(at, 1, "x", "normal", {"mu": 0, "sigma": 1})
    add_var(at, 2, "y", "normal", {"mu": 0, "sigma": 1})

    nomes = ["x", "y"]
    df = pd.DataFrame(np.eye(2), index=nomes, columns=nomes)
    if onde == "superior":
        df.iloc[0, 1] = rho
    else:
        df.iloc[1, 0] = rho
    at.session_state["corr_df"] = df
    at.session_state["usar_corr"] = True
    at.run()

    campo_formula(at).set_value("x + y").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


@pytest.mark.parametrize("onde", ["superior", "inferior"])
def test_correlacao_de_um_triangulo_so_e_respeitada_na_simulacao(onde):
    """Var(x+y) = 2 + 2*rho. Se o app dividir rho por 2, o desvio denuncia."""
    rho = 0.8
    at = _rodar_com_correlacao(rho, onde)
    desvio = num(metrica(at, "Desvio-padrao"))
    esperado_certo = np.sqrt(2 + 2 * rho)
    esperado_metade = np.sqrt(2 + 2 * (rho / 2))
    assert abs(desvio - esperado_certo) < abs(desvio - esperado_metade), (
        f"desvio {desvio:.4f} esta mais perto de rho={rho/2} "
        f"({esperado_metade:.4f}) que de rho={rho} ({esperado_certo:.4f})"
    )
    assert abs(desvio - esperado_certo) < 0.05


def test_correlacao_negativa_de_um_triangulo_so():
    at = _rodar_com_correlacao(-0.6, "superior")
    desvio = num(metrica(at, "Desvio-padrao"))
    assert abs(desvio - np.sqrt(2 + 2 * (-0.6))) < 0.06


def test_tabela_de_correlacao_obtida_aparece_nos_resultados():
    """Melhoria: o usuario precisa poder conferir o que saiu, nao so o que pediu."""
    at = _rodar_com_correlacao(0.7, "superior", iteracoes=20_000)
    tabelas = [d.value for d in at.dataframe]
    achou = any(
        isinstance(t, pd.DataFrame) and "pedida" in t.columns and "obtida" in t.columns
        for t in tabelas
    )
    assert achou, "faltou a tabela comparando correlacao pedida e obtida"


def test_conflito_entre_triangulos_gera_aviso_na_tela():
    at = novo_app()
    add_var(at, 1, "x", "normal", {"mu": 0, "sigma": 1})
    add_var(at, 2, "y", "normal", {"mu": 0, "sigma": 1})
    nomes = ["x", "y"]
    df = pd.DataFrame(np.eye(2), index=nomes, columns=nomes)
    df.iloc[0, 1] = 0.8
    df.iloc[1, 0] = 0.2
    at.session_state["corr_df"] = df
    at.session_state["usar_corr"] = True
    at.run()
    avisos = " ".join(w.value for w in at.warning)
    assert "conflitantes" in avisos.lower()


# ===========================================================================
# DEFEITO 2 - resultados do ajuste somem ao interagir
# ===========================================================================


def test_resultado_do_ajuste_sobrevive_a_interacao():
    """O seletor 'Inspecionar ajuste' faz parte do proprio bloco de
    resultados. Se o bloco depender apenas do retorno de `st.button`, mexer
    no seletor apaga a tabela e o Q-Q plot que ele deveria controlar."""
    at = novo_app()
    dados = st_.norm(loc=50, scale=8).rvs(200, random_state=7)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in dados)).run()
    botao(at, "Ajustar").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]

    tabelas_antes = len(at.dataframe)
    assert tabelas_antes > 0, "o ajuste deveria produzir uma tabela"

    sel = next(s for s in at.selectbox if s.label == "Inspecionar ajuste")
    sel.set_value(list(sel.options)[1]).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.dataframe) >= tabelas_antes, (
        "a tabela sumiu ao trocar a distribuicao inspecionada"
    )


def test_ajuste_persiste_ao_mexer_na_barra_lateral():
    at = novo_app()
    dados = st_.norm(loc=10, scale=2).rvs(150, random_state=3)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in dados)).run()
    botao(at, "Ajustar").click().run()
    tabelas_antes = len(at.dataframe)

    at.number_input[1].set_value(999).run()  # semente, controle nao relacionado
    assert len(at.dataframe) >= tabelas_antes, (
        "o resultado do ajuste sumiu ao mexer em um controle nao relacionado"
    )


def test_ajuste_avisa_quando_os_dados_mudam():
    """Mostrar um ajuste velho ao lado de dados novos seria pior que apagar."""
    at = novo_app()
    dados = st_.norm(loc=10, scale=2).rvs(150, random_state=3)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in dados)).run()
    botao(at, "Ajustar").click().run()

    outros = st_.norm(loc=99, scale=1).rvs(150, random_state=4)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in outros)).run()
    infos = " ".join(i.value for i in at.info)
    assert "dados mudaram" in infos.lower()


# ===========================================================================
# DEFEITO 3 - rotulos duplicados
# ===========================================================================


def test_rotulos_duplicados_sao_sinalizados_ao_usuario():
    at = novo_app()
    add_var(at, 1, "custo", "normal", {"mu": 100, "sigma": 10})
    add_var(at, 2, "custo", "normal", {"mu": 50, "sigma": 5})
    assert not at.exception, [str(e.value) for e in at.exception]
    msgs = " ".join([e.value for e in at.error] + [w.value for w in at.warning])
    assert "repetido" in msgs.lower() or "duplicad" in msgs.lower()


def test_rotulos_duplicados_nao_quebram_a_aba_de_correlacao():
    """Antes: StreamlitAPIException nao tratada derrubava a pagina inteira."""
    at = novo_app()
    add_var(at, 1, "custo", "normal", {"mu": 100, "sigma": 10})
    add_var(at, 2, "custo", "normal", {"mu": 50, "sigma": 5})
    c = [x for x in at.checkbox if "Aplicar correlacao" in x.label]
    if c:
        c[0].set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_rotulos_duplicados_bloqueiam_a_simulacao():
    at = novo_app()
    add_var(at, 1, "custo", "normal", {"mu": 100, "sigma": 10})
    add_var(at, 2, "custo", "normal", {"mu": 50, "sigma": 5})
    assert not any("Rodar simulacao" in b.label for b in at.button), (
        "o botao de rodar nao deveria estar disponivel com rotulos ambiguos"
    )


# ===========================================================================
# DEFEITO 4 - achieved_spearman com exatamente duas variaveis
# ===========================================================================
#
# Encontrado ao implementar a tabela "pedida vs. obtida": scipy.spearmanr
# devolve uma matriz para k >= 3 e um ESCALAR para k == 2. O codigo repassava
# o escalar por atleast_2d e produzia uma matriz 1x1. Passou despercebido
# porque todos os testes anteriores usavam tres variaveis.


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_achieved_spearman_sempre_devolve_matriz_k_por_k(k):
    from mcrisk.correlation import achieved_spearman

    rng = np.random.default_rng(k)
    X = rng.normal(size=(3000, k))
    m = achieved_spearman(X)
    assert m.shape == (k, k), f"k={k} devolveu {m.shape}"
    assert np.allclose(np.diag(m), 1.0, atol=1e-9)
    assert np.allclose(m, m.T)


def test_achieved_spearman_com_duas_variaveis_bate_com_o_scipy():
    from mcrisk.correlation import achieved_spearman

    rng = np.random.default_rng(7)
    a = rng.normal(size=4000)
    b = 0.9 * a + 0.44 * rng.normal(size=4000)
    X = np.column_stack([a, b])
    esperado = st_.spearmanr(a, b).statistic
    assert achieved_spearman(X)[0, 1] == pytest.approx(esperado, abs=1e-12)


def test_correlation_error_com_duas_variaveis():
    from mcrisk.correlation import correlation_error

    rng = np.random.default_rng(11)
    a = rng.normal(size=4000)
    b = 0.9 * a + 0.44 * rng.normal(size=4000)
    X = np.column_stack([a, b])
    alvo = np.array([[1.0, 0.8], [0.8, 1.0]])
    err_max, err_med, diff = correlation_error(X, alvo)
    assert diff.shape == (2, 2)
    assert err_max == pytest.approx(err_med)  # so ha um par fora da diagonal


def test_renomear_resolve_e_libera_a_simulacao():
    at = novo_app()
    add_var(at, 1, "custo", "normal", {"mu": 100, "sigma": 10})
    add_var(at, 2, "custo", "normal", {"mu": 50, "sigma": 5})
    at.text_input(key="lab2").set_value("custo_b").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    campo_formula(at).set_value("custo + custo_b").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert abs(num(metrica(at, "Media")) - 150.0) < 1.0
