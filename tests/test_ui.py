"""
Testes funcionais da INTERFACE do dashboard.

Diferente dos demais arquivos em tests/, que exercitam o motor diretamente,
este dirige a interface Streamlit de ponta a ponta com
`streamlit.testing.v1.AppTest`: clica nos botoes, preenche os campos, roda a
simulacao e le os numeros que o usuario efetivamente veria na tela.

Onde ha solucao analitica conhecida, o valor esperado NAO vem da simulacao.

Foi este arquivo que revelou o bug de IDs duplicados corrigido em app.py
(ver `test_duas_variaveis_identicas_nao_derrubam_o_app`): carregar o app
vazio passava, mas usa-lo de verdade quebrava.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as st_
from streamlit.testing.v1 import AppTest

TIMEOUT = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def novo_app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=TIMEOUT).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def add_var(at: AppTest, vid: int, label: str, dist: str, params: dict) -> AppTest:
    """Adiciona uma variavel pela interface, como um usuario faria."""
    at.button[0].click().run()  # "Adicionar variavel"
    at.text_input(key=f"lab{vid}").set_value(label).run()
    at.selectbox(key=f"dist{vid}").set_value(dist).run()
    for nome, valor in params.items():
        at.number_input(key=f"p{vid}_{nome}").set_value(float(valor)).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def campo_formula(at: AppTest):
    return next(t for t in at.text_input if t.label == "Expressao")


def botao(at: AppTest, trecho: str):
    return next(b for b in at.button if trecho in b.label)


def metrica(at: AppTest, rotulo: str) -> str:
    return next(m for m in at.metric if m.label == rotulo).value


def num(txt: str) -> float:
    """Converte o texto exibido numa metrica de volta para float."""
    return float(txt.replace(",", ""))


def rodar_soma_de_normais(iteracoes: int = 50_000, metodo: str = "lhs") -> AppTest:
    """X~N(10,2), Y~N(5,1), saida x+y. Analitico: N(15, sqrt(5))."""
    at = novo_app()
    at.number_input[0].set_value(iteracoes).run()  # Iteracoes
    at.selectbox[0].set_value(metodo).run()  # Metodo de amostragem
    add_var(at, 1, "x", "normal", {"mu": 10, "sigma": 2})
    add_var(at, 2, "y", "normal", {"mu": 5, "sigma": 1})
    campo_formula(at).set_value("x + y").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


# ---------------------------------------------------------------------------
# 1. Estado inicial
# ---------------------------------------------------------------------------


def test_app_abre_sem_erro_e_sem_numeros_pre_preenchidos():
    """A promessa do README: nada vem preenchido."""
    at = novo_app()
    assert at.title[0].value.startswith("Simulacao de Monte Carlo")
    assert len(at.tabs) == 6
    # Nenhuma variavel definida => nenhuma metrica de resultado na tela
    assert len(at.metric) == 0
    # E o unico botao disponivel e o de adicionar variavel
    assert len(at.button) == 1 and "Adicionar variavel" in at.button[0].label


def test_mensagem_orienta_quando_nao_ha_variaveis():
    at = novo_app()
    textos = " ".join(i.value for i in at.info)
    assert "Nenhuma variavel definida" in textos
    assert "ao menos uma variavel valida" in textos


# ---------------------------------------------------------------------------
# 2. Fluxo principal e validacao analitica
# ---------------------------------------------------------------------------


def test_fluxo_completo_produz_resultado_correto():
    """O numero que aparece na tela bate com a solucao fechada?"""
    at = rodar_soma_de_normais()
    assert abs(num(metrica(at, "Media")) - 15.0) < 0.05
    assert abs(num(metrica(at, "Desvio-padrao")) - np.sqrt(5.0)) < 0.05


@pytest.mark.parametrize("p,rotulo", [(5, "P5"), (50, "Mediana (P50)"), (95, "P95")])
def test_percentis_exibidos_batem_com_a_teoria(p, rotulo):
    at = rodar_soma_de_normais()
    teorico = st_.norm.ppf(p / 100, loc=15.0, scale=np.sqrt(5.0))
    assert abs(num(metrica(at, rotulo)) - teorico) < 0.08


def test_metricas_secundarias_sao_coerentes():
    at = rodar_soma_de_normais()
    assert num(metrica(at, "Minimo")) < num(metrica(at, "Media"))
    assert num(metrica(at, "Maximo")) > num(metrica(at, "Media"))
    # Normal: assimetria e curtose de excesso proximas de zero
    assert abs(num(metrica(at, "Assimetria"))) < 0.1
    assert abs(num(metrica(at, "Curtose (excesso)"))) < 0.15
    # CV = desvio/media = sqrt(5)/15
    assert abs(num(metrica(at, "Coef. de variacao")) - np.sqrt(5) / 15) < 0.01


def test_var_e_cvar_aparecem_e_cvar_e_mais_extremo():
    at = rodar_soma_de_normais()
    var = num(metrica(at, "VaR 95%"))
    cvar = num(metrica(at, "CVaR 95%"))
    assert cvar < var, "CVaR deve ser mais extremo que o VaR na cauda esquerda"
    # VaR 95% = percentil 5 da N(15, sqrt(5))
    assert abs(var - st_.norm.ppf(0.05, 15.0, np.sqrt(5.0))) < 0.08


def test_probabilidade_de_limiar_e_exibida_e_complementar():
    at = rodar_soma_de_normais()
    abaixo = float(metrica(at, "P(X ≤ 13.49)").rstrip("%"))
    acima = float(metrica(at, "P(X > 13.49)").rstrip("%"))
    assert abs(abaixo + acima - 100.0) < 0.01


def test_semente_torna_o_resultado_reproduzivel():
    a = rodar_soma_de_normais(iteracoes=5000)
    b = rodar_soma_de_normais(iteracoes=5000)
    assert metrica(a, "Media") == metrica(b, "Media")
    assert metrica(a, "P95") == metrica(b, "P95")


def test_monte_carlo_simples_tambem_funciona():
    at = rodar_soma_de_normais(iteracoes=50_000, metodo="mc")
    assert abs(num(metrica(at, "Media")) - 15.0) < 0.1


# ---------------------------------------------------------------------------
# 3. Avisos honestos aparecem na tela
# ---------------------------------------------------------------------------


def test_aviso_de_erro_padrao_invalido_sob_lhs():
    """A ressalva metodologica precisa chegar ao usuario, nao so ao README."""
    at = rodar_soma_de_normais(metodo="lhs")
    avisos = " ".join(w.value for w in at.warning)
    assert "NAO" in avisos and "Latin Hypercube" in avisos


def test_sem_aviso_de_lhs_quando_o_metodo_e_monte_carlo():
    at = rodar_soma_de_normais(metodo="mc")
    avisos = " ".join(w.value for w in at.warning)
    assert "Latin Hypercube" not in avisos


def test_nota_sobre_independencia_aparece_nos_resultados():
    at = rodar_soma_de_normais(metodo="lhs")
    infos = " ".join(i.value for i in at.info)
    assert "NAO sao independentes" in infos


# ---------------------------------------------------------------------------
# 4. Sensibilidade na interface
# ---------------------------------------------------------------------------


def test_tornado_ordena_a_variavel_dominante_em_primeiro():
    at = novo_app()
    at.number_input[0].set_value(20_000).run()
    add_var(at, 1, "a", "normal", {"mu": 0, "sigma": 1})
    add_var(at, 2, "b", "normal", {"mu": 0, "sigma": 1})
    campo_formula(at).set_value("10*a + b").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # Localiza por COLUNA, nao por posicao. "variavel" sozinha deixou de
    # identificar: as tabelas de cenario e de contribuicao para a variancia
    # tambem tem essa coluna. Quem identifica a de sensibilidade e o par
    # (variavel, spearman).
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns")
               and {"variavel", "spearman"} <= set(d.value.columns)]
    assert tabelas, "tabela de sensibilidade nao encontrada na tela"
    df = tabelas[-1]
    assert list(df["variavel"])[0] == "a"
    assert abs(df["spearman"].iloc[0]) > abs(df["spearman"].iloc[1])


def test_modelo_nao_monotono_dispara_aviso_de_r2_na_tela():
    at = novo_app()
    at.number_input[0].set_value(20_000).run()
    add_var(at, 1, "a", "normal", {"mu": 0, "sigma": 1})
    add_var(at, 2, "b", "normal", {"mu": 0, "sigma": 1})
    campo_formula(at).set_value("a * b").run()
    botao(at, "Rodar simulacao").click().run()
    avisos = " ".join(w.value for w in at.warning)
    assert "Sobol" in avisos


# ---------------------------------------------------------------------------
# 5. Validacao de entradas e mensagens de erro
# ---------------------------------------------------------------------------


def test_formula_invalida_e_bloqueada_com_mensagem():
    at = novo_app()
    add_var(at, 1, "x", "normal", {"mu": 0, "sigma": 1})
    campo_formula(at).set_value("x + inexistente").run()
    erros = " ".join(e.value for e in at.error)
    assert "inexistente" in erros
    assert botao(at, "Rodar simulacao").disabled


@pytest.mark.parametrize(
    "ataque",
    [
        "__import__('os').system('id')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "[c for c in range(10)]",
        "x.__class__",
    ],
)
def test_tentativa_de_injecao_de_codigo_e_bloqueada_na_interface(ataque):
    """O bloqueio precisa acontecer na UI, com mensagem, sem derrubar o app."""
    at = novo_app()
    add_var(at, 1, "x", "normal", {"mu": 0, "sigma": 1})
    campo_formula(at).set_value(ataque).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.error, "a interface deveria exibir uma mensagem de erro"
    assert botao(at, "Rodar simulacao").disabled, "o botao deveria ficar bloqueado"


def test_parametro_invalido_mostra_erro_e_nao_quebra():
    at = novo_app()
    at.button[0].click().run()
    at.text_input(key="lab1").set_value("x").run()
    at.selectbox(key="dist1").set_value("normal").run()
    at.number_input(key="p1_mu").set_value(0.0).run()
    at.number_input(key="p1_sigma").set_value(-1.0).run()  # invalido
    assert not at.exception
    erros = " ".join(e.value for e in at.error)
    assert "sigma deve ser > 0" in erros


def test_triangular_com_ordem_invalida_e_recusada():
    at = novo_app()
    at.button[0].click().run()
    at.text_input(key="lab1").set_value("d").run()
    at.selectbox(key="dist1").set_value("triangular").run()
    for k, v in (("minimo", 10.0), ("moda", 1.0), ("maximo", 5.0)):
        at.number_input(key=f"p1_{k}").set_value(v).run()
    erros = " ".join(e.value for e in at.error)
    assert "minimo <= moda <= maximo" in erros


def test_duas_variaveis_identicas_nao_derrubam_o_app():
    """Regressao: bug encontrado por estes testes.

    Duas variaveis com a MESMA distribuicao e os MESMOS parametros geram
    graficos de previa identicos. Sem uma `key` explicita, o Streamlit
    derivava o mesmo ID automatico para os dois elementos e levantava
    StreamlitDuplicateElementId, derrubando a pagina inteira. E um cenario
    banal (dois custos iguais, duas atividades iguais), nao um caso de borda.
    """
    at = novo_app()
    add_var(at, 1, "custo_a", "normal", {"mu": 100, "sigma": 10})
    add_var(at, 2, "custo_b", "normal", {"mu": 100, "sigma": 10})
    assert not at.exception, [str(e.value) for e in at.exception]
    campo_formula(at).set_value("custo_a + custo_b").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert abs(num(metrica(at, "Media")) - 200.0) < 1.0


def test_tres_variaveis_identicas_tambem_funcionam():
    at = novo_app()
    for i in (1, 2, 3):
        add_var(at, i, f"item{i}", "pert", {"minimo": 1, "moda": 2, "maximo": 5, "lam": 4})
    assert not at.exception, [str(e.value) for e in at.exception]


def test_botao_de_rodar_fica_desabilitado_sem_formula():
    at = novo_app()
    add_var(at, 1, "x", "normal", {"mu": 0, "sigma": 1})
    assert botao(at, "Rodar simulacao").disabled


# ---------------------------------------------------------------------------
# 6. Distribuicoes na interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist,params,media_teorica",
    [
        ("pert", {"minimo": 1, "moda": 4, "maximo": 10, "lam": 4}, (1 + 16 + 10) / 6),
        ("uniform", {"a": 0, "b": 10}, 5.0),
        ("exponential", {"escala": 4, "loc": 0}, 4.0),
        ("poisson", {"lam": 3}, 3.0),
        ("lognormal_real", {"media": 100, "desvio": 25}, 100.0),
    ],
)
def test_distribuicoes_pela_interface_batem_com_a_media_teorica(
    dist, params, media_teorica
):
    at = novo_app()
    at.number_input[0].set_value(50_000).run()
    add_var(at, 1, "v", dist, params)
    campo_formula(at).set_value("v").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    obtida = num(metrica(at, "Media"))
    assert abs(obtida - media_teorica) / max(abs(media_teorica), 1) < 0.02, (
        f"{dist}: media exibida {obtida} vs teorica {media_teorica}"
    )


def test_discreta_customizada_respeita_as_probabilidades():
    at = novo_app()
    at.number_input[0].set_value(50_000).run()
    at.button[0].click().run()
    at.text_input(key="lab1").set_value("evento").run()
    at.selectbox(key="dist1").set_value("discrete_custom").run()
    at.text_input(key="val1").set_value("0, 100").run()
    at.text_input(key="prb1").set_value("0.8, 0.2").run()
    campo_formula(at).set_value("evento").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # media teorica = 0.8*0 + 0.2*100 = 20
    assert abs(num(metrica(at, "Media")) - 20.0) < 0.5


def test_probabilidades_que_nao_somam_um_geram_aviso():
    at = novo_app()
    at.button[0].click().run()
    at.text_input(key="lab1").set_value("evento").run()
    at.selectbox(key="dist1").set_value("discrete_custom").run()
    at.text_input(key="val1").set_value("0, 100").run()
    at.text_input(key="prb1").set_value("0.8, 0.5").run()
    avisos = " ".join(w.value for w in at.warning)
    assert "normalizadas" in avisos


def test_empirica_avisa_que_nao_extrapola():
    at = novo_app()
    at.button[0].click().run()
    at.text_input(key="lab1").set_value("hist").run()
    at.selectbox(key="dist1").set_value("empirical").run()
    at.text_area(key="dat1").set_value("1,2,3,4,5,6,7,8,9,10").run()
    avisos = " ".join(w.value for w in at.warning)
    assert "NUNCA gera valores fora do" in avisos


# ---------------------------------------------------------------------------
# 7. Correlacao pela interface
# ---------------------------------------------------------------------------


def test_aba_de_correlacao_aparece_com_duas_variaveis():
    at = novo_app()
    add_var(at, 1, "x", "normal", {"mu": 10, "sigma": 2})
    add_var(at, 2, "y", "normal", {"mu": 5, "sigma": 1})
    # com 2 variaveis validas surge o checkbox de aplicar correlacao
    assert any("Aplicar correlacao" in c.label for c in at.checkbox)
    avisos = " ".join(w.value for w in at.warning)
    assert "correlacao tambem nao e causalidade" in avisos.lower()


def test_uma_variavel_so_nao_oferece_correlacao():
    at = novo_app()
    add_var(at, 1, "x", "normal", {"mu": 0, "sigma": 1})
    infos = " ".join(i.value for i in at.info)
    assert "ao menos duas variaveis validas" in infos


# ---------------------------------------------------------------------------
# 8. Ajuste a dados pela interface
# ---------------------------------------------------------------------------


def test_ajuste_a_dados_roda_e_escolhe_a_familia_correta():
    at = novo_app()
    dados = st_.norm(loc=50, scale=8).rvs(400, random_state=7)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in dados)).run()
    assert not at.exception
    # com dados suficientes o app mostra as estatisticas descritivas
    assert any(m.label == "n" for m in at.metric)
    botao(at, "Ajustar").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns")
               and {"distribuicao", "AIC"} <= set(d.value.columns)]
    assert tabelas, "tabela de ajuste nao encontrada na tela"
    assert "normal" in list(tabelas[0]["distribuicao"])[:3]


def test_ajuste_avisa_com_amostra_pequena():
    at = novo_app()
    at.text_area(key="fit_txt").set_value("1,2,3,4,5,6,7,8,9,10").run()
    avisos = " ".join(w.value for w in at.warning)
    assert "observacoes" in avisos and "exploratorio" in avisos


def test_ajuste_recusa_menos_de_cinco_observacoes():
    at = novo_app()
    at.text_area(key="fit_txt").set_value("1,2,3").run()
    infos = " ".join(i.value for i in at.info)
    assert "ao menos 5 observacoes" in infos


# ---------------------------------------------------------------------------
# 9. Exportacao
# ---------------------------------------------------------------------------


def test_os_tres_botoes_de_exportacao_aparecem_apos_simular():
    at = rodar_soma_de_normais(iteracoes=2000)
    rotulos = [d.label for d in at.get("download_button")]
    assert any("CSV" in r for r in rotulos)
    assert any("Excel" in r for r in rotulos)
    assert any("JSON" in r for r in rotulos)


# ---------------------------------------------------------------------------
# Interface das capacidades novas: copulas, correlacao obtida e cenarios
# ---------------------------------------------------------------------------


def _com_duas_variaveis(iteracoes: int = 20_000) -> AppTest:
    at = novo_app()
    at.number_input[0].set_value(iteracoes).run()
    add_var(at, 1, "a", "normal", {"mu": 0, "sigma": 1})
    add_var(at, 2, "b", "normal", {"mu": 0, "sigma": 1})
    campo_formula(at).set_value("a + b").run()
    return at


def _ligar_correlacao(at: AppTest, rho: float = 0.6) -> AppTest:
    # A caixa nao tem `key` (o app grava o retorno direto em session_state),
    # entao ela e localizada pelo rotulo. Localizar por indice quebraria a cada
    # caixa nova acrescentada acima dela.
    caixas = [c for c in at.checkbox if "correlacao" in (c.label or "").lower()]
    assert caixas, "caixa de aplicar correlacao nao encontrada"
    caixas[0].set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    df = at.session_state["corr_df"]
    df.iloc[0, 1] = rho
    df.iloc[1, 0] = rho
    at.session_state["corr_df"] = df
    at.run()
    return at


def test_seletor_de_dependencia_aparece_e_tem_iman_conover_como_padrao():
    at = _com_duas_variaveis()
    escolhas = [r for r in at.radio if r.key == "dependence"]
    assert escolhas, "o esquema de dependencia precisa estar visivel na aba de correlacao"
    assert escolhas[0].value == "iman_conover", (
        "mudar o padrao em silencio alteraria todo resultado ja publicado"
    )


def test_copula_t_mostra_o_coeficiente_de_dependencia_de_cauda():
    at = _com_duas_variaveis()
    _ligar_correlacao(at, 0.6)
    at.radio(key="dependence").set_value("t").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    textos = " ".join(i.value for i in at.info)
    assert "dependencia de cauda" in textos
    assert "zero" in textos.lower(), (
        "o contraste com a Gaussiana e a informacao que justifica a escolha"
    )


def test_copula_gaussiana_avisa_que_nao_ha_dependencia_de_cauda():
    at = _com_duas_variaveis()
    _ligar_correlacao(at, 0.6)
    at.radio(key="dependence").set_value("gaussian").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    legendas = " ".join(c.value for c in at.caption)
    assert "cauda ZERO" in legendas or "cauda zero" in legendas.lower()


def test_simulacao_com_copula_t_roda_pela_interface():
    at = _com_duas_variaveis(30_000)
    _ligar_correlacao(at, 0.6)
    at.radio(key="dependence").set_value("t").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    res = at.session_state["resultado"]
    assert res.spec.dependence == "t"
    assert res.n == 30_000


def test_tabela_de_correlacao_obtida_aparece_apos_simular():
    """O motor manda o usuario conferir esta tabela; ela precisa existir."""
    at = _com_duas_variaveis(30_000)
    _ligar_correlacao(at, 0.6)
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    titulos = " ".join(m.value for m in at.markdown)
    assert "Obtida na amostra" in titulos and "Pedida" in titulos


def test_correlacao_obtida_bate_com_a_pedida_na_tela():
    at = _com_duas_variaveis(40_000)
    _ligar_correlacao(at, 0.6)
    botao(at, "Rodar simulacao").click().run()
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "shape") and d.value.shape == (2, 2)]
    assert tabelas, "matrizes 2x2 nao encontradas na tela"
    obtida = tabelas[-1].to_numpy(dtype=float)
    assert obtida[0, 1] == pytest.approx(0.6, abs=0.03)
    assert obtida[0, 0] == pytest.approx(1.0)


def test_sem_correlacao_a_secao_explica_em_vez_de_sumir():
    at = _com_duas_variaveis()
    botao(at, "Rodar simulacao").click().run()
    legendas = " ".join(c.value for c in at.caption)
    assert "independentes" in legendas


def test_painel_de_cenario_condicional_recorta_a_amostra():
    at = _com_duas_variaveis(30_000)
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    legendas = " ".join(c.value for c in at.caption)
    assert "iteracoes" in legendas, "o painel condicional precisa reportar a contagem"


def test_cenario_de_estresse_roda_e_reporta_delta():
    at = _com_duas_variaveis(20_000)
    botao(at, "Rodar simulacao").click().run()
    at.number_input(key="est_a_mu").set_value(5.0).run()
    botao(at, "Rodar cenario de estresse").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns") and "delta" in d.value.columns]
    assert tabelas, "tabela de estresse nao apareceu"
    linha = tabelas[-1].set_index("metrica").loc["media"]
    assert float(linha["delta"]) == pytest.approx(5.0, abs=0.15), (
        "deslocar mu de 'a' em 5 tem de deslocar a media de a+b em 5"
    )


def test_estresse_avisa_que_o_cenario_nao_tem_probabilidade_do_modelo():
    at = _com_duas_variaveis(20_000)
    botao(at, "Rodar simulacao").click().run()
    at.number_input(key="est_a_mu").set_value(3.0).run()
    botao(at, "Rodar cenario de estresse").click().run()
    avisos = " ".join(w.value for w in at.warning)
    assert "probabilidade" in avisos


# ---------------------------------------------------------------------------
# Interface dos metodos acrescentados na paridade com @RISK e ModelRisk
# ---------------------------------------------------------------------------


def test_os_quatro_metodos_de_sensibilidade_estao_na_tela():
    at = _com_duas_variaveis(10_000)
    botao(at, "Rodar simulacao").click().run()
    escolhas = [r for r in at.radio if r.key == "sens_metodo"]
    assert escolhas, "o seletor de metodo de sensibilidade precisa existir"
    # `.options` traz os rotulos formatados, nao os valores; o contrato que
    # importa e que os quatro valores sejam selecionaveis.
    assert len(escolhas[0].options) == 4
    assert escolhas[0].value == "srrc", (
        "trocar o padrao mudaria em silencio o tornado de quem ja usa"
    )
    for valor in ("spearman", "condicional", "variancia", "srrc"):
        at.radio(key="sens_metodo").set_value(valor).run()
        assert not at.exception, [str(e.value) for e in at.exception]


def test_metodo_condicional_desenha_o_spider_e_reporta_swing():
    at = _com_duas_variaveis(10_000)
    botao(at, "Rodar simulacao").click().run()
    at.radio(key="sens_metodo").set_value("condicional").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns") and "swing" in d.value.columns]
    assert tabelas, "a tabela de swing por faixa nao apareceu"
    infos = " ".join(i.value for i in at.info)
    assert "NAO" in infos and "monotona" in infos


def test_contribuicao_para_variancia_reporta_o_que_sobra():
    """A parte nao explicada nao pertence a nenhuma entrada, e a tela precisa
    dizer isso -- senao o usuario le as fracoes como se somassem 100%."""
    at = _com_duas_variaveis(10_000)
    botao(at, "Rodar simulacao").click().run()
    at.radio(key="sens_metodo").set_value("variancia").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    rotulos = [m.label for m in at.metric]
    assert "Nao explicada" in rotulos and "Variancia explicada" in rotulos
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns")
               and "fracao_variancia" in d.value.columns]
    assert tabelas


def test_painel_de_significancia_de_cenario_aparece_e_ordena():
    at = _com_duas_variaveis(20_000)
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns")
               and "significancia" in d.value.columns]
    assert tabelas, "a tabela de significancia de cenario nao apareceu"
    assert "significativa" in tabelas[-1].columns


def test_painel_de_convergencia_reporta_criterio_e_nao_so_um_grafico():
    at = _com_duas_variaveis(20_000)
    botao(at, "Rodar simulacao").click().run()
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns")
               and "iteracoes_ate_convergir" in d.value.columns]
    assert tabelas, "a tabela de convergencia nao apareceu"
    assert {"tolerancia", "erro_relativo", "convergiu"} <= set(tabelas[-1].columns)


def test_estatisticas_menos_comuns_estao_disponiveis():
    at = rodar_soma_de_normais(iteracoes=20_000)
    rotulos = [m.label for m in at.metric]
    for esperado in ("Moda", "Desvio absoluto medio", "Amplitude",
                     "Semi-desvio (abaixo da media)", "Semi-variancia"):
        assert esperado in rotulos, f"falta a metrica {esperado}"
    assert any("Pearson" in r for r in rotulos), (
        "as duas convencoes de curtose precisam aparecer nomeadas"
    )


def test_semi_desvio_exibido_bate_com_a_teoria():
    """X~N(10,2), Y~N(5,1) => saida N(15, sqrt(5)); semi-desvio = sigma/sqrt(2)."""
    at = rodar_soma_de_normais(iteracoes=50_000)
    obtido = num(metrica(at, "Semi-desvio (abaixo da media)"))
    assert abs(obtido - np.sqrt(5.0) / np.sqrt(2.0)) < 0.05


def test_convencao_descendente_de_percentil_muda_os_numeros():
    at = rodar_soma_de_normais(iteracoes=20_000)
    def p5_exibido(a):
        tab = [d.value for d in a.dataframe
               if hasattr(d.value, "columns") and "percentil" in d.value.columns][0]
        return float(tab["valor"].iloc[1])
    asc = p5_exibido(at)
    at.toggle(key="pct_desc").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    desc = p5_exibido(at)
    assert desc > asc, "P5 descendente e a cauda de cima, nao a de baixo"
    assert abs(desc - st_.norm.ppf(0.95, 15.0, np.sqrt(5.0))) < 0.1


def test_as_seis_opcoes_de_dependencia_estao_no_seletor():
    at = _com_duas_variaveis()
    escolhas = [r for r in at.radio if r.key == "dependence"]
    assert len(escolhas[0].options) == 6
    # O seletor fica desabilitado ate a correlacao ser ligada -- copula sem
    # matriz seria especificacao invalida, e o motor a recusaria.
    _ligar_correlacao(at, 0.6)
    for valor in ("gaussian", "t", "clayton", "gumbel", "frank", "iman_conover"):
        at.radio(key="dependence").set_value(valor).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.session_state["dependence"] == valor


@pytest.mark.parametrize("fam", ["clayton", "gumbel", "frank"])
def test_arquimediana_mostra_theta_e_a_cauda_calibrados(fam):
    at = _com_duas_variaveis()
    _ligar_correlacao(at, 0.6)
    at.radio(key="dependence").set_value(fam).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    rotulos = [m.label for m in at.metric]
    assert "theta calibrado" in rotulos
    assert "lambda inferior" in rotulos and "lambda superior" in rotulos


def test_clayton_e_gumbel_colocam_a_cauda_em_lados_opostos_na_tela():
    """O contraste e a razao de as duas estarem no menu. Se a tela mostrasse o
    mesmo numero para as duas, a escolha seria decorativa."""
    def lambdas(fam):
        at = _com_duas_variaveis()
        _ligar_correlacao(at, 0.6)
        at.radio(key="dependence").set_value(fam).run()
        return (
            float(metrica(at, "lambda inferior")),
            float(metrica(at, "lambda superior")),
        )
    ci, cs = lambdas("clayton")
    gi, gs = lambdas("gumbel")
    assert ci > 0 and cs == 0.0
    assert gs > 0 and gi == 0.0


def test_simulacao_com_arquimediana_roda_pela_interface():
    at = _com_duas_variaveis(15_000)
    _ligar_correlacao(at, 0.6)
    at.radio(key="dependence").set_value("clayton").run()
    botao(at, "Rodar simulacao").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    res = at.session_state["resultado"]
    assert res.spec.dependence == "clayton"
    texto = " ".join(res.notes)
    assert "calibrada em theta" in texto


def test_comparacao_de_distribuicoes_oferece_os_quatro_formatos():
    at = _com_duas_variaveis(10_000)
    botao(at, "Rodar simulacao").click().run()
    escolhas = [r for r in at.radio if r.key == "cmp_tipo"]
    assert escolhas, "o seletor de visualizacao comparada nao apareceu"
    assert len(escolhas[0].options) == 4
    for tipo in ("acumulada", "box"):
        at.radio(key="cmp_tipo").set_value(tipo).run()
        assert not at.exception, [str(e.value) for e in at.exception]


def test_tendencia_com_uma_serie_so_orienta_em_vez_de_quebrar():
    at = _com_duas_variaveis(8_000)
    botao(at, "Rodar simulacao").click().run()
    at.radio(key="cmp_tipo").set_value("tendencia").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    infos = " ".join(i.value for i in at.info)
    assert "duas distribuicoes" in infos


def test_incerteza_de_parametro_esta_disponivel_apos_o_ajuste():
    at = novo_app()
    dados = st_.norm(loc=50, scale=8).rvs(200, random_state=3)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in dados)).run()
    botao(at, "Ajustar").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.button(key="btn_inc_param").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns") and "erro_padrao" in d.value.columns]
    assert tabelas, "a tabela de incerteza de parametro nao apareceu"
    assert {"IC_inf", "IC_sup", "erro_relativo_pct"} <= set(tabelas[-1].columns)


def test_media_de_modelos_esta_disponivel_e_usa_pesos_de_akaike():
    at = novo_app()
    dados = st_.lognorm(0.6, scale=np.exp(3)).rvs(150, random_state=4)
    at.text_area(key="fit_txt").set_value(", ".join(f"{v:.4f}" for v in dados)).run()
    botao(at, "Ajustar").click().run()
    at.button(key="btn_model_avg").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    tabelas = [d.value for d in at.dataframe
               if hasattr(d.value, "columns") and "peso_akaike" in d.value.columns]
    assert tabelas, "a tabela de pesos de Akaike nao apareceu"
    assert abs(float(tabelas[-1]["peso_akaike"].sum()) - 1.0) < 1e-6
