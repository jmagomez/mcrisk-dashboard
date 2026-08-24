"""
Dashboard de simulacao de Monte Carlo para analise quantitativa de risco.

Execute com:  streamlit run app.py

O app NAO vem com numeros pre-preenchidos. Todo valor exibido vem de algo
que voce digitou ou de um arquivo que voce carregou.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
from mcrisk import convergence as conv_mod
from mcrisk import copula as copula_mod
from mcrisk import correlation as corr_mod
from mcrisk import scenarios as scen_mod
from mcrisk import distributions as dists
from mcrisk import fitting, sensitivity, summary
from mcrisk.engine import SimulationSpec, Variable, run, run_replicates, to_dataframe
from mcrisk.formula import FUNC_HELP, sanitize_variable_name, validate_formula

st.set_page_config(
    page_title="Monte Carlo - Analise de Risco",
    page_icon="\U0001F4CA",
    layout="wide",
)

COR = {
    "principal": "#2563eb",
    "alerta": "#dc2626",
    "neutro": "#9ca3af",
    "ok": "#059669",
}


# ===========================================================================
# Helpers
# ===========================================================================


def parse_lista(texto: str) -> list[float]:
    """Converte texto livre (virgulas, espacos, quebras de linha) em floats."""
    if not texto:
        return []
    bruto = str(texto).replace("\n", ",").replace(";", ",").replace("\t", ",")
    out: list[float] = []
    for pedaco in bruto.split(","):
        p = pedaco.strip().replace(" ", "")
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def construir_variavel(v: dict) -> Variable:
    return Variable(
        name=v["name"],
        label=v["label"] or v["name"],
        dist_key=v["dist"],
        params={k: x for k, x in v["params"].items() if x is not None},
        values=parse_lista(v["values"]) if v["dist"] == "discrete_custom" else None,
        probs=parse_lista(v["probs"]) if v["dist"] == "discrete_custom" else None,
        data=parse_lista(v["data"]) if v["dist"] == "empirical" else None,
    )


def variaveis_validas() -> list[Variable]:
    """Lista de Variable prontas, apenas as que passam na validacao."""
    out = []
    for v in st.session_state["vars"]:
        if not v["label"]:
            continue
        var = construir_variavel(v)
        if not var.validate():
            out.append(var)
    return out


def rotulos_duplicados(vv: list[Variable]) -> list[str]:
    """Rotulos repetidos entre as variaveis validas.

    Rotulos iguais viram o MESMO nome na formula, o que torna a expressao
    ambigua, e quebram a grade de correlacao (que usa os rotulos como
    cabecalho de coluna). Melhor barrar cedo, com mensagem clara.
    """
    nomes = [x.label for x in vv]
    return sorted({n for n in nomes if nomes.count(n) > 1})


def previa_distribuicao(var: Variable, chave: str, altura: int = 200) -> None:
    """Histograma rapido da marginal, para conferencia visual.

    `chave` precisa ser unica por variavel: duas variaveis com a MESMA
    distribuicao e os MESMOS parametros geram graficos identicos, e o
    Streamlit derivaria o mesmo ID automatico para ambos, o que levanta
    StreamlitDuplicateElementId e derruba a pagina.
    """
    try:
        u = (np.arange(1, 5001) - 0.5) / 5000.0
        x = var.ppf(u)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
    except Exception as e:
        st.warning(f"Nao foi possivel gerar a previa: {e}")
        return

    fig = go.Figure(
        go.Histogram(x=x, nbinsx=60, marker_color=COR["principal"], opacity=0.85)
    )
    fig.update_layout(
        height=altura,
        margin=dict(l=0, r=0, t=6, b=0),
        showlegend=False,
        yaxis_title=None,
        xaxis_title=None,
        bargap=0.02,
    )
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch", key=f"previa_graf_{chave}")
    with c2:
        linhas = {
            "media": float(np.mean(x)),
            "desvio": float(np.std(x, ddof=1)),
            "P5": float(np.percentile(x, 5)),
            "P50": float(np.percentile(x, 50)),
            "P95": float(np.percentile(x, 95)),
        }
        st.dataframe(
            pd.DataFrame({"estatistica": linhas.keys(), "valor": linhas.values()}),
            hide_index=True,
            width="stretch",
            key=f"previa_tab_{chave}",
        )
        st.caption("Previa deterministica (quantis igualmente espacados), n=5.000.")


def fmt(x: float, casas: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:,.{casas}g}"


# ===========================================================================
# Estado
# ===========================================================================

st.session_state.setdefault("vars", [])
st.session_state.setdefault("formula", "")
st.session_state.setdefault("corr_df", None)
st.session_state.setdefault("usar_corr", False)
st.session_state.setdefault("resultado", None)
st.session_state.setdefault("replicas", None)
st.session_state.setdefault("ajuste", None)
st.session_state.setdefault("incerteza", None)
st.session_state.setdefault("ver_media_modelos", False)
st.session_state.setdefault("next_id", 1)


def novo_id() -> int:
    i = st.session_state["next_id"]
    st.session_state["next_id"] = i + 1
    return i


# ===========================================================================
# Barra lateral
# ===========================================================================

with st.sidebar:
    st.header("Configuracao")

    iteracoes = st.number_input(
        "Iteracoes",
        min_value=100,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
        help=(
            "O erro de amostragem cai com 1/sqrt(n): quadruplicar as iteracoes "
            "reduz o erro pela metade. Iteracoes NAO corrigem erros de premissa."
        ),
    )

    metodo = st.selectbox(
        "Metodo de amostragem",
        options=["lhs", "mc", "lhs_median"],
        format_func=lambda m: {
            "lhs": "Latin Hypercube (recomendado)",
            "mc": "Monte Carlo simples",
            "lhs_median": "Latin Hypercube (mediana do estrato)",
        }[m],
        help=(
            "LHS estratifica cada marginal e atinge a mesma precisao com menos "
            "iteracoes (McKay et al., 1979). Em troca, as iteracoes deixam de "
            "ser independentes."
        ),
    )

    seed = int(
        st.number_input(
            "Semente aleatoria",
            min_value=0,
            max_value=2**31 - 1,
            value=12345,
            help="Mesma semente + mesma especificacao = resultado identico.",
        )
    )

    st.divider()
    st.caption("**Erro de simulacao**")
    usar_replicas = st.checkbox(
        "Rodar replicacoes independentes",
        value=False,
        help=(
            "Sob LHS o erro padrao s/sqrt(n) NAO e valido, porque as iteracoes "
            "sao dependentes. Replicacoes independentes sao a forma correta de "
            "medir o erro. O custo de tempo e proporcional ao numero de replicas."
        ),
    )
    n_replicas = int(
        st.number_input(
            "Numero de replicacoes",
            min_value=2,
            max_value=100,
            value=10,
            disabled=not usar_replicas,
        )
    )

    st.divider()
    st.caption(
        "Reimplementacao aberta de tecnicas padrao de analise de risco. "
        "Sem afiliacao com a Lumivero/@RISK. Leia `LIMITATIONS.md` antes de "
        "usar em decisao real."
    )


# ===========================================================================
# Cabecalho
# ===========================================================================

st.title("Simulacao de Monte Carlo para Analise de Risco")
st.caption(
    "Substitui valores fixos por distribuicoes de probabilidade e propaga a "
    "incerteza ate o resultado. Metodologia e referencias na aba 6."
)

aba_vars, aba_corr, aba_modelo, aba_result, aba_ajuste, aba_metodo = st.tabs(
    [
        "1 · Variaveis",
        "2 · Correlacao",
        "3 · Modelo",
        "4 · Resultados",
        "5 · Ajuste a dados",
        "6 · Metodologia",
    ]
)


# ===========================================================================
# Aba 1 - variaveis de entrada
# ===========================================================================

with aba_vars:
    st.subheader("Defina as entradas incertas")
    st.markdown(
        "Cada variavel recebe um nome utilizavel na formula da aba 3. "
        "Nenhum valor e sugerido: os numeros precisam vir da sua evidencia "
        "(dados historicos, cotacoes, elicitacao com especialistas)."
    )

    if st.button("➕ Adicionar variavel", type="primary"):
        st.session_state["vars"].append(
            {
                "id": novo_id(),
                "label": "",
                "name": "",
                "dist": "pert",
                "params": {},
                "values": "",
                "probs": "",
                "data": "",
            }
        )
        st.session_state["corr_df"] = None
        st.rerun()

    if not st.session_state["vars"]:
        st.info(
            "Nenhuma variavel definida. Comece adicionando uma — por exemplo, o "
            "preco unitario, o custo de uma atividade ou a duracao de uma etapa."
        )

    opcoes = [(s.key, s.label) for s in dists.list_distributions()] + list(
        dists.SPECIAL_KINDS.items()
    )
    chaves = [k for k, _ in opcoes]
    rotulos = dict(opcoes)

    remover = None
    for idx, v in enumerate(st.session_state["vars"]):
        titulo = v["label"] or f"Variavel {idx + 1} (sem rotulo)"
        with st.expander(f"**{titulo}**", expanded=not v["label"]):
            c1, c2, c3 = st.columns([2, 3, 1])
            with c1:
                v["label"] = st.text_input(
                    "Rotulo",
                    value=v["label"],
                    key=f"lab{v['id']}",
                    placeholder="ex.: Preco unitario",
                )
                v["name"] = sanitize_variable_name(v["label"] or f"var{idx + 1}")
                st.caption(f"Nome na formula: `{v['name']}`")
            with c2:
                v["dist"] = st.selectbox(
                    "Distribuicao",
                    options=chaves,
                    index=chaves.index(v["dist"]) if v["dist"] in chaves else 0,
                    format_func=lambda k: rotulos[k],
                    key=f"dist{v['id']}",
                )
            with c3:
                st.write("")
                st.write("")
                if st.button("\U0001F5D1️", key=f"del{v['id']}", help="Remover variavel"):
                    remover = idx

            if v["dist"] == "discrete_custom":
                cc1, cc2 = st.columns(2)
                v["values"] = cc1.text_input(
                    "Valores (separados por virgula)",
                    value=v["values"],
                    key=f"val{v['id']}",
                    placeholder="ex.: 0, 1000, 5000",
                )
                v["probs"] = cc2.text_input(
                    "Probabilidades",
                    value=v["probs"],
                    key=f"prb{v['id']}",
                    placeholder="ex.: 0.7, 0.2, 0.1",
                )
                ps = parse_lista(v["probs"])
                if ps and abs(sum(ps) - 1.0) > 1e-9:
                    st.warning(
                        f"As probabilidades somam {sum(ps):.6g}, nao 1. Elas serao "
                        f"normalizadas, o que altera os valores efetivos em "
                        f"relacao ao que voce digitou.",
                        icon="⚠️",
                    )
            elif v["dist"] == "empirical":
                v["data"] = st.text_area(
                    "Serie historica (virgulas ou quebras de linha)",
                    value=v["data"],
                    key=f"dat{v['id']}",
                    height=90,
                )
                st.warning(
                    "A reamostragem empirica NUNCA gera valores fora do "
                    "minimo/maximo observados. Se o risco relevante esta na cauda, "
                    "este metodo o subestima por construcao. Considere ajustar uma "
                    "distribuicao parametrica na aba 5.",
                    icon="⚠️",
                )
            else:
                spec = dists.get(v["dist"])
                cols = st.columns(min(len(spec.params), 4))
                for i, p in enumerate(spec.params):
                    with cols[i % len(cols)]:
                        atual = v["params"].get(p.name, p.default)
                        v["params"][p.name] = st.number_input(
                            p.label,
                            value=float(atual) if atual is not None else None,
                            key=f"p{v['id']}_{p.name}",
                            help=p.help or None,
                            format="%.6g",
                            placeholder="obrigatorio",
                        )
                if spec.notes:
                    st.info(spec.notes, icon="ℹ️")
                if spec.reference:
                    st.caption(f"Referencia: {spec.reference}")

            if v["label"]:
                var = construir_variavel(v)
                errs = var.validate()
                if errs:
                    st.error("  \n".join(f"• {e}" for e in errs), icon="\U0001F6AB")
                else:
                    previa_distribuicao(var, chave=str(v["id"]))
            else:
                st.caption("Informe um rotulo para habilitar a previa.")

    if remover is not None:
        st.session_state["vars"].pop(remover)
        st.session_state["corr_df"] = None
        st.rerun()

    vv = variaveis_validas()
    dups = rotulos_duplicados(vv)
    if dups:
        st.error(
            "Rotulos repetidos: "
            + ", ".join(f"`{d}`" for d in dups)
            + ". Rotulos iguais viram o mesmo nome na formula, e a simulacao nao "
            "teria como saber a qual das variaveis voce se refere. Renomeie "
            "antes de continuar.",
            icon="\U0001F6AB",
        )
    elif vv:
        st.success(
            f"{len(vv)} variavel(is) pronta(s): "
            + ", ".join(f"`{x.name}`" for x in vv),
            icon="✅",
        )


# ===========================================================================
# Aba 2 - correlacao
# ===========================================================================

with aba_corr:
    st.subheader("Correlacao entre as entradas")
    vv = variaveis_validas()
    dups = rotulos_duplicados(vv)

    if dups:
        st.error(
            "Ha rotulos repetidos ("
            + ", ".join(f"`{d}`" for d in dups)
            + "). Renomeie as variaveis na aba 1 para poder especificar "
            "correlacoes.",
            icon="\U0001F6AB",
        )
    elif len(vv) < 2:
        st.info("Defina ao menos duas variaveis validas para especificar correlacoes.")
    else:
        st.markdown(
            "Informe a correlacao de **posto (Spearman)** desejada. O metodo de "
            "Iman-Conover (1982) reordena as amostras para atingir esse alvo "
            "**preservando as marginais exatamente**."
        )
        st.warning(
            "Tratar entradas correlacionadas como independentes e um dos erros "
            "mais caros em analise de risco: subestima a variancia da saida "
            "quando a correlacao e positiva e superestima quando e negativa. "
            "Correlacao tambem nao e causalidade — impor uma correlacao nao "
            "modela o mecanismo que a produz.",
            icon="⚠️",
        )

        nomes = [x.label for x in vv]
        k = len(nomes)
        anterior = st.session_state.get("corr_df")
        if (
            anterior is None
            or list(anterior.columns) != nomes
            or list(anterior.index) != nomes
        ):
            anterior = pd.DataFrame(np.eye(k), index=nomes, columns=nomes)

        st.session_state["usar_corr"] = st.checkbox(
            "Aplicar correlacao na simulacao", value=st.session_state["usar_corr"]
        )

        editado = st.data_editor(
            anterior,
            width="stretch",
            disabled=not st.session_state["usar_corr"],
            column_config={
                n: st.column_config.NumberColumn(
                    n, min_value=-1.0, max_value=1.0, step=0.05, format="%.3f"
                )
                for n in nomes
            },
        )
        st.session_state["corr_df"] = editado
        st.caption(
            "Preencha apenas o triangulo superior ou inferior — o lado preenchido "
            "e espelhado no outro. A diagonal e forcada em 1. Se voce preencher "
            "os dois lados com valores diferentes, o app avisa em vez de escolher "
            "por conta propria."
        )

        C, conflitos = corr_mod.mirror_triangle(editado.to_numpy(dtype=float))
        if conflitos:
            st.warning(
                "Valores conflitantes nos dois triangulos (sera usado o valor "
                "acima da diagonal): "
                + "; ".join(
                    f"{nomes[i]} x {nomes[j]}: {a:.3f} acima, {b:.3f} abaixo"
                    for i, j, a, b in conflitos
                ),
                icon="⚠️",
            )
        problemas = corr_mod.check_correlation_matrix(C)
        if problemas:
            st.error("  \n".join(f"• {p}" for p in problemas), icon="\U0001F6AB")
            reparada = corr_mod.nearest_psd_correlation(C)
            st.markdown("**Matriz positiva semidefinida mais proxima (Higham, 2002):**")
            st.dataframe(
                pd.DataFrame(np.round(reparada, 4), index=nomes, columns=nomes),
                width="stretch",
            )
            st.caption(
                "Se voce rodar a simulacao assim, esta matriz reparada sera usada "
                "e as correlacoes efetivas DIFERIRAO das que voce pediu."
            )
        elif st.session_state["usar_corr"]:
            st.success("Matriz valida e internamente consistente.", icon="✅")
            st.markdown("**Matriz efetivamente usada (apos o espelhamento):**")
            st.dataframe(
                pd.DataFrame(np.round(C, 4), index=nomes, columns=nomes),
                width="stretch",
            )

        st.markdown("---")
        st.markdown("**Como a dependencia e imposta**")
        esquemas = {
            "iman_conover": "Iman-Conover (posto) — padrao",
            "gaussian": "Copula Gaussiana",
            "t": "Copula t de Student",
            "clayton": "Copula Clayton (cauda inferior)",
            "gumbel": "Copula Gumbel (cauda superior)",
            "frank": "Copula Frank (sem cauda)",
        }
        dep = st.radio(
            "Esquema",
            list(esquemas),
            format_func=lambda k: esquemas[k],
            horizontal=False,
            disabled=not st.session_state["usar_corr"],
            key="dependence",
            help=(
                "Iman-Conover reordena as amostras: preserva as marginais "
                "exatamente e nao supoe forma de dependencia. As copulas dao "
                "forma declarada a dependencia — e so a t produz eventos "
                "extremos simultaneos."
            ),
        )
        if dep == "t":
            gl = st.slider(
                "Graus de liberdade da copula t",
                min_value=2.0, max_value=50.0, value=5.0, step=0.5,
                key="copula_df",
                help="Menos graus de liberdade = cauda conjunta mais pesada.",
            )
            if st.session_state["usar_corr"] and len(nomes) >= 2:
                rho_max = float(np.max(np.abs(C - np.eye(len(nomes)))))
                lam = copula_mod.tail_dependence_t(rho_max, gl)
                st.info(
                    f"Com o maior |rho| da sua matriz ({rho_max:.2f}) e {gl:g} graus "
                    f"de liberdade, o coeficiente de dependencia de cauda e "
                    f"**{lam:.3f}**: em cerca de {lam:.1%} das vezes em que uma "
                    f"variavel atinge um extremo, a outra tambem atinge. Sob copula "
                    f"Gaussiana esse numero e **zero** para qualquer correlacao menor "
                    f"que 1 — e essa e a diferenca entre as duas escolhas.",
                    icon="💡",
                )
        elif dep == "gaussian":
            st.caption(
                "Copula Gaussiana: dependencia de cauda ZERO para qualquer "
                "correlacao < 1. Extremos conjuntos ficam assintoticamente "
                "independentes — foi a critica central a modelagem de credito "
                "estruturado antes de 2008."
            )
        elif dep in copula_mod.ARQUIMEDIANAS:
            rho_medio = copula_mod.rho_medio_fora_da_diagonal(C)
            dispersao = copula_mod.dispersao_fora_da_diagonal(C)
            if rho_medio <= 0:
                st.error(
                    f"As copulas arquimedianas deste app so representam "
                    f"dependencia POSITIVA, e o rho medio da sua matriz e "
                    f"{rho_medio:.3f}. Use Gaussiana ou t.",
                    icon="\U0001F6AB",
                )
            else:
                try:
                    theta = copula_mod.theta_from_spearman(dep, rho_medio)
                except ValueError as exc:
                    st.error(str(exc), icon="\U0001F6AB")
                else:
                    lam_inf, lam_sup = copula_mod.tail_dependence_archimedean(
                        dep, theta
                    )
                    tau = copula_mod.tau_from_theta(dep, theta)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("theta calibrado", f"{theta:.4g}")
                    m2.metric("lambda inferior", f"{lam_inf:.4f}")
                    m3.metric("lambda superior", f"{lam_sup:.4f}")
                    st.caption(
                        f"Calibrado a partir do rho medio {rho_medio:.4f} "
                        f"(tau de Kendall = {tau:.4f}). Ao contrario da t, que "
                        f"e radialmente simetrica, estas familias colocam a "
                        f"dependencia em UM lado so — e o lado e a escolha "
                        f"que voce esta fazendo aqui."
                    )
                    if dispersao > 1e-9:
                        st.warning(
                            f"Esta familia e PERMUTAVEL: um unico parametro "
                            f"governa todos os pares. Seus rho variam em "
                            f"{dispersao:.3f} entre os pares, e essa "
                            f"heterogeneidade sera DESCARTADA — todos os pares "
                            f"receberao {rho_medio:.4f}. Se a diferenca entre os "
                            f"pares importa, use Gaussiana ou t.",
                            icon="⚠️",
                        )


# ===========================================================================
# Aba 3 - modelo e execucao
# ===========================================================================

with aba_modelo:
    st.subheader("Formula da saida")
    vv = variaveis_validas()
    dups = rotulos_duplicados(vv)

    if dups:
        st.error(
            "Renomeie os rotulos repetidos na aba 1 antes de rodar: "
            + ", ".join(f"`{d}`" for d in dups),
            icon="\U0001F6AB",
        )
        ok = False
    elif not vv:
        st.info("Defina ao menos uma variavel valida na aba 1.")
        ok = False
    else:
        st.markdown(
            "Escreva a expressao que combina as entradas no resultado de "
            "interesse (lucro, VPL, custo total, duracao do projeto)."
        )
        st.code(" · ".join(f"{x.name}" for x in vv), language=None)

        st.session_state["formula"] = st.text_input(
            "Expressao",
            value=st.session_state["formula"],
            placeholder="ex.: (preco - custo_unitario) * volume - custo_fixo",
        )

        with st.expander("Funcoes disponiveis"):
            st.table(
                pd.DataFrame(
                    {"funcao": FUNC_HELP.keys(), "descricao": FUNC_HELP.values()}
                )
            )
            st.caption(
                "A expressao e analisada com o modulo `ast` e apenas as "
                "construcoes acima sao aceitas. Nao ha `eval` cru: strings "
                "arbitrarias nao conseguem executar codigo."
            )

        nomes = [x.name for x in vv]
        if st.session_state["formula"]:
            ok, msg = validate_formula(st.session_state["formula"], nomes)
            (st.success if ok else st.error)(msg, icon="✅" if ok else "\U0001F6AB")
        else:
            ok = False

        st.divider()
        nome_saida = st.text_input("Nome do resultado", value="Resultado")

        col1, col2 = st.columns([1, 3])
        rodar = col1.button(
            "▶ Rodar simulacao",
            type="primary",
            disabled=not ok,
            width="stretch",
        )

        if rodar:
            C = None
            if st.session_state["usar_corr"] and st.session_state["corr_df"] is not None:
                Cm = st.session_state["corr_df"].to_numpy(dtype=float)
                if Cm.shape == (len(vv), len(vv)):
                    C, _ = corr_mod.mirror_triangle(Cm)

            spec = SimulationSpec(
                variables=vv,
                formula=st.session_state["formula"],
                iterations=int(iteracoes),
                method=metodo,
                seed=seed,
                correlation=C,
                dependence=(st.session_state.get("dependence", "iman_conover")
                            if C is not None else "iman_conover"),
                copula_df=float(st.session_state.get("copula_df", 5.0)),
            )
            try:
                with st.spinner("Simulando..."):
                    if usar_replicas:
                        res, outs = run_replicates(spec, replicates=n_replicas)
                        st.session_state["replicas"] = outs
                    else:
                        res = run(spec)
                        st.session_state["replicas"] = None
                st.session_state["resultado"] = res
                st.session_state["nome_saida"] = nome_saida
                st.success(
                    f"Simulacao concluida: {res.n:,} iteracoes. Veja a aba 4.",
                    icon="✅",
                )
            except Exception as e:
                st.error(f"Falha na simulacao: {e}", icon="\U0001F6AB")


# ===========================================================================
# Aba 4 - resultados
# ===========================================================================

with aba_result:
    res = st.session_state.get("resultado")
    if res is None:
        st.info("Nenhuma simulacao executada ainda. Configure e rode na aba 3.")
    else:
        nome_saida = st.session_state.get("nome_saida", "Resultado")
        y = res.output
        d = summary.describe(y)

        for nota in res.notes:
            st.info(nota, icon="ℹ️")

        st.subheader("Estatisticas do resultado")
        c = st.columns(5)
        c[0].metric("Media", fmt(d["media"]))
        c[1].metric("Desvio-padrao", fmt(d["desvio"]))
        c[2].metric("P5", fmt(d["P5"]))
        c[3].metric("Mediana (P50)", fmt(d["P50"]))
        c[4].metric("P95", fmt(d["P95"]))

        c = st.columns(5)
        c[0].metric("Minimo", fmt(d["minimo"]))
        c[1].metric("Maximo", fmt(d["maximo"]))
        c[2].metric("Assimetria", fmt(d["assimetria"], 3))
        c[3].metric("Curtose (excesso)", fmt(d["curtose_excesso"], 3))
        c[4].metric("Coef. de variacao", fmt(d["coef_variacao"], 3))

        with st.expander("Medidas menos comuns (dispersao assimetrica, moda, convencoes)"):
            e1, e2, e3 = st.columns(3)
            e1.metric("Moda", fmt(summary.mode(y)))
            e1.caption("A menos estavel das tres medidas centrais: depende do binning.")
            e2.metric("Desvio absoluto medio", fmt(summary.mean_absolute_deviation(y)))
            e2.caption("Nao eleva ao quadrado: da muito menos peso aos extremos.")
            e3.metric("Amplitude", fmt(summary.value_range(y)))
            e3.caption("Cresce com o numero de iteracoes; nao use para comparar.")
            f1, f2, f3 = st.columns(3)
            f1.metric("Semi-desvio (abaixo da media)", fmt(summary.semi_std(y)))
            f2.metric("Semi-variancia", fmt(summary.semi_variance(y)))
            f3.metric("Curtose (Pearson, normal = 3)", fmt(summary.kurtosis_pearson(y), 3))
            razao = (
                d["desvio"] / summary.mean_absolute_deviation(y)
                if summary.mean_absolute_deviation(y) > 0
                else float("nan")
            )
            st.caption(
                f"**Razao desvio/DAM = {fmt(razao, 3)}.** Para a normal ela vale "
                f"{np.sqrt(np.pi / 2):.4f}. Quanto mais acima disso, mais a "
                f"dispersao esta concentrada em poucas observacoes extremas — o "
                f"que e diagnostico de cauda pesada, e um aviso de que a media "
                f"e o desvio descrevem mal esta saida."
            )
            st.caption(
                "**Semi-desvio** mede so a dispersao ABAIXO da media. O desvio "
                "comum trata surpresa boa e ruim como equivalentes, o que quase "
                "nunca corresponde a preferencia de quem decide."
            )
            st.caption(
                "**Duas convencoes de curtose** existem e diferem por exatamente "
                "3. O resto deste app reporta a de excesso (normal = 0); a linha "
                "acima traz a de Pearson (normal = 3), que e a usada pelo @RISK. "
                "Comparar um numero com o outro faz uma normal parecer ter cauda "
                "pesada."
            )

        if abs(d["assimetria"]) > 0.5:
            st.warning(
                f"A saida e assimetrica (assimetria = {d['assimetria']:.2f}). "
                f"A media ({fmt(d['media'])}) nao representa bem o caso tipico; "
                f"a mediana ({fmt(d['P50'])}) e mais informativa, e as decisoes "
                f"deveriam olhar os percentis, nao o valor esperado.",
                icon="⚠️",
            )

        # ---------------- correlacao pedida vs. obtida ----------------
        if res.spec.correlation is not None and len(res.names) > 1:
            with st.expander("Correlacao pedida vs. obtida"):
                obtida = corr_mod.achieved_spearman(res.inputs)
                alvo = np.asarray(res.spec.correlation, dtype=float)
                linhas = []
                for i in range(len(res.labels)):
                    for j in range(i + 1, len(res.labels)):
                        linhas.append(
                            {
                                "par": f"{res.labels[i]} × {res.labels[j]}",
                                "pedida": alvo[i, j],
                                "obtida": obtida[i, j],
                                "diferenca": obtida[i, j] - alvo[i, j],
                            }
                        )
                st.dataframe(
                    pd.DataFrame(linhas), hide_index=True, width="stretch"
                )
                st.caption(
                    "Iman-Conover atinge a correlacao alvo de forma aproximada. "
                    "Diferencas na terceira casa sao normais; diferencas grandes "
                    "indicam que a matriz precisou ser reparada por nao ser "
                    "positiva semidefinida."
                )

        # ---------------- erro de simulacao ----------------
        st.subheader("Quanto disso e ruido de simulacao?")
        replicas = st.session_state.get("replicas")
        if replicas:
            medias = [float(np.mean(o[np.isfinite(o)])) for o in replicas]
            rs = summary.replicate_summary(medias)
            cc = st.columns(4)
            cc[0].metric("Replicacoes", rs["replicacoes"])
            cc[1].metric("Media das replicas", fmt(rs["media"]))
            cc[2].metric("Erro padrao (valido)", fmt(rs["erro_padrao"]))
            cc[3].metric(
                "IC 95% da media", f"[{fmt(rs['ic_inf'])}, {fmt(rs['ic_sup'])}]"
            )
            st.caption(
                "Erro estimado a partir de replicacoes independentes — valido "
                "tanto para Monte Carlo simples quanto para LHS."
            )
        else:
            se = summary.mc_standard_error(y)
            lo, hi = summary.mean_ci(y)
            cc = st.columns(3)
            cc[0].metric("Erro padrao s/√n", fmt(se))
            cc[1].metric("IC 95% da media", f"[{fmt(lo)}, {fmt(hi)}]")
            cc[2].metric(
                "n para ±1% da media",
                f"{summary.iterations_for_precision(y, abs(d['media']) * 0.01):,.0f}"
                if d["media"] != 0
                else "—",
            )
            if res.spec.method != "mc":
                st.warning(
                    "Estes numeros assumem iteracoes independentes, o que NAO "
                    "vale sob Latin Hypercube. Eles tendem a superestimar o erro "
                    "real da media. Para uma medida valida, ative as replicacoes "
                    "independentes na barra lateral.",
                    icon="⚠️",
                )

        # ---------------- graficos ----------------
        st.subheader("Distribuicao do resultado")
        yf = y[np.isfinite(y)]
        g1, g2 = st.columns(2)

        fig = go.Figure(
            go.Histogram(x=yf, nbinsx=80, marker_color=COR["principal"], opacity=0.85)
        )
        for p, cor in ((5, COR["alerta"]), (50, COR["neutro"]), (95, COR["ok"])):
            fig.add_vline(
                x=float(np.percentile(yf, p)),
                line_dash="dash",
                line_color=cor,
                annotation_text=f"P{p}",
            )
        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Histograma",
            bargap=0.02,
            showlegend=False,
        )
        g1.plotly_chart(fig, width="stretch")

        xs = np.sort(yf)
        fig2 = go.Figure(
            go.Scatter(
                x=xs,
                y=np.arange(1, xs.size + 1) / xs.size,
                mode="lines",
                line=dict(color=COR["principal"], width=2),
            )
        )
        fig2.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=30, b=0),
            title="Distribuicao acumulada (CDF)",
            yaxis_title="P(X ≤ x)",
            showlegend=False,
        )
        g2.plotly_chart(fig2, width="stretch")

        # ---------------- percentis e probabilidades ----------------
        p1, p2 = st.columns([3, 2])
        with p1:
            st.markdown("**Percentis com intervalo de confianca de 95%**")
            desc = st.toggle(
                "Convencao descendente (P5 = 5% da probabilidade ACIMA)",
                value=False,
                key="pct_desc",
                help=(
                    "Parte da literatura de risco usa a convencao oposta a do "
                    "NumPy. 'P5' pode significar o valor com 5% abaixo ou o "
                    "valor com 5% acima, e os dois numeros sao muito diferentes "
                    "numa cauda."
                ),
            )
            linhas = []
            for p in summary.DEFAULT_PERCENTILES:
                q = (100 - p) if desc else p
                lo, hi = summary.quantile_ci(yf, q / 100.0)
                linhas.append(
                    {
                        "percentil": f"P{p}" + ("↓" if desc else ""),
                        "valor": float(np.percentile(yf, q)),
                        "IC inf": lo,
                        "IC sup": hi,
                    }
                )
            st.dataframe(
                pd.DataFrame(linhas), hide_index=True, width="stretch"
            )
            st.caption(
                "IC nao parametrico via estatisticas de ordem (metodo binomial). "
                "Assume iteracoes independentes."
                + (
                    "  \nNa convencao **descendente**, P5 e o valor superado por "
                    "apenas 5% das iteracoes."
                    if desc
                    else ""
                )
            )

        with p2:
            st.markdown("**Probabilidade de nao atingir um limiar**")
            limiar = st.number_input(
                "Limiar", value=float(np.percentile(yf, 25)), format="%.6g"
            )
            pb = summary.prob_below(yf, limiar)
            st.metric(f"P(X ≤ {fmt(limiar)})", f"{pb * 100:.2f}%")
            st.metric(f"P(X > {fmt(limiar)})", f"{(1 - pb) * 100:.2f}%")

            st.markdown("**Medidas de cauda**")
            alfa = st.slider("Nivel alfa", 0.01, 0.25, 0.05, 0.01)
            lado = st.radio(
                "A perda esta na cauda",
                ["esquerda (saida = ganho)", "direita (saida = perda)"],
                horizontal=False,
            )
            baixo = lado.startswith("esquerda")
            st.metric(
                f"VaR {int((1 - alfa) * 100)}%",
                fmt(summary.value_at_risk(yf, alfa, baixo)),
            )
            st.metric(
                f"CVaR {int((1 - alfa) * 100)}%",
                fmt(summary.conditional_value_at_risk(yf, alfa, baixo)),
            )
            st.caption(
                "VaR nao e uma medida coerente de risco (nao e subaditivo) — "
                "Artzner et al. (1999). O CVaR e, e por isso deve ser lido junto."
            )

        # ---------------- convergencia ----------------
        with st.expander("Diagnostico de convergencia"):
            idx, med = summary.convergence_path(yf)
            figc = go.Figure(
                go.Scatter(x=idx, y=med, mode="lines", line=dict(color=COR["principal"]))
            )
            figc.add_hline(
                y=float(np.mean(yf)), line_dash="dot", line_color=COR["neutro"]
            )
            figc.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="iteracoes",
                yaxis_title="media acumulada",
            )
            st.plotly_chart(figc, width="stretch")
            st.caption(
                "Estabilizacao visual e condicao necessaria, nao suficiente: uma "
                "curva plana nao garante que a cauda foi bem amostrada. Sob LHS a "
                "ordem das iteracoes e arbitraria, entao leia o grafico apenas "
                "como diagnostico qualitativo."
            )

        # ---------------- sensibilidade ----------------
        st.subheader("Analise de sensibilidade")
        sres = sensitivity.analyze(res.inputs, res.output, res.labels)
        for w in sres.warnings:
            st.warning(w, icon="⚠️")

        st.caption(
            "Quatro metodos, quatro perguntas diferentes. Nao existe o "
            "\"melhor\": eles discordam quando o modelo tem estrutura que um "
            "deles nao enxerga, e a DISCORDANCIA e o achado."
        )
        criterio = st.radio(
            "Metodo",
            ["srrc", "spearman", "condicional", "variancia"],
            format_func=lambda x: {
                "srrc": "Regressao de posto (SRRC) — controla as demais entradas",
                "spearman": "Correlacao de posto — marginal",
                "condicional": "Mudanca na estatistica da saida — nao supoe monotonia",
                "variancia": "Contribuicao para a variancia — soma de quadrados sequencial",
            }[x],
            horizontal=False,
            key="sens_metodo",
        )

        if criterio in ("srrc", "spearman"):
            ordem = sensitivity.tornado_order(sres, by=criterio)
            vals = (sres.srrc if criterio == "srrc" else sres.spearman)[ordem]
            labs = [sres.names[i] for i in ordem]
            st.plotly_chart(
                charts.tornado(vals, labs, "Grafico tornado", criterio.upper()),
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(sres.as_records()).iloc[ordem],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"R² da regressao de postos = {fmt(sres.rank_r2, 3)}. É a fracao do "
                f"comportamento do modelo que indices monotonos conseguem explicar. "
                f"A coluna de contribuicao para a variancia (SRRC²) so faz sentido "
                f"com R² alto e entradas pouco correlacionadas entre si."
            )

        elif criterio == "condicional":
            cc1, cc2 = st.columns(2)
            stat_cond = cc1.selectbox(
                "Estatistica da saida", list(sensitivity.STATS_CONDICIONAIS),
                key="sens_stat",
            )
            n_faixas = cc2.slider("Faixas", 4, 25, 10, key="sens_bins")
            try:
                cond = sensitivity.change_in_output_statistic(
                    res.inputs, res.output, res.labels,
                    stat=stat_cond, bins=int(n_faixas),
                )
            except ValueError as exc:
                st.error(str(exc), icon="\U0001F6AB")
            else:
                for w in cond.warnings:
                    st.warning(w, icon="⚠️")
                oc = cond.ordem()
                st.plotly_chart(
                    charts.tornado(
                        cond.swing[oc], [cond.names[i] for i in oc],
                        f"Amplitude da {stat_cond} da saida por faixa da entrada",
                        f"swing da {stat_cond}",
                    ),
                    width="stretch",
                )
                st.plotly_chart(
                    charts.spider(
                        cond.centros[oc], cond.valores[oc],
                        [cond.names[i] for i in oc], cond.base, stat_cond,
                    ),
                    width="stretch",
                )
                st.dataframe(
                    pd.DataFrame(cond.as_records()).iloc[oc],
                    hide_index=True, width="stretch",
                )
                st.info(
                    "Este e o unico dos quatro metodos que enxerga relacao NAO "
                    "monotona. Uma entrada em U tem correlacao de posto e SRRC "
                    "proximos de zero mesmo dominando o modelo; aqui ela aparece "
                    "com swing alto e o spider mostra a curvatura. Em compensacao, "
                    "e marginal: nao controla as demais entradas.",
                    icon="ℹ️",
                )

        else:
            usar_postos = st.checkbox(
                "Regressao sobre os postos (capta relacao monotona nao linear)",
                value=False, key="sens_postos",
            )
            try:
                cv = sensitivity.contribution_to_variance(
                    res.inputs, res.output, res.labels, use_ranks=usar_postos,
                )
            except ValueError as exc:
                st.error(str(exc), icon="\U0001F6AB")
            else:
                for w in cv.warnings:
                    st.warning(w, icon="⚠️")
                ov = list(np.argsort(-cv.fracao))
                st.plotly_chart(
                    charts.tornado(
                        cv.fracao[ov] * 100.0, [cv.names[i] for i in ov],
                        "Contribuicao para a variancia da saida",
                        "% da variancia",
                    ),
                    width="stretch",
                )
                st.dataframe(
                    pd.DataFrame(cv.as_records()).iloc[ov],
                    hide_index=True, width="stretch",
                )
                v1, v2 = st.columns(2)
                v1.metric("Variancia explicada", f"{cv.r2_total:.1%}")
                v2.metric("Nao explicada", f"{cv.nao_explicada:.1%}")
                st.caption(
                    "As fracoes somam a variancia EXPLICADA, nao 100%. A parte "
                    "nao explicada e o que escapa de uma regressao linear "
                    + ("nos postos" if cv.usa_postos else "nos valores")
                    + " — e ela nao pertence a nenhuma entrada. A ordem da coluna "
                    "'passo' e a ordem de entrada na regressao, e importa: com "
                    "entradas correlacionadas, quem entra antes fica com a parte "
                    "compartilhada."
                )

        with st.expander("Dispersao entrada × saida"):
            escolha = st.selectbox("Variavel", res.labels)
            j = res.labels.index(escolha)
            passo = max(1, res.n // 4000)  # limita a 4k pontos por desempenho
            figs = go.Figure(
                go.Scattergl(
                    x=res.inputs[::passo, j],
                    y=res.output[::passo],
                    mode="markers",
                    marker=dict(size=4, opacity=0.35, color=COR["principal"]),
                )
            )
            figs.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=escolha,
                yaxis_title=nome_saida,
            )
            st.plotly_chart(figs, width="stretch")
            st.caption(
                "Inspecione a forma: se a nuvem nao for monotona, os indices do "
                "tornado sao pouco confiaveis para esta variavel."
            )

        # ---------------- exportacao ----------------
        st.markdown("---")
        st.subheader("Correlacao obtida")
        if res.spec.correlation is not None and len(res.names) > 1:
            obtida = corr_mod.achieved_spearman(res.inputs)
            pedida = np.asarray(res.spec.correlation, dtype=float)
            emax, emed, _ = corr_mod.correlation_error(res.inputs, pedida)
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Pedida (Spearman)**")
                st.dataframe(pd.DataFrame(np.round(pedida, 3), index=res.labels,
                                          columns=res.labels), width="stretch")
            with cB:
                st.markdown("**Obtida na amostra**")
                st.dataframe(pd.DataFrame(np.round(obtida, 3), index=res.labels,
                                          columns=res.labels), width="stretch")
            (st.success if emax < 0.05 else st.warning)(
                f"Maior desvio entre pedida e obtida: **{emax:.4f}** (medio {emed:.4f}).",
                icon="✅" if emax < 0.05 else "⚠️")
            st.caption(
                "As mensagens do motor mandam conferir esta tabela quando a matriz "
                "precisa de reparo. Ela e o unico lugar onde a correlacao que de fato "
                "vigorou na simulacao aparece — a pedida e apenas o alvo."
            )
        else:
            st.caption("Simulacao sem matriz de correlacao: as entradas sao independentes.")

        st.markdown("---")
        st.subheader("Cenarios")
        st.caption(
            "Duas perguntas diferentes. **Condicional** recorta as iteracoes que ja "
            "existem — mesmo modelo, mesmas probabilidades. **Estresse** troca uma "
            "distribuicao de entrada e re-simula: o resultado vale para aquele mundo "
            "hipotetico e nao tem a probabilidade do modelo original."
        )
        _ta, _tb, _tc = st.tabs(["Condicional", "Estresse", "Quem leva ao cenario"])

        with _ta:
            var_c = st.selectbox("Variavel", res.names,
                                 format_func=lambda n: res.labels[res.names.index(n)],
                                 key="cen_var")
            _col = res.inputs[:, res.names.index(var_c)]
            _col = _col[np.isfinite(_col)]
            _c1, _c2 = st.columns([1, 2])
            with _c1:
                op = st.radio("Condicao", ["maior que", "menor que"], key="cen_op")
            with _c2:
                corte = st.slider("Corte", float(np.min(_col)), float(np.max(_col)),
                                  float(np.percentile(_col, 90)), key="cen_corte")
            cen = scen_mod.conditional(
                res,
                (lambda v: v[var_c] > corte) if op == "maior que"
                else (lambda v: v[var_c] < corte),
                f"{var_c} {op} {corte:.4g}",
            )
            for _a in cen.avisos:
                st.warning(_a, icon="⚠️")
            if cen.n:
                st.caption(f"{cen.n:,} de {res.n:,} iteracoes ({cen.fracao:.2%})")
                _geral = dict(summary.describe(res.output))
                _linhas = [{"metrica": k, "no cenario": cen.resumo[k],
                            "no modelo inteiro": _geral.get(k, float("nan"))}
                           for k in ("media", "p05", "p50", "p95") if k in cen.resumo]
                st.dataframe(pd.DataFrame(_linhas), width="stretch", hide_index=True)

        with _tb:
            var_e = st.selectbox("Variavel a estressar", res.names,
                                 format_func=lambda n: res.labels[res.names.index(n)],
                                 key="est_var")
            _orig = next(v for v in res.spec.variables if v.name == var_e)
            if not _orig.params:
                st.info("Esta variavel nao tem parametros numericos para estressar.")
            else:
                _novos = {}
                _cols = st.columns(min(4, len(_orig.params)))
                for _i, (_np_, _val) in enumerate(_orig.params.items()):
                    with _cols[_i % len(_cols)]:
                        _novos[_np_] = st.number_input(_np_, value=float(_val),
                                                       key=f"est_{var_e}_{_np_}")
                if st.button("Rodar cenario de estresse", key="btn_estresse"):
                    _mudou = {k: v for k, v in _novos.items()
                              if not np.isclose(v, float(_orig.params[k]))}
                    if not _mudou:
                        st.info("Nenhum parametro foi alterado.")
                    else:
                        with st.spinner("Re-simulando o cenario..."):
                            _sr = scen_mod.stress(res.spec, {var_e: _novos},
                                                  f"{var_e} estressada", base=res)
                        for _a in _sr.avisos:
                            st.warning(_a, icon="⚠️")
                        _linhas = [{"metrica": k, "base": _sr.resumo_base[k],
                                    "estressado": _sr.resumo_estressado[k],
                                    "delta": _sr.delta[k], "delta %": _sr.delta_pct[k]}
                                   for k in ("media", "p05", "p50", "p95", "var_95", "cvar_95")
                                   if k in _sr.delta]
                        st.dataframe(pd.DataFrame(_linhas), width="stretch", hide_index=True)

        with _tc:
            st.caption(
                "As duas abas anteriores perguntam o que acontece com a SAIDA. "
                "Esta pergunta o contrario: quais ENTRADAS levaram ate la. O "
                "criterio compara a mediana de cada entrada nas iteracoes que "
                "atingiram o alvo com a mediana em todas as iteracoes, medindo a "
                "diferenca em desvios-padrao."
            )
            sc1, sc2 = st.columns(2)
            lado = sc1.radio(
                "Cauda de interesse", ["superior", "inferior"],
                format_func=lambda x: (
                    "Saida ALTA (acima do percentil)" if x == "superior"
                    else "Saida BAIXA (abaixo do percentil)"
                ),
                key="sig_cauda",
            )
            pct = sc2.slider("Percentil de corte", 1, 99, 90, key="sig_pct")
            try:
                sig = scen_mod.scenario_significance(
                    res, percentil=float(pct), cauda=lado,
                )
            except ValueError as exc:
                st.error(str(exc), icon="\U0001F6AB")
            else:
                for a_ in sig.avisos:
                    st.warning(a_, icon="⚠️")
                if sig.n:
                    st.caption(
                        f"{sig.n:,} de {res.n:,} iteracoes atingiram o cenario "
                        f"({sig.fracao:.2%})."
                    )
                    osig = sig.ordem()
                    st.plotly_chart(
                        charts.tornado(
                            sig.significancia[osig],
                            [sig.labels[i] for i in osig],
                            "Significancia para o cenario",
                            "desvios-padrao de deslocamento da mediana",
                            limiar=sig.limiar,
                        ),
                        width="stretch",
                    )
                    st.dataframe(
                        pd.DataFrame(sig.as_records()).iloc[osig],
                        hide_index=True, width="stretch",
                    )
                    st.caption(
                        f"As barras cinzas ficam abaixo do limiar de "
                        f"{sig.limiar:g} e sao consideradas insignificantes. O "
                        f"limiar e CONVENCAO, nao teste de hipotese: nao ha "
                        f"p-valor associado e ele nao se ajusta ao tamanho do "
                        f"recorte. Com poucas iteracoes no cenario, ruido sozinho "
                        f"cruza a linha."
                    )

        # ---------------- convergencia com criterio ----------------
        st.markdown("---")
        st.subheader("Ja rodei iteracoes suficientes?")
        st.caption(
            "O grafico de media acumulada, acima, responde \"parece ter "
            "estabilizado?\". Aqui a pergunta e quantitativa: a estimativa ja "
            "esta dentro de uma tolerancia declarada, com uma confianca "
            "declarada?"
        )
        k1, k2, k3 = st.columns(3)
        tol = k1.slider("Tolerancia relativa", 0.005, 0.10, 0.03, 0.005,
                        format="%.3f", key="conv_tol")
        conf = k2.select_slider(
            "Nivel de confianca", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
            key="conv_conf",
        )
        alvos = k3.multiselect(
            "Estatisticas", list(conv_mod.ESTATISTICAS), default=["media", "p95"],
            key="conv_stats",
        )
        if not alvos:
            st.info("Escolha ao menos uma estatistica para monitorar.")
        else:
            try:
                rel = conv_mod.monitor(
                    res.output, alvos, tolerancia=float(tol),
                    confianca=float(conf), metodo_amostragem=res.spec.method,
                )
            except ValueError as exc:
                st.error(str(exc), icon="\U0001F6AB")
            else:
                for a_ in rel.avisos:
                    st.warning(a_, icon="⚠️")
                st.dataframe(
                    pd.DataFrame(rel.as_records()), hide_index=True,
                    width="stretch",
                )
                if rel.tudo_convergiu:
                    st.success(
                        f"Todas as estatisticas monitoradas convergiram; a mais "
                        f"lenta levou {rel.iteracoes_necessarias:,} iteracoes de "
                        f"{rel.n_total:,}.",
                        icon="✅",
                    )
                else:
                    proj = conv_mod.iteracoes_para_tolerancia(
                        res.output, float(tol), float(conf)
                    )
                    st.warning(
                        "Nem tudo convergiu na tolerancia pedida."
                        + (
                            f" Projecao para a MEDIA (so para ela): cerca de "
                            f"{proj:,.0f} iteracoes."
                            if np.isfinite(proj) else ""
                        ),
                        icon="⚠️",
                    )
                escolhida = st.selectbox(
                    "Detalhar", alvos, key="conv_detalhe",
                )
                tr = rel.trilha[escolhida]
                st.plotly_chart(
                    charts.convergencia(
                        [e.n for e in tr], [e.valor for e in tr],
                        [e.meia_largura for e in tr], float(tol),
                        rel.convergiu_em[escolhida], escolhida,
                    ),
                    width="stretch",
                )
                st.caption(
                    "Convergencia e sobre erro de AMOSTRAGEM, que e a menor das "
                    "fontes de erro em analise de risco. Uma simulacao convergida "
                    "com premissas erradas produz um numero errado com barra de "
                    "erro estreita — e a barra estreita convida a confiar."
                )

        # ---------------- comparacao de distribuicoes ----------------
        st.markdown("---")
        st.subheader("Comparar distribuicoes")
        st.caption(
            "Sobreposicao, tendencia e box plot lado a lado. As series "
            "disponiveis sao a saida, as entradas e — se voce tiver rodado "
            "replicacoes — cada replica, o que torna visivel quanto do formato "
            "e ruido de simulacao."
        )
        disponiveis: dict[str, np.ndarray] = {nome_saida: res.output}
        for _j, _lab in enumerate(res.labels):
            disponiveis[f"entrada: {_lab}"] = res.inputs[:, _j]
        _reps = st.session_state.get("replicas")
        if _reps:
            for _i, _o in enumerate(_reps):
                disponiveis[f"replica {_i + 1}"] = np.asarray(_o, dtype=float)
        sel = st.multiselect(
            "Series", list(disponiveis), default=[nome_saida], key="cmp_series",
        )
        if len(sel) < 1:
            st.info("Escolha ao menos uma serie.")
        else:
            series = [disponiveis[s_] for s_ in sel]
            tipo = st.radio(
                "Visualizacao",
                ["densidade", "acumulada", "tendencia", "box"],
                horizontal=True, key="cmp_tipo",
            )
            try:
                if tipo in ("densidade", "acumulada"):
                    fig_cmp = charts.overlay(
                        series, sel, cumulativa=(tipo == "acumulada"),
                        titulo="Sobreposicao com binning comum",
                    )
                elif tipo == "tendencia":
                    fig_cmp = charts.summary_trend(series, sel)
                else:
                    fig_cmp = charts.box_plot(series, sel)
            except ValueError as exc:
                st.info(str(exc))
            else:
                st.plotly_chart(fig_cmp, width="stretch")
                if tipo == "densidade":
                    st.caption(
                        "As bordas dos intervalos sao calculadas sobre a UNIAO "
                        "das series. Histogramas com intervalos proprios nao sao "
                        "comparaveis: a mesma distribuicao com 40 e com 80 "
                        "intervalos tem alturas diferentes."
                    )
                elif tipo == "box":
                    st.caption(
                        "Os bigodes vao ate P5 e P95, nao ate 1,5x o intervalo "
                        "interquartil. A regra de 1,5x marcaria como atipica uma "
                        "fracao enorme das iteracoes numa saida de cauda pesada — "
                        "que e o caso normal em analise de risco."
                    )

        st.subheader("Exportar")
        df = to_dataframe(res, output_name=nome_saida)
        e1, e2, e3 = st.columns(3)
        e1.download_button(
            "⬇ Iteracoes (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="simulacao_iteracoes.csv",
            mime="text/csv",
            width="stretch",
        )

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.head(100_000).to_excel(w, sheet_name="iteracoes", index=False)
            pd.DataFrame([d]).T.rename(columns={0: "valor"}).to_excel(
                w, sheet_name="estatisticas"
            )
            pd.DataFrame(sres.as_records()).to_excel(
                w, sheet_name="sensibilidade", index=False
            )
        e2.download_button(
            "⬇ Relatorio (Excel)",
            buf.getvalue(),
            file_name="simulacao_relatorio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        if len(df) > 100_000:
            st.caption(
                f"O Excel traz as primeiras 100.000 de {len(df):,} iteracoes. "
                f"O CSV traz todas."
            )

        modelo = {
            "iteracoes": res.spec.iterations,
            "metodo": res.spec.method,
            "semente": res.spec.seed,
            "formula": res.spec.formula,
            "variaveis": [
                {
                    "nome": v.name,
                    "rotulo": v.label,
                    "distribuicao": v.dist_key,
                    "parametros": v.params,
                    "valores": v.values,
                    "probabilidades": v.probs,
                }
                for v in res.spec.variables
            ],
            "correlacao": (
                res.spec.correlation.tolist()
                if res.spec.correlation is not None
                else None
            ),
        }
        e3.download_button(
            "⬇ Especificacao (JSON)",
            json.dumps(modelo, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="modelo.json",
            mime="application/json",
            width="stretch",
        )
        st.caption(
            "O JSON documenta a especificacao completa, incluindo a semente: "
            "e o que torna o resultado auditavel e reproduzivel por terceiros."
        )


# ===========================================================================
# Aba 5 - ajuste a dados
# ===========================================================================

with aba_ajuste:
    st.subheader("Ajustar distribuicoes a dados historicos")
    st.markdown(
        "Carregue uma serie e compare candidatas por maxima verossimilhanca. "
        "Use o resultado para escolher a distribuicao de uma variavel na aba 1."
    )

    origem = st.radio(
        "Origem dos dados", ["Colar valores", "Arquivo CSV"], horizontal=True
    )
    dados: list[float] = []

    if origem == "Colar valores":
        txt = st.text_area(
            "Valores (virgulas ou quebras de linha)", height=130, key="fit_txt"
        )
        dados = parse_lista(txt)
    else:
        up = st.file_uploader("Arquivo CSV", type=["csv"])
        if up is not None:
            try:
                dfu = pd.read_csv(up)
                num = dfu.select_dtypes(include=[np.number])
                if num.empty:
                    st.error("Nenhuma coluna numerica encontrada no arquivo.")
                else:
                    col = st.selectbox("Coluna", list(num.columns))
                    dados = num[col].dropna().tolist()
            except Exception as e:
                st.error(f"Nao foi possivel ler o arquivo: {e}")

    if len(dados) < 5:
        st.info("Sao necessarias ao menos 5 observacoes numericas.")
    else:
        desc = fitting.describe_data(dados)
        cc = st.columns(len(desc))
        for i, (k, val) in enumerate(desc.items()):
            cc[i].metric(k, f"{int(val)}" if k == "n" else fmt(val, 4))

        if desc["n"] < 30:
            st.warning(
                f"Apenas {int(desc['n'])} observacoes. Com amostras pequenas, o "
                f"ranking entre distribuicoes e instavel e os testes de aderencia "
                f"tem pouco poder: quase nenhuma candidata sera rejeitada. Trate o "
                f"resultado como exploratorio.",
                icon="⚠️",
            )

        candidatas = st.multiselect(
            "Candidatas",
            options=list(fitting.FITTABLE),
            default=[
                k
                for k in ["normal", "lognormal", "gamma", "weibull", "logistic", "gumbel_r"]
                if k in fitting.FITTABLE
            ],
        )
        b = st.select_slider(
            "Replicas de bootstrap para o p-valor",
            options=[0, 100, 200, 500, 1000],
            value=200,
            help=(
                "0 desliga o teste. O bootstrap parametrico e a unica forma "
                "correta de obter p-valor quando os parametros sao estimados dos "
                "mesmos dados. Custa tempo proporcional ao numero de replicas."
            ),
        )

        # O resultado precisa sobreviver aos reruns: `st.button` so devolve True
        # no rerun do proprio clique. Sem guardar em session_state, qualquer
        # interacao posterior — inclusive o seletor "Inspecionar ajuste", que
        # faz parte deste bloco — apagaria a tabela inteira da tela.
        if st.button("Ajustar", type="primary") and candidatas:
            with st.spinner("Ajustando e reamostrando..."):
                resultados = fitting.fit_many(
                    dados,
                    candidatas,
                    bootstrap=int(b),
                    rng=np.random.default_rng(seed),
                )
            st.session_state["ajuste"] = {
                "resultados": resultados,
                "dados": list(dados),
                "candidatas": list(candidatas),
                "bootstrap": int(b),
            }

        aj = st.session_state.get("ajuste")
        if aj is not None and aj["dados"] != list(dados):
            st.info(
                "Os dados mudaram desde o ultimo ajuste. Clique em **Ajustar** "
                "para recalcular.",
                icon="ℹ️",
            )
        elif aj is not None:
            resultados = aj["resultados"]
            if not resultados:
                st.error("Nenhum ajuste convergiu para as candidatas selecionadas.")
            else:
                pesos = fitting.akaike_weights(resultados)
                tab = pd.DataFrame([r.as_record() for r in resultados])
                tab.insert(1, "peso_akaike", pesos)
                st.dataframe(tab, hide_index=True, width="stretch")

                st.info(
                    "**Como ler esta tabela.** AIC/BIC apenas ORDENAM as "
                    "candidatas: a primeira colocada pode ainda ser um ajuste "
                    "ruim. O peso de Akaike e evidencia RELATIVA dentro do "
                    "conjunto testado. O p-valor por bootstrap e o unico numero "
                    "aqui que responde 'este modelo e compativel com os dados?' — "
                    "valores baixos (< 0,05) indicam incompatibilidade. Sempre "
                    "confira o Q-Q plot abaixo.",
                    icon="ℹ️",
                )

                melhor = resultados[0]
                if pesos[0] < 0.6 and len(resultados) > 1:
                    st.warning(
                        f"O peso de Akaike da primeira colocada e {pesos[0]:.2f}: "
                        f"os dados nao distinguem claramente entre as candidatas. "
                        f"Escolher apenas a vencedora subestima a incerteza de "
                        f"modelo. Considere rodar a simulacao com mais de uma "
                        f"distribuicao e comparar os resultados.",
                        icon="⚠️",
                    )
                if melhor.ks_pvalue_bootstrap is not None and melhor.ks_pvalue_bootstrap < 0.05:
                    st.error(
                        f"Mesmo a distribuicao melhor colocada e rejeitada pelo "
                        f"teste K-S (p = {melhor.ks_pvalue_bootstrap:.3f}). Nenhuma "
                        f"candidata descreve bem estes dados — usar qualquer uma "
                        f"delas na simulacao propaga um erro de especificacao.",
                        icon="\U0001F6AB",
                    )

                escolhida = st.selectbox(
                    "Inspecionar ajuste",
                    [r.name for r in resultados],
                    key="ajuste_inspecionar",
                )
                r = next(x for x in resultados if x.name == escolhida)
                gg1, gg2 = st.columns(2)

                x = np.asarray(aj["dados"], dtype=float)
                x = x[np.isfinite(x)]
                fr = fitting.FITTABLE[r.dist_key](*r.params)
                grade = np.linspace(x.min(), x.max(), 400)
                figd = go.Figure()
                figd.add_trace(
                    go.Histogram(
                        x=x,
                        histnorm="probability density",
                        nbinsx=min(50, max(10, int(np.sqrt(x.size)))),
                        name="dados",
                        marker_color=COR["neutro"],
                        opacity=0.6,
                    )
                )
                figd.add_trace(
                    go.Scatter(
                        x=grade,
                        y=fr.pdf(grade),
                        mode="lines",
                        name=r.name,
                        line=dict(color=COR["principal"], width=3),
                    )
                )
                figd.update_layout(
                    height=360, margin=dict(l=0, r=0, t=30, b=0), title="Densidade"
                )
                gg1.plotly_chart(figd, width="stretch", key="ajuste_densidade")

                teo, amo = fitting.qq_points(x, r)
                lim = [min(teo.min(), amo.min()), max(teo.max(), amo.max())]
                figq = go.Figure()
                figq.add_trace(
                    go.Scattergl(
                        x=teo,
                        y=amo,
                        mode="markers",
                        marker=dict(size=5, color=COR["principal"], opacity=0.6),
                        name="quantis",
                    )
                )
                figq.add_trace(
                    go.Scatter(
                        x=lim,
                        y=lim,
                        mode="lines",
                        line=dict(dash="dash", color=COR["alerta"]),
                        name="ajuste perfeito",
                    )
                )
                figq.update_layout(
                    height=360,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title="Q-Q plot",
                    xaxis_title="quantis teoricos",
                    yaxis_title="quantis amostrais",
                )
                gg2.plotly_chart(figq, width="stretch", key="ajuste_qq")
                st.caption(
                    "No Q-Q plot, olhe especialmente as PONTAS. Desvios no centro "
                    "quase nao afetam a decisao; desvios na cauda mudam "
                    "completamente o VaR e a probabilidade de cenarios extremos."
                )

                st.markdown(
                    f"**Parametros ajustados ({r.name}):** "
                    f"`{', '.join(f'{p:.6g}' for p in r.params)}` — na "
                    f"parametrizacao loc/scale do `scipy.stats`. Confira a "
                    f"correspondencia antes de transcrever para a aba 1."
                )

                # ---------------- incerteza que o ajuste pontual descarta ----
                st.markdown("---")
                st.subheader("O que o ajuste pontual esconde")
                st.caption(
                    "Levar so os parametros vencedores para a aba 1 descarta duas "
                    "fontes de incerteza, e as duas empurram o resultado na mesma "
                    "direcao: para MENOS incerteza do que existe."
                )
                _ia, _ib = st.tabs(
                    ["Incerteza de parametro", "Incerteza de modelo"]
                )

                with _ia:
                    st.caption(
                        "Os parametros vieram de uma amostra finita. Bootstrap "
                        "nao parametrico: reamostra as observacoes com reposicao "
                        "e reajusta em cada replica. Nao parametrico de proposito "
                        "— o bootstrap parametrico simularia do proprio modelo "
                        "ajustado e, se ele estiver errado, devolveria um "
                        "intervalo estreito e igualmente errado."
                    )
                    _nb = st.select_slider(
                        "Replicas de bootstrap", options=[50, 100, 200, 500],
                        value=200, key="inc_reps",
                    )
                    if st.button("Estimar incerteza dos parametros",
                                 key="btn_inc_param"):
                        try:
                            with st.spinner("Reamostrando e reajustando..."):
                                _inc = fitting.parameter_uncertainty(
                                    aj["dados"], r.dist_key, replicas=int(_nb),
                                    rng=np.random.default_rng(seed),
                                )
                        except ValueError as exc:
                            st.error(str(exc), icon="\U0001F6AB")
                        else:
                            st.session_state["incerteza"] = _inc
                    _inc = st.session_state.get("incerteza")
                    if _inc is not None and _inc.dist_key == r.dist_key:
                        for _a in _inc.avisos:
                            st.warning(_a, icon="⚠️")
                        st.dataframe(
                            pd.DataFrame(_inc.as_records()), hide_index=True,
                            width="stretch",
                        )
                        _rng = np.random.default_rng(seed)
                        _n_cmp = 60_000
                        _pontual = fitting.FITTABLE[r.dist_key].rvs(
                            *r.params, size=_n_cmp, random_state=_rng
                        )
                        _preditiva = fitting.simular_com_incerteza(
                            _inc, _n_cmp, np.random.default_rng(seed + 1)
                        )
                        st.plotly_chart(
                            charts.overlay(
                                [_pontual, _preditiva],
                                ["parametros pontuais", "com incerteza de parametro"],
                                cumulativa=True,
                                titulo="O efeito na cauda",
                            ),
                            width="stretch",
                        )
                        _lin = []
                        for _q in (90, 95, 99, 99.9):
                            _a1 = float(np.percentile(_pontual, _q))
                            _a2 = float(np.percentile(_preditiva, _q))
                            _lin.append({
                                "percentil": f"P{_q:g}",
                                "pontual": _a1,
                                "com incerteza": _a2,
                                "diferenca %": (
                                    (_a2 - _a1) / abs(_a1) * 100.0 if _a1 else float("nan")
                                ),
                            })
                        st.dataframe(pd.DataFrame(_lin), hide_index=True,
                                     width="stretch")
                        st.caption(
                            "O efeito cresce com a raridade do percentil e "
                            "encolhe com o tamanho da amostra — e proporcional a "
                            "1/n na variancia. Com muitos dados a diferenca "
                            "desaparece, e nesse caso ignorar a incerteza de "
                            "parametro passa a ser defensavel."
                        )

                with _ib:
                    if len(resultados) < 2:
                        st.info(
                            "E preciso mais de uma candidata ajustada para haver "
                            "incerteza de modelo."
                        )
                    else:
                        st.caption(
                            "Quando os pesos de Akaike ficam repartidos, os dados "
                            "nao escolheram. Rodar so com a vencedora apresenta "
                            "como certa uma decisao que foi quase empate — e o "
                            "desempate cai na cauda, que e onde ha menos "
                            "observacoes para decidir."
                        )
                        _ver_ma = st.button(
                            "Combinar as candidatas por peso de Akaike",
                            key="btn_model_avg",
                        )
                        if _ver_ma:
                            st.session_state["ver_media_modelos"] = True
                        try:
                            _ma = fitting.model_average(resultados)
                        except ValueError as exc:
                            st.error(str(exc), icon="\U0001F6AB")
                            _ma_ok = False
                        else:
                            _ma_ok = True
                            for _a in _ma.avisos:
                                st.info(_a, icon="ℹ️")
                            st.dataframe(
                                pd.DataFrame(_ma.as_records()), hide_index=True,
                                width="stretch",
                            )
                        if _ma_ok and st.session_state.get("ver_media_modelos"):
                            _n_cmp = 60_000
                            _so_vencedora = fitting.FITTABLE[
                                resultados[0].dist_key
                            ].rvs(*resultados[0].params, size=_n_cmp,
                                  random_state=np.random.default_rng(seed))
                            _mistura = fitting.simular_media_de_modelos(
                                _ma, _n_cmp, np.random.default_rng(seed + 2)
                            )
                            st.plotly_chart(
                                charts.overlay(
                                    [_so_vencedora, _mistura],
                                    ["so a vencedora", "media de modelos"],
                                    cumulativa=True,
                                    titulo="O efeito na cauda",
                                ),
                                width="stretch",
                            )
                            _lin = []
                            for _q in (90, 95, 99, 99.9):
                                _a1 = float(np.percentile(_so_vencedora, _q))
                                _a2 = float(np.percentile(_mistura, _q))
                                _lin.append({
                                    "percentil": f"P{_q:g}",
                                    "so a vencedora": _a1,
                                    "media de modelos": _a2,
                                    "diferenca %": (
                                        (_a2 - _a1) / abs(_a1) * 100.0
                                        if _a1 else float("nan")
                                    ),
                                })
                            st.dataframe(pd.DataFrame(_lin), hide_index=True,
                                         width="stretch")
                            st.caption(
                                "A mistura sorteia a FAMILIA a cada iteracao, "
                                "ponderada pelos pesos de Akaike. Fazer a media "
                                "dos quantis das candidatas seria diferente e "
                                "errado: produziria uma curva que nao e a "
                                "preditiva de nada e que tem cauda mais leve que "
                                "a mais pesada das candidatas."
                            )


# ===========================================================================
# Aba 6 - metodologia
# ===========================================================================

with aba_metodo:
    st.subheader("Metodologia")
    st.markdown(
        """
### O que o app faz, nesta ordem

1. **Amostragem** — gera uma matriz `n × k` no cubo unitario, por Monte Carlo
   simples ou Latin Hypercube.
2. **Transformada inversa** — aplica a funcao quantil de cada marginal,
   coluna a coluna.
3. **Dependencia** — se especificada, aplica Iman-Conover (reordena as
   colunas para atingir a correlacao de posto alvo preservando as marginais)
   ou uma das cinco copulas, escolhida na aba 2. Iman-Conover e a Gaussiana
   nao produzem eventos extremos simultaneos; a t produz em ambas as caudas;
   Clayton so na inferior e Gumbel so na superior.
4. **Avaliacao** — calcula a formula de saida de forma vetorizada.
5. **Diagnostico** — estatisticas, erro de simulacao, sensibilidade por
   quatro metodos, significancia de cenario e criterio de convergencia.

### Decisoes metodologicas que valem explicitar

- **Amostragem por transformada inversa** para tudo. E o que permite trocar
  o esquema de amostragem sem tocar nas distribuicoes, e o que faz LHS e
  Iman-Conover funcionarem.
- **Correcao arcsin no Iman-Conover.** O algoritmo original controla a
  correlacao de Pearson dos escores normais; miramos
  `rho_P = 2·sin(π·rho_S/6)` para que o **Spearman final** bata com o alvo.
  Medido no repositorio: erro medio absoluto de **0,0010 com** a correcao
  contra **0,0144 sem** ela.
- **Espelhamento da matriz de correlacao.** A grade e preenchida em um
  triangulo so. Tirar a media com o lado vazio dividiria por dois toda
  correlacao digitada, silenciosamente; o app espelha o lado preenchido e
  avisa quando os dois lados conflitam.
- **p-valor por bootstrap parametrico** no ajuste. Com parametros estimados
  dos mesmos dados, o p-valor assintotico do K-S e invalido. Medido no
  repositorio, sob H0 verdadeira: o teste ingenuo devolve p-valor medio
  **0,77** e rejeita em **0%** dos casos ao nivel de 5%; o bootstrap devolve
  media **0,50** e rejeita em **8,8%** — proximo do nominal.
- **Erro de simulacao sob LHS.** `s/√n` nao e valido quando as iteracoes sao
  dependentes. Por isso ha replicacoes independentes como opcao.
- **Sem `eval` cru.** A formula e analisada com `ast` e restrita a uma lista
  branca de construcoes.
- **Quatro metodos de sensibilidade, nao um.** Correlacao de posto e SRRC sao
  cegos para relacao nao monotona: medido no repositorio com `y = a²`, ambos
  devolvem **−0,0022** para a unica variavel que importa. O metodo condicional
  (faixas equiprovaveis) devolve swing **3,23** contra **0,08** das
  irrelevantes. A discordancia entre metodos e o achado, nao um defeito.
- **Faixas equiprovaveis, nao de largura igual**, no metodo condicional.
  Dividir por largura esvaziaria as faixas de cauda em qualquer entrada
  assimetrica, e a estatistica ali seria ruido puro.
- **Contribuicao para a variancia soma o R², nao 100%.** O que sobra nao
  pertence a entrada nenhuma. Com `y = a·b`, medido: **99,98%** nao explicada.
- **Calibracao das copulas arquimedianas por medicao.** Nao ha forma fechada
  elementar ligando o rho de Spearman ao parametro dessas familias, e inventar
  uma seria pior que nao ter. A grade e simulada com semente fixa e invertida
  por interpolacao; o erro medido esta no `BENCHMARK.md` (ate 0,0077, contra
  0,0025 da calibracao por tau de Kendall, que tem forma fechada).
- **Bootstrap basico na incerteza de parametro.** Reamostrar as replicas do
  MLE sem refletir ESTREITA a preditiva em vez de alargar, porque o vies para
  baixo do estimador de escala cancela o alargamento da incerteza de locacao.
  Medido para a normal com n=40: variancia **0,7970** crua contra **0,8043**
  pontual e **0,8912** exata; com reflexao, **0,8720**.

### O que este app NAO faz

Nao tem indices de Sobol, otimizacao sob incerteza, series temporais e
processos estocasticos, nem integracao com Excel.

Ha ajuste de copula a dados (`mcrisk.copula.fit_copula`, por
pseudo-verossimilhanca), mas ele NAO esta ligado ao seletor da aba 2: a copula
usada na simulacao continua sendo ESCOLHIDA por voce. Os graus de liberdade da
t sao um parametro que voce informa, nao estimado. Das arquimedianas ha
Clayton, Gumbel e Frank -- faltam as assimetricas de dois parametros, e as tres
implementadas sao PERMUTAVEIS: um unico parametro para todos os pares. Uma
matriz heterogenea e achatada no rho medio, com aviso na tela.

A incerteza de parametro e propagada por bootstrap e as candidatas podem ser
combinadas por peso de Akaike (aba 5), mas nao ha inferencia bayesiana
propriamente dita: nao ha priori, posteriori nem MCMC. Ver `LIMITATIONS.md`.
"""
    )

    st.subheader("Referencias")
    st.markdown(
        """
**Amostragem**

- McKay, M.D., Beckman, R.J. & Conover, W.J. (1979). *A Comparison of Three
  Methods for Selecting Values of Input Variables in the Analysis of Output
  from a Computer Code*. Technometrics 21(2):239-245.
- Stein, M. (1987). *Large Sample Properties of Simulations Using Latin
  Hypercube Sampling*. Technometrics 29(2):143-151.
- Helton, J.C. & Davis, F.J. (2003). *Latin hypercube sampling and the
  propagation of uncertainty in analyses of complex systems*. Reliability
  Engineering & System Safety 81(1):23-69.

**Correlacao**

- Iman, R.L. & Conover, W.J. (1982). *A distribution-free approach to inducing
  rank correlation among input variables*. Communications in Statistics -
  Simulation and Computation 11(3):311-334.
- Higham, N.J. (2002). *Computing the nearest correlation matrix - a problem
  from finance*. IMA Journal of Numerical Analysis 22(3):329-343.
- Kruskal, W.H. (1958). *Ordinal Measures of Association*. JASA 53:814-861.

**Ajuste e selecao de modelo**

- Akaike, H. (1974). IEEE Transactions on Automatic Control 19(6):716-723.
- Schwarz, G. (1978). Annals of Statistics 6(2):461-464.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel
  Inference*, 2a ed., Springer.
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
- Hoeting, J.A. et al. (1999). *Bayesian Model Averaging: A Tutorial*.
  Statistical Science 14(4):382-401.
- Lilliefors, H.W. (1967). JASA 62(318):399-402.
- Stephens, M.A. (1974). *EDF Statistics for Goodness of Fit and Some
  Comparisons*. JASA 69(347):730-737.
- Babu, G.J. & Rao, C.R. (2004). *Goodness-of-fit tests when parameters are
  estimated*. Sankhya 66(1):63-74.

**Copulas**

- Sklar, A. (1959). *Fonctions de repartition a n dimensions et leurs marges*.
  Publ. Inst. Statist. Univ. Paris 8:229-231.
- Marshall, A.W. & Olkin, I. (1988). *Families of Multivariate Distributions*.
  JASA 83(403):834-841. (amostragem por frailty das arquimedianas)
- Genest, C. & Rivest, L.-P. (1993). *Statistical Inference Procedures for
  Bivariate Archimedean Copulas*. JASA 88(423):1034-1043.
- Demarta, S. & McNeil, A.J. (2005). *The t Copula and Related Copulas*.
  International Statistical Review 73(1):111-129.
- Nelsen, R.B. (2006). *An Introduction to Copulas*, 2a ed., Springer.
- Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk
  Management*, ed. revisada, Princeton University Press.

**Sensibilidade**

- Helton, J.C. & Davis, F.J. (2002). *Illustration of Sampling-Based Methods
  for Uncertainty and Sensitivity Analysis*. Risk Analysis 22(3):591-622.
- Saltelli, A. & Sobol', I.M. (1995). *About the use of rank transformation in
  sensitivity analysis of model output*. Reliability Engineering & System
  Safety 50(3):225-239.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*, Wiley.

**Medidas de risco e distribuicoes**

- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). *Coherent Measures
  of Risk*. Mathematical Finance 9(3):203-228.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*,
  Springer.
- Vose, D. (2008). *Risk Analysis: A Quantitative Guide*, 3a ed., Wiley.
- Embrechts, P., Kluppelberg, C. & Mikosch, T. (1997). *Modelling Extremal
  Events for Insurance and Finance*, Springer.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme
  Values*, Springer.

**Critica ao uso de modelos quantitativos de risco**

- Savage, S.L. (2009). *The Flaw of Averages*, Wiley.
- Taleb, N.N. (2007). *The Black Swan*, Random House.
- Saltelli, A. et al. (2020). *Five ways to ensure that models serve society:
  a manifesto*. Nature 582:482-484.
"""
    )
