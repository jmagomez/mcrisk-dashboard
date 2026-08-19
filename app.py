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
        }
        dep = st.radio(
            "Esquema",
            list(esquemas),
            format_func=lambda k: esquemas[k],
            horizontal=True,
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
            linhas = []
            for p in summary.DEFAULT_PERCENTILES:
                lo, hi = summary.quantile_ci(yf, p / 100.0)
                linhas.append(
                    {
                        "percentil": f"P{p}",
                        "valor": d[f"P{p}"],
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

        criterio = st.radio(
            "Ordenar por",
            ["srrc", "spearman"],
            format_func=lambda x: {
                "srrc": "Coef. de regressao de posto padronizado (controla as demais)",
                "spearman": "Correlacao de posto (marginal)",
            }[x],
            horizontal=True,
        )
        ordem = sensitivity.tornado_order(sres, by=criterio)
        vals = (sres.srrc if criterio == "srrc" else sres.spearman)[ordem]
        labs = [sres.names[i] for i in ordem]

        figt = go.Figure(
            go.Bar(
                x=vals[::-1],
                y=labs[::-1],
                orientation="h",
                marker_color=[
                    COR["alerta"] if x < 0 else COR["principal"] for x in vals[::-1]
                ],
            )
        )
        figt.update_layout(
            height=max(240, 44 * len(labs)),
            margin=dict(l=0, r=0, t=30, b=0),
            title="Grafico tornado",
            xaxis_title=criterio.upper(),
        )
        st.plotly_chart(figt, width="stretch")

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
        _ta, _tb = st.tabs(["Condicional", "Estresse"])

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
   ou uma copula (Gaussiana ou t), escolhida na aba 2. So a copula t produz
   eventos extremos simultaneos.
4. **Avaliacao** — calcula a formula de saida de forma vetorizada.
5. **Diagnostico** — estatisticas, erro de simulacao, sensibilidade.

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

### O que este app NAO faz

Nao tem indices de Sobol, otimizacao sob incerteza, series temporais e
processos estocasticos, ajuste bayesiano, nem integracao com Excel. As
copulas disponiveis (Gaussiana e t) sao ESCOLHIDAS por voce, nao ajustadas
aos dados: os graus de liberdade sao um parametro que voce informa, e nao ha
teste de aderencia da estrutura de dependencia. Ver `LIMITATIONS.md`.
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
- Lilliefors, H.W. (1967). JASA 62(318):399-402.
- Stephens, M.A. (1974). *EDF Statistics for Goodness of Fit and Some
  Comparisons*. JASA 69(347):730-737.
- Babu, G.J. & Rao, C.R. (2004). *Goodness-of-fit tests when parameters are
  estimated*. Sankhya 66(1):63-74.

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
