"""
Construtores de figuras para o dashboard.

Ficam FORA do pacote `mcrisk` de proposito. O pacote e o motor: puro NumPy e
SciPy, sem dependencia de biblioteca grafica, porque e ele que roda tambem no
navegador via Pyodide e em scripts de terceiros. Grafico e apresentacao, e
apresentacao pertence a camada da interface.

Cada funcao aqui recebe numeros ja calculados pelo motor e devolve uma figura
Plotly. Nenhuma delas calcula estatistica: se um grafico precisasse de um
numero novo, esse numero seria calculado no motor, onde ha teste, e nao aqui.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

COR = {
    "principal": "#2563eb",
    "alerta": "#dc2626",
    "neutro": "#9ca3af",
    "ok": "#059669",
}

# Paleta para series multiplas. Escolhida para continuar distinguivel em
# escala de cinza e para daltonismo do tipo deuteranopia (Okabe & Ito, 2008).
PALETA = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
)


def _base(fig: go.Figure, titulo: str, altura: int = 400) -> go.Figure:
    fig.update_layout(
        height=altura,
        margin=dict(l=0, r=0, t=36, b=0),
        title=titulo,
        hovermode="closest",
    )
    return fig


def spider(
    centros: np.ndarray,
    valores: np.ndarray,
    labels: Sequence[str],
    base: float,
    stat: str = "media",
    titulo: str = "Grafico spider",
) -> go.Figure:
    """Como a estatistica da saida anda ao longo da faixa de cada entrada.

    Uma linha por entrada. O eixo x e o PERCENTIL da entrada, nao o valor
    dela: sem isso as linhas ficariam em unidades diferentes (reais, dias,
    toneladas) e nao poderiam dividir o mesmo eixo.

    A linha horizontal e a estatistica sobre a amostra inteira. Entrada
    irrelevante produz linha colada nela; entrada importante produz
    inclinacao. E o unico grafico deste dashboard que mostra a FORMA da
    relacao - o tornado so mostra a magnitude, e por isso nao distingue uma
    relacao monotona de uma em U.
    """
    centros = np.atleast_2d(np.asarray(centros, dtype=float))
    valores = np.atleast_2d(np.asarray(valores, dtype=float))
    if centros.shape != valores.shape:
        raise ValueError(
            f"centros {centros.shape} e valores {valores.shape} precisam ter a "
            f"mesma forma"
        )
    if len(labels) != centros.shape[0]:
        raise ValueError("um rotulo por entrada")

    k, bins = centros.shape
    eixo = (np.arange(bins) + 0.5) / bins * 100.0
    fig = go.Figure()
    for j in range(k):
        fig.add_trace(
            go.Scatter(
                x=eixo,
                y=valores[j],
                mode="lines+markers",
                name=labels[j],
                line=dict(color=PALETA[j % len(PALETA)], width=2),
                marker=dict(size=6),
                customdata=centros[j],
                hovertemplate=(
                    f"<b>{labels[j]}</b><br>valor da entrada: %{{customdata:.4g}}"
                    f"<br>{stat} da saida: %{{y:.4g}}<extra></extra>"
                ),
            )
        )
    fig.add_hline(
        y=float(base),
        line_dash="dot",
        line_color=COR["neutro"],
        annotation_text=f"{stat} geral",
    )
    fig.update_layout(
        xaxis_title="percentil da entrada (faixas equiprovaveis)",
        yaxis_title=f"{stat} da saida",
    )
    return _base(fig, titulo, 440)


def tornado(
    valores: np.ndarray,
    labels: Sequence[str],
    titulo: str,
    eixo_x: str,
    limiar: float | None = None,
) -> go.Figure:
    """Barras horizontais ordenadas por magnitude.

    A ordem vem pronta de quem chama - ordenar aqui esconderia a escolha do
    criterio, que e uma decisao metodologica e nao de desenho.
    """
    valores = np.asarray(valores, dtype=float)
    if len(labels) != valores.size:
        raise ValueError("um rotulo por barra")
    cores = [COR["alerta"] if v < 0 else COR["principal"] for v in valores]
    if limiar is not None:
        cores = [
            c if abs(v) >= limiar else COR["neutro"]
            for c, v in zip(cores, valores)
        ]
    fig = go.Figure(
        go.Bar(
            x=valores[::-1],
            y=list(labels)[::-1],
            orientation="h",
            marker_color=cores[::-1],
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )
    )
    if limiar is not None:
        for sinal in (-1, 1):
            fig.add_vline(
                x=sinal * limiar, line_dash="dash", line_color=COR["neutro"]
            )
    fig.update_layout(xaxis_title=eixo_x)
    return _base(fig, titulo, max(240, 44 * len(labels)))


def summary_trend(
    series: Sequence[np.ndarray],
    labels: Sequence[str],
    bandas: Sequence[tuple[float, float]] = ((5, 95), (25, 75)),
    titulo: str = "Tendencia resumida",
) -> go.Figure:
    """Faixas de percentil ao longo de uma sequencia de distribuicoes.

    Serve para ver como a incerteza EVOLUI: por periodo, por cenario, por
    variante do modelo. Cada elemento de `series` e uma distribuicao inteira;
    o grafico mostra a mediana e as faixas pedidas.

    Mostrar as faixas em vez de barras de erro simetricas e deliberado -
    distribuicoes de risco sao assimetricas, e barra simetrica esconde
    exatamente o lado que interessa.
    """
    if len(series) != len(labels):
        raise ValueError("um rotulo por serie")
    if len(series) < 2:
        raise ValueError("a tendencia precisa de ao menos duas distribuicoes")
    x = list(range(len(series)))
    limpas = [np.asarray(s, dtype=float)[np.isfinite(np.asarray(s, dtype=float))]
              for s in series]
    if any(s.size == 0 for s in limpas):
        raise ValueError("ha serie sem nenhum valor finito")

    fig = go.Figure()
    alphas = np.linspace(0.12, 0.30, len(bandas))
    for (lo, hi), alpha in zip(bandas, alphas):
        y_lo = [float(np.percentile(s, lo)) for s in limpas]
        y_hi = [float(np.percentile(s, hi)) for s in limpas]
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=y_hi + y_lo[::-1],
                fill="toself",
                fillcolor=f"rgba(37, 99, 235, {alpha:.2f})",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"P{lo:g}-P{hi:g}",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[float(np.median(s)) for s in limpas],
            mode="lines+markers",
            name="mediana",
            line=dict(color=COR["principal"], width=3),
        )
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=x, ticktext=list(labels)),
        yaxis_title="valor",
    )
    return _base(fig, titulo, 420)


def box_plot(
    series: Sequence[np.ndarray],
    labels: Sequence[str],
    titulo: str = "Box plot resumido",
) -> go.Figure:
    """Caixas lado a lado, uma por distribuicao.

    Complementa a tendencia: a tendencia enfatiza a evolucao, o box plot
    enfatiza a comparacao. Os bigodes vao ate P5 e P95, nao ate 1,5x o
    intervalo interquartil - a regra de 1,5x marcaria como "atipica" uma
    fracao enorme das iteracoes numa saida de cauda pesada, que e o caso
    normal em analise de risco, nao a excecao.
    """
    if len(series) != len(labels):
        raise ValueError("um rotulo por serie")
    fig = go.Figure()
    for j, (s, nome) in enumerate(zip(series, labels)):
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        q = np.percentile(s, [5, 25, 50, 75, 95])
        fig.add_trace(
            go.Box(
                q1=[q[1]], median=[q[2]], q3=[q[3]],
                lowerfence=[q[0]], upperfence=[q[4]],
                mean=[float(s.mean())],
                name=nome,
                marker_color=PALETA[j % len(PALETA)],
                boxmean=True,
            )
        )
    fig.update_layout(yaxis_title="valor", showlegend=False)
    return _base(fig, titulo, 420)


def overlay(
    series: Sequence[np.ndarray],
    labels: Sequence[str],
    bins: int = 80,
    cumulativa: bool = False,
    titulo: str = "Sobreposicao",
) -> go.Figure:
    """Varias distribuicoes num eixo so, com binning COMUM.

    O binning comum e o ponto todo. Histogramas construidos com intervalos
    proprios nao sao comparaveis: a mesma distribuicao com 40 e com 80
    intervalos tem alturas diferentes, e sobrepor dois desses compara altura
    de barra com altura de barra sem que signifiquem a mesma coisa. Aqui as
    bordas sao calculadas sobre a uniao das series.

    Densidade, nao contagem, para que series de tamanhos diferentes possam
    dividir o eixo.
    """
    if len(series) != len(labels):
        raise ValueError("um rotulo por serie")
    limpas = [np.asarray(s, dtype=float)[np.isfinite(np.asarray(s, dtype=float))]
              for s in series]
    limpas = [s for s in limpas if s.size]
    if not limpas:
        raise ValueError("nenhuma serie com valores finitos")
    lo = min(float(s.min()) for s in limpas)
    hi = max(float(s.max()) for s in limpas)
    if hi <= lo:
        hi = lo + 1.0
    bordas = np.linspace(lo, hi, bins + 1)
    centros = (bordas[:-1] + bordas[1:]) / 2.0

    fig = go.Figure()
    for j, (s, nome) in enumerate(zip(limpas, labels)):
        cor = PALETA[j % len(PALETA)]
        if cumulativa:
            xs = np.sort(s)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=np.arange(1, xs.size + 1) / xs.size,
                    mode="lines",
                    name=nome,
                    line=dict(color=cor, width=2),
                )
            )
        else:
            dens, _ = np.histogram(s, bins=bordas, density=True)
            fig.add_trace(
                go.Scatter(
                    x=centros, y=dens, mode="lines", name=nome,
                    line=dict(color=cor, width=2), fill="tozeroy",
                    opacity=0.35,
                )
            )
    fig.update_layout(
        xaxis_title="valor",
        yaxis_title="P(X <= x)" if cumulativa else "densidade",
    )
    return _base(fig, titulo, 420)


def convergencia(
    trilha_n: Sequence[int],
    trilha_valor: Sequence[float],
    trilha_meia: Sequence[float],
    tolerancia: float,
    convergiu_em: int | None,
    nome: str,
) -> go.Figure:
    """Estimativa com banda de confianca e a faixa de tolerancia pedida.

    O grafico responde a pergunta que a media acumulada sozinha nao responde:
    a banda ja CABE na tolerancia? O ponto em que a banda entra na faixa e o
    momento em que iteracoes adicionais deixam de comprar precisao.
    """
    n = np.asarray(trilha_n, dtype=float)
    v = np.asarray(trilha_valor, dtype=float)
    m = np.asarray(trilha_meia, dtype=float)
    final = float(v[-1])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([n, n[::-1]]),
            y=np.concatenate([v + m, (v - m)[::-1]]),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=n, y=v, mode="lines", name=nome,
                   line=dict(color=COR["principal"], width=2))
    )
    for sinal in (-1, 1):
        fig.add_hline(
            y=final * (1 + sinal * tolerancia),
            line_dash="dash",
            line_color=COR["ok"],
        )
    if convergiu_em is not None:
        fig.add_vline(
            x=float(convergiu_em),
            line_dash="dot",
            line_color=COR["alerta"],
            annotation_text=f"convergiu em {convergiu_em:,}",
        )
    fig.update_layout(xaxis_title="iteracoes", yaxis_title=nome)
    return _base(fig, f"Convergencia: {nome}", 360)
