"""
Mede qualidade, velocidade e confiabilidade. Gera os numeros do BENCHMARK.md.

Rode com:  python tools/benchmark.py

Nenhum numero do BENCHMARK.md e digitado a mao: todos saem daqui. Se um deles
estiver errado, a correcao e mudar o codigo ou o metodo -- nao o documento.

As medicoes de TEMPO dependem da maquina e nao devem ser comparadas entre
maquinas diferentes. O que se transporta e a RAZAO entre elas, que e o que as
tabelas destacam.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrisk import convergence as conv  # noqa: E402
from mcrisk import copula, correlation, fitting, sensitivity, summary  # noqa: E402
from mcrisk.engine import SimulationSpec, Variable, run  # noqa: E402


def cron(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return time.perf_counter() - t0, out


def _normais(k: int, n: int, **kw) -> SimulationSpec:
    vs = [
        Variable(f"v{j}", f"V{j}", "normal", {"mu": 10.0, "sigma": 2.0})
        for j in range(k)
    ]
    return SimulationSpec(
        variables=vs, formula=" + ".join(f"v{j}" for j in range(k)),
        iterations=n, seed=12345, **kw,
    )


# ---------------------------------------------------------------------------
# 1. Qualidade: erro contra solucao fechada
# ---------------------------------------------------------------------------


def qualidade_soma_de_normais(n: int = 200_000) -> list[dict]:
    """Soma de 3 N(10,2) => N(30, 2*sqrt(3)). Tudo tem valor exato."""
    mu, sg = 30.0, 2.0 * np.sqrt(3.0)
    linhas = []
    for metodo in ("mc", "lhs"):
        r = run(_normais(3, n, method=metodo))
        y = r.output
        d = summary.describe(y)
        alvo = {
            "media": (float(d["media"]), mu),
            "desvio": (float(d["desvio"]), sg),
            "P5": (float(d["P5"]), float(stats.norm.ppf(0.05, mu, sg))),
            "P95": (float(d["P95"]), float(stats.norm.ppf(0.95, mu, sg))),
            "P99": (float(d["P99"]), float(stats.norm.ppf(0.99, mu, sg))),
            "semi-desvio": (summary.semi_std(y), sg / np.sqrt(2.0)),
            "DAM": (
                summary.mean_absolute_deviation(y), sg * np.sqrt(2.0 / np.pi)
            ),
            "curtose Pearson": (summary.kurtosis_pearson(y), 3.0),
        }
        for nome, (obtido, exato) in alvo.items():
            linhas.append({
                "metodo": metodo, "estatistica": nome,
                "obtido": obtido, "exato": exato,
                "erro_rel_pct": abs(obtido - exato) / abs(exato) * 100.0,
            })
    return linhas


def qualidade_sensibilidade(n: int = 100_000) -> list[dict]:
    """y = 3a + b + 0c com entradas independentes: 90% / 10% / 0% da variancia."""
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(n, 3))
    y = 3 * X[:, 0] + X[:, 1]
    cv = sensitivity.contribution_to_variance(X, y, list("abc"))
    exato = [0.9, 0.1, 0.0]
    linhas = [
        {"variavel": c, "obtido": float(cv.fracao[j]), "exato": exato[j],
         "erro_abs": abs(float(cv.fracao[j]) - exato[j])}
        for j, c in enumerate("abc")
    ]
    # Ponto cego dos metodos de posto, medido em vez de afirmado.
    y2 = X[:, 0] ** 2
    posto = sensitivity.analyze(X, y2, list("abc"))
    cond = sensitivity.change_in_output_statistic(X, y2, list("abc"), bins=10)
    linhas.append({
        "variavel": "a em y = a^2 (relacao em U)",
        "obtido": float(abs(posto.spearman[0])),
        "exato": float("nan"),
        "erro_abs": float("nan"),
        "spearman": float(posto.spearman[0]),
        "srrc": float(posto.srrc[0]),
        "swing_condicional": float(cond.swing[0]),
        "swing_das_irrelevantes": float(max(cond.swing[1], cond.swing[2])),
    })
    return linhas


def qualidade_copulas(n: int = 200_000) -> list[dict]:
    rng = np.random.default_rng(4242)
    linhas = []
    for fam in copula.ARQUIMEDIANAS:
        for alvo in (0.3, 0.5, 0.7):
            th_tau = copula.theta_from_tau(fam, alvo)
            u = copula.archimedean_copula_u(fam, th_tau, n, 2, rng)
            tau_obt = float(stats.kendalltau(u[:, 0], u[:, 1]).statistic)

            th_rho = copula.theta_from_spearman(fam, alvo)
            u2 = copula.archimedean_copula_u(fam, th_rho, n, 2, rng)
            rho_obt = float(stats.spearmanr(u2[:, 0], u2[:, 1]).statistic)

            li, ls = copula.tail_dependence_archimedean(fam, th_tau)
            li_emp = copula.empirical_tail_dependence(1 - u[:, 0], 1 - u[:, 1], 0.99)
            ls_emp = copula.empirical_tail_dependence(u[:, 0], u[:, 1], 0.99)
            linhas.append({
                "familia": fam, "alvo": alvo,
                "theta_por_tau": th_tau, "tau_obtido": tau_obt,
                "erro_tau": abs(tau_obt - alvo),
                "theta_por_rho": th_rho, "rho_obtido": rho_obt,
                "erro_rho": abs(rho_obt - alvo),
                "lambda_inf_teorico": li, "lambda_inf_empirico": li_emp,
                "lambda_sup_teorico": ls, "lambda_sup_empirico": ls_emp,
            })
    return linhas


def qualidade_ajuste_de_copula(n: int = 4_000, repeticoes: int = 20) -> list[dict]:
    """Com que frequencia a comparacao por AIC escolhe a familia geradora?"""
    linhas = []
    for fam in copula.ARQUIMEDIANAS:
        theta = copula.theta_from_tau(fam, 0.45)
        acertos = 0
        for s in range(repeticoes):
            u = copula.archimedean_copula_u(
                fam, theta, n, 2, np.random.default_rng(1000 + s)
            )
            venc = copula.fit_copula(u[:, 0], u[:, 1])[0]
            acertos += int(venc.familia == fam)
        linhas.append({
            "familia_geradora": fam, "n": n, "repeticoes": repeticoes,
            "acertos": acertos, "taxa": acertos / repeticoes,
        })
    return linhas


def qualidade_incerteza_de_parametro() -> list[dict]:
    """Variancia da preditiva contra a solucao bayesiana exata da normal."""
    linhas = []
    for n_dados in (20, 40, 100, 1_000):
        x = stats.norm(0.0, 1.0).rvs(n_dados, random_state=65)
        inc = fitting.parameter_uncertainty(
            x, "normal", replicas=1_000, rng=np.random.default_rng(66)
        )
        N = 400_000
        pontual = float(np.var(x, ddof=0))
        crua = fitting.simular_com_incerteza(
            inc, N, np.random.default_rng(68), corrigir_vies=False
        ).var(ddof=1)
        corr = fitting.simular_com_incerteza(
            inc, N, np.random.default_rng(68)
        ).var(ddof=1)
        s2 = float(np.var(x, ddof=1))
        exata = s2 * (1 + 1 / n_dados) * (n_dados - 1) / (n_dados - 3)
        linhas.append({
            "n_dados": n_dados,
            "var_pontual": pontual,
            "var_bootstrap_cru": float(crua),
            "var_bootstrap_corrigido": float(corr),
            "var_exata_bayesiana": exata,
            "fracao_da_distancia_fechada": (
                (float(corr) - pontual) / (exata - pontual)
                if exata != pontual else float("nan")
            ),
        })
    return linhas


def qualidade_convergencia() -> list[dict]:
    """A projecao de iteracoes bate com a formula fechada?"""
    linhas = []
    rng = np.random.default_rng(31415)
    for tol in (0.05, 0.01, 0.005):
        y = rng.normal(100.0, 20.0, 400_000)
        obtido = conv.iteracoes_para_tolerancia(y, tol, 0.95)
        z = float(stats.norm.ppf(0.975))
        exato = np.ceil((z * y.std(ddof=1) / (tol * abs(y.mean()))) ** 2)
        linhas.append({
            "tolerancia": tol, "projetado": float(obtido),
            "formula_fechada": float(exato),
            "erro_abs": abs(float(obtido) - float(exato)),
        })
    return linhas


# ---------------------------------------------------------------------------
# 2. Velocidade
# ---------------------------------------------------------------------------


def velocidade_simulacao(n: int = 200_000) -> list[dict]:
    C = np.full((3, 3), 0.5)
    np.fill_diagonal(C, 1.0)
    run(_normais(3, 20_000))  # aquece
    linhas = []
    casos = [("sem dependencia", dict(method="lhs"))]
    casos += [("iman_conover", dict(method="lhs", correlation=C))]
    for dep in ("gaussian", "t") + copula.ARQUIMEDIANAS:
        casos.append((dep, dict(correlation=C, dependence=dep)))
    for nome, kw in casos:
        t, r = cron(run, _normais(3, n, **kw))
        linhas.append({
            "esquema": nome, "iteracoes": n, "segundos": t,
            "us_por_iteracao": t / n * 1e6,
        })
    return linhas


def velocidade_analises(n: int = 200_000) -> list[dict]:
    C = np.full((3, 3), 0.4)
    np.fill_diagonal(C, 1.0)
    r = run(_normais(3, n, method="lhs", correlation=C))
    nomes = r.labels
    linhas = []
    for nome, fn in (
        ("sensibilidade: posto + SRRC",
         lambda: sensitivity.analyze(r.inputs, r.output, nomes)),
        ("sensibilidade: condicional (10 faixas)",
         lambda: sensitivity.change_in_output_statistic(
             r.inputs, r.output, nomes, bins=10)),
        ("sensibilidade: contribuicao para a variancia",
         lambda: sensitivity.contribution_to_variance(r.inputs, r.output, nomes)),
        ("correlacao obtida (Spearman k x k)",
         lambda: correlation.achieved_spearman(r.inputs)),
        ("convergencia: 3 estatisticas, 40 pontos",
         lambda: conv.monitor(r.output, ("media", "desvio", "p95"),
                              passo=n // 40)),
        ("estatisticas descritivas completas",
         lambda: summary.describe(r.output)),
    ):
        t, _ = cron(fn)
        linhas.append({"analise": nome, "n": n, "segundos": t})
    x = stats.lognorm(0.6, scale=np.exp(3)).rvs(500, random_state=7)
    t, _ = cron(fitting.fit_many, x, ["normal", "lognormal", "gamma", "weibull"])
    linhas.append({"analise": "ajuste de 4 familias (n=500, sem bootstrap)",
                   "n": 500, "segundos": t})
    t, _ = cron(fitting.parameter_uncertainty, x, "lognormal", 200,
                np.random.default_rng(1))
    linhas.append({"analise": "incerteza de parametro (200 replicas, n=500)",
                   "n": 500, "segundos": t})
    u = copula.archimedean_copula_u("clayton", 2.0, 5_000, 2,
                                    np.random.default_rng(2))
    t, _ = cron(copula.fit_copula, u[:, 0], u[:, 1])
    linhas.append({"analise": "ajuste de 3 copulas (n=5.000)", "n": 5_000,
                   "segundos": t})
    return linhas


def velocidade_escala() -> list[dict]:
    linhas = []
    for n in (25_000, 50_000, 100_000, 200_000, 400_000):
        t, _ = cron(run, _normais(3, n, method="lhs"))
        linhas.append({"iteracoes": n, "segundos": t,
                       "us_por_iteracao": t / n * 1e6})
    return linhas


# ---------------------------------------------------------------------------
# 3. Confiabilidade
# ---------------------------------------------------------------------------


def confiabilidade_cobertura(repeticoes: int = 300, n: int = 4_000) -> list[dict]:
    """Um IC de 95% precisa conter o valor verdadeiro em ~95% das rodadas."""
    mu, sg = 30.0, 2.0 * np.sqrt(3.0)
    p95_exato = float(stats.norm.ppf(0.95, mu, sg))
    linhas = []
    for metodo in ("mc", "lhs"):
        acertos_media = acertos_p95 = 0
        for s in range(repeticoes):
            spec = _normais(3, n, method=metodo)
            spec.seed = 700_000 + s
            y = run(spec).output
            lo, hi = summary.mean_ci(y)
            acertos_media += int(lo <= mu <= hi)
            lo2, hi2 = summary.quantile_ci(y, 0.95)
            acertos_p95 += int(lo2 <= p95_exato <= hi2)
        linhas.append({
            "metodo": metodo, "repeticoes": repeticoes, "n": n,
            "cobertura_IC_media": acertos_media / repeticoes,
            "cobertura_IC_P95": acertos_p95 / repeticoes,
            "nominal": 0.95,
        })
    return linhas


def confiabilidade_criterio_de_convergencia(
    repeticoes: int = 200, tol: float = 0.02
) -> list[dict]:
    """O criterio promete tolerancia `tol` com 95% de confianca. Cumpre?

    Para cada rodada: encontra o n em que o criterio declara convergencia e
    verifica se a estimativa NAQUELE ponto esta de fato dentro de `tol` do
    valor verdadeiro. E o teste que separa um criterio calibrado de um numero
    decorativo.
    """
    mu = 30.0
    dentro = 0
    ns = []
    for s in range(repeticoes):
        spec = _normais(3, 40_000, method="mc")
        spec.seed = 900_000 + s
        y = run(spec).output
        rel = conv.monitor(y, ("media",), tolerancia=tol, confianca=0.95,
                           passo=250, metodo_amostragem="mc")
        m = rel.convergiu_em["media"]
        if m is None:
            continue
        ns.append(m)
        est = float(np.mean(y[:m]))
        dentro += int(abs(est - mu) <= tol * abs(mu))
    return [{
        "tolerancia": tol, "confianca_declarada": 0.95,
        "repeticoes": repeticoes, "convergiram": len(ns),
        "fracao_dentro_da_tolerancia": dentro / len(ns) if ns else float("nan"),
        "iteracoes_medianas_ate_convergir": float(np.median(ns)) if ns else float("nan"),
    }]


def confiabilidade_reprodutibilidade() -> list[dict]:
    linhas = []
    C = np.full((3, 3), 0.5)
    np.fill_diagonal(C, 1.0)
    for dep in ("iman_conover", "gaussian", "t") + copula.ARQUIMEDIANAS:
        kw = dict(correlation=C) if dep == "iman_conover" else dict(
            correlation=C, dependence=dep
        )
        a = run(_normais(3, 20_000, **kw)).output
        b = run(_normais(3, 20_000, **kw)).output
        linhas.append({
            "esquema": dep,
            "identico_bit_a_bit": bool(np.array_equal(a, b)),
        })
    return linhas


def confiabilidade_ganho_do_lhs(repeticoes: int = 200, n: int = 2_000) -> list[dict]:
    mu = 30.0
    out = {}
    for metodo in ("mc", "lhs"):
        erros = []
        for s in range(repeticoes):
            spec = _normais(3, n, method=metodo)
            spec.seed = 500_000 + s
            erros.append(abs(float(np.mean(run(spec).output)) - mu))
        out[metodo] = float(np.mean(erros))
    return [{
        "repeticoes": repeticoes, "n": n,
        "erro_medio_absoluto_mc": out["mc"],
        "erro_medio_absoluto_lhs": out["lhs"],
        "razao_mc_sobre_lhs": out["mc"] / out["lhs"],
    }]


def confiabilidade_dependencia_obtida(n: int = 100_000) -> list[dict]:
    """Quanto a dependencia efetiva difere da pedida, por esquema."""
    C = np.array([[1.0, 0.6, 0.6], [0.6, 1.0, 0.6], [0.6, 0.6, 1.0]])
    linhas = []
    for dep in ("iman_conover", "gaussian", "t") + copula.ARQUIMEDIANAS:
        kw = dict(correlation=C) if dep == "iman_conover" else dict(
            correlation=C, dependence=dep
        )
        r = run(_normais(3, n, **kw))
        emax, emed, _ = correlation.correlation_error(r.inputs, C)
        linhas.append({
            "esquema": dep, "rho_pedido": 0.6,
            "erro_maximo": emax, "erro_medio": emed,
        })
    return linhas


# ---------------------------------------------------------------------------


def main() -> None:
    resultados = {
        "ambiente": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "plataforma": platform.platform(),
            "processador": platform.processor() or "desconhecido",
        },
        "qualidade_soma_de_normais": qualidade_soma_de_normais(),
        "qualidade_sensibilidade": qualidade_sensibilidade(),
        "qualidade_copulas": qualidade_copulas(),
        "qualidade_ajuste_de_copula": qualidade_ajuste_de_copula(),
        "qualidade_incerteza_de_parametro": qualidade_incerteza_de_parametro(),
        "qualidade_convergencia": qualidade_convergencia(),
        "velocidade_simulacao": velocidade_simulacao(),
        "velocidade_analises": velocidade_analises(),
        "velocidade_escala": velocidade_escala(),
        "confiabilidade_cobertura": confiabilidade_cobertura(),
        "confiabilidade_criterio_de_convergencia":
            confiabilidade_criterio_de_convergencia(),
        "confiabilidade_reprodutibilidade": confiabilidade_reprodutibilidade(),
        "confiabilidade_ganho_do_lhs": confiabilidade_ganho_do_lhs(),
        "confiabilidade_dependencia_obtida": confiabilidade_dependencia_obtida(),
    }
    print(json.dumps(resultados, indent=1, default=float))


if __name__ == "__main__":
    main()
