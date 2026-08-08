# mcrisk — Dashboard de Simulação de Monte Carlo para Análise de Risco

Dashboard em Python/Streamlit para análise quantitativa de risco por simulação
de Monte Carlo. Substitui valores fixos por distribuições de probabilidade e
propaga a incerteza até o resultado — o mesmo princípio do add-in @RISK, aqui
implementado de forma aberta, auditável e testada.

> **Sem afiliação.** Este projeto não é afiliado, patrocinado nem validado pela
> Lumivero (proprietária do @RISK). @RISK é marca de seus respectivos titulares.
> Trata-se de uma reimplementação independente de técnicas publicadas na
> literatura estatística, todas referenciadas.

---

## O que faz

| Recurso | Implementação |
|---|---|
| Distribuições | 21 no registro (Normal, Lognormal em 2 parametrizações, Triangular, PERT, Uniforme, Beta, Gama, Exponencial, Weibull, t de Student, Logística, Gumbel, Pareto, GPD, Bernoulli, Binomial, Poisson, Binomial Negativa, Geométrica, Uniforme Discreta) + discreta customizada e reamostragem empírica |
| Amostragem | Monte Carlo simples e Latin Hypercube (McKay et al., 1979) |
| Correlação | Iman-Conover (1982) com correção arcsin; reparo de matriz não-PSD por Higham (2002) |
| Fórmula de saída | Expressão livre, avaliada por *parsing* AST com lista branca — **sem `eval` cru** |
| Estatísticas | Percentis com IC não paramétrico, VaR, CVaR, P(X≤t), erro de simulação |
| Sensibilidade | Correlação de posto e SRRC, gráfico tornado, com R² e VIF reportados |
| Ajuste a dados | MLE, AIC/AICc/BIC, pesos de Akaike, K-S e Anderson-Darling com **p-valor por bootstrap paramétrico** |
| Exportação | CSV das iterações, relatório Excel, especificação JSON reprodutível |

**O app não vem com números pré-preenchidos.** Todo valor exibido vem de algo
que você digitou ou de um arquivo que você carregou.

---

## Instalação

```bash
git clone https://github.com/jmagomez/mcrisk-dashboard.git
cd mcrisk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Requer Python 3.10+. Abra `http://localhost:8501`.

---

## Uso em 4 passos

1. **Aba 1 — Variáveis.** Adicione cada entrada incerta, escolha a distribuição
   e informe os parâmetros. Cada variável ganha um nome usável na fórmula.
2. **Aba 2 — Correlação.** Se as entradas não forem independentes, informe a
   matriz de correlação de Spearman. O app valida a consistência interna.
3. **Aba 3 — Modelo.** Escreva a fórmula de saída, por exemplo
   `(preco - custo_unitario) * volume - custo_fixo`. Rode a simulação.
4. **Aba 4 — Resultados.** Distribuição, percentis, medidas de cauda,
   tornado de sensibilidade e exportação.

A **aba 5** ajusta distribuições a dados históricos; a **aba 6** traz
metodologia e referências completas.

---

## Uso como biblioteca

O motor é independente da interface:

```python
import numpy as np
from mcrisk.engine import SimulationSpec, Variable, run
from mcrisk import summary, sensitivity

spec = SimulationSpec(
    variables=[
        Variable("preco", "Preço", "pert",
                 {"minimo": 10, "moda": 12, "maximo": 20, "lam": 4.0}),
        Variable("volume", "Volume", "lognormal_real",
                 {"media": 1000, "desvio": 200}),
    ],
    formula="preco * volume",
    iterations=50_000,
    method="lhs",
    seed=42,
    correlation=np.array([[1.0, -0.6], [-0.6, 1.0]]),  # Spearman
)

res = run(spec)
print(summary.describe(res.output))
print(sensitivity.analyze(res.inputs, res.output, res.labels).as_records())
```

---

## Três decisões metodológicas, com o efeito medido

Este repositório evita afirmações não verificadas. As três escolhas abaixo
foram medidas por testes automatizados, e os números são reprodutíveis com
`pytest`.

### 1. Correção arcsin no Iman-Conover

O algoritmo original controla a correlação de **Pearson** dos escores normais,
mas o alvo declarado é o **Spearman**. Para escores normais vale
`rho_S = (6/π)·arcsin(rho_P/2)`, então miramos `rho_P = 2·sin(π·rho_S/6)`.

| | Erro médio absoluto vs. alvo |
|---|---|
| Com correção (padrão) | **0,0010** |
| Sem correção | 0,0144 |

Teste: `tests/test_correlation.py::test_correcao_arcsin_melhora_a_recuperacao_do_alvo`

### 2. p-valor por bootstrap paramétrico no ajuste

Quando os parâmetros são estimados dos mesmos dados, a distribuição nula
tabelada do K-S deixa de valer. Sob H₀ **verdadeira**, um teste calibrado deve
produzir p-valores ~U(0,1) e rejeitar em ~5% dos casos ao nível de 5%:

| | p-valor médio | Taxa de rejeição a 5% |
|---|---|---|
| K-S ingênuo (parâmetros estimados tratados como conhecidos) | 0,77 | **0,0%** |
| Bootstrap paramétrico (implementado) | 0,50 | **8,8%** |

O teste ingênuo praticamente nunca rejeita — aceita ajustes ruins. Por isso o
app **não reporta** o p-valor assintótico do K-S.

Teste: `tests/test_fitting.py::test_pvalor_bootstrap_e_aproximadamente_uniforme_sob_h0`

### 3. Erro de simulação sob LHS

O LHS reduz muito o erro do estimador da média — no modelo de teste
(soma de normais, n=400, 200 replicações), o erro médio absoluto foi
**0,0916 com Monte Carlo** contra **0,0016 com LHS** (~56× menor).

Mas isso vem ao custo da independência entre iterações, e o erro padrão
clássico `s/√n` deixa de ser válido (Stein, 1987). O app avisa disso e oferece
**replicações independentes** como forma correta de medir o erro.

---

## Testes

```bash
pytest -q            # 210 testes
```

Duas camadas, com propósitos diferentes:

**168 testes de motor** (`tests/test_distributions.py`, `test_sampling.py`,
`test_correlation.py`, `test_formula.py`, `test_engine_and_stats.py`,
`test_fitting.py`). Cobrem momentos teóricos vs. amostrais das 21
distribuições, monotonicidade das ppf, preservação exata das marginais sob
Iman-Conover, recuperação da correlação alvo, ganho de variância do LHS,
ausência de viés, cobertura empírica dos intervalos de confiança, recuperação
de parâmetros no ajuste, calibração dos testes de aderência e 15 vetores de
ataque contra o avaliador de fórmulas.

**42 testes de interface** (`tests/test_ui.py`). Dirigem o app de ponta a ponta
com `streamlit.testing.v1.AppTest`: clicam nos botões, preenchem os campos,
rodam a simulação e conferem os números que aparecem na tela contra a solução
analítica. Também verificam que as ressalvas metodológicas chegam ao usuário —
não basta estarem documentadas aqui.

> **Por que essa segunda camada existe.** Um teste que apenas carregava o app
> vazio passava sem problemas. Ao dirigir a interface de verdade, os testes
> encontraram um travamento: duas variáveis com a mesma distribuição e os
> mesmos parâmetros geravam prévias idênticas, o Streamlit derivava o mesmo ID
> automático para os dois elementos e a página inteira caía. É um cenário
> banal — dois custos iguais, duas atividades iguais. Corrigido, com teste de
> regressão que falha na versão anterior.

---

## Estrutura

```
mcrisk/
  distributions.py   registro de distribuições, ppf, momentos teóricos
  sampling.py        Monte Carlo e Latin Hypercube
  correlation.py     Iman-Conover, reparo PSD, conversões Pearson↔Spearman
  formula.py         avaliador seguro de fórmulas (AST + lista branca)
  engine.py          orquestração da simulação
  summary.py         estatísticas de saída e erro de simulação
  sensitivity.py     índices de sensibilidade e tornado
  fitting.py         MLE, AIC/BIC, K-S e A-D com bootstrap
app.py               interface Streamlit
tests/               210 testes (168 de motor + 42 de interface)
```

---

## Antes de usar em decisão real

Leia **[LIMITATIONS.md](LIMITATIONS.md)**. Ele lista o que este app não faz, os
modos de falha conhecidos e — mais importante — por que a maior fonte de erro
em análise de risco quantitativa não está no código, e sim nas premissas que
você fornece.

A metodologia completa está em **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Licença

MIT — veja [LICENSE](LICENSE).
