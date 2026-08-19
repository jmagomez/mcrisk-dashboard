# mcrisk — Dashboard de Simulação de Monte Carlo para Análise de Risco

[![tests](https://github.com/jmagomez/mcrisk-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/jmagomez/mcrisk-dashboard/actions/workflows/tests.yml)
[![e2e](https://github.com/jmagomez/mcrisk-dashboard/actions/workflows/e2e.yml/badge.svg)](https://github.com/jmagomez/mcrisk-dashboard/actions/workflows/e2e.yml)

Dashboard em Python para análise quantitativa de risco por simulação de Monte
Carlo. Substitui valores fixos por distribuições de probabilidade e propaga a
incerteza até o resultado — o mesmo princípio do add-in @RISK, aqui
implementado de forma aberta, auditável e testada.

### ▶ [Abrir no navegador, sem instalar nada](https://jmagomez.github.io/mcrisk-dashboard/)

A versão web roda o **mesmo pacote Python** deste repositório dentro do
navegador, via Pyodide/WebAssembly — os números não podem divergir da versão
testada. Nenhum dado sai da sua máquina. A primeira carga baixa cerca de 30 MB
(Python + NumPy + SciPy) e leva de 20 segundos a alguns minutos; depois fica em
cache.

> **Sem afiliação.** Este projeto não é afiliado, patrocinado nem validado pela
> Lumivero (proprietária do @RISK). @RISK é marca de seus respectivos titulares.
> Trata-se de uma reimplementação independente de técnicas publicadas na
> literatura estatística, todas referenciadas.

---

## Duas interfaces, mesmo motor

| | Versão web | Versão Streamlit |
|---|---|---|
| Acesso | [link direto](https://jmagomez.github.io/mcrisk-dashboard/), sem instalar | `streamlit run app.py` |
| Motor | idêntico (`mcrisk/`, via Pyodide) | idêntico (`mcrisk/`) |
| Variáveis, correlação, fórmula, resultados | sim | sim |
| Prévia das marginais, tornado, dispersão, convergência | sim | sim |
| Ajuste de distribuições a dados | sim | sim |
| Replicações independentes | sim | sim |
| Importar especificação (JSON) | sim | não |
| Exportação | CSV, JSON | CSV, JSON, Excel |
| Velocidade | limitada pelo WebAssembly | nativa |

Para modelos grandes (centenas de milhares de iterações, ou replicações com
muitas réplicas), use a versão Streamlit: o Pyodide roda na mesma thread da
interface, e a aba fica travada durante o cálculo.

---

## O que faz

| Recurso | Implementação |
|---|---|
| Distribuições | 21 no registro (Normal, Lognormal em 2 parametrizações, Triangular, PERT, Uniforme, Beta, Gama, Exponencial, Weibull, t de Student, Logística, Gumbel, Pareto, GPD, Bernoulli, Binomial, Poisson, Binomial Negativa, Geométrica, Uniforme Discreta) + discreta customizada e reamostragem empírica |
| Amostragem | Monte Carlo simples e Latin Hypercube (McKay et al., 1979) |
| Correlação | Iman-Conover (1982) com correção arcsin; reparo de matriz não-PSD por Higham (2002) |
| Dependência de cauda | Cópulas Gaussiana e t de Student, como alternativa ao Iman-Conover — só a t produz extremos simultâneos |
| Cenários | Condicional (recorta a amostra existente) e estresse (re-simula com outra distribuição de entrada) |
| Fórmula de saída | Expressão livre, avaliada por *parsing* AST com lista branca — **sem `eval` cru** |
| Estatísticas | Percentis com IC não paramétrico, VaR, CVaR, P(X≤t), erro de simulação |
| Sensibilidade | Correlação de posto e SRRC, gráfico tornado, com R² e VIF reportados |
| Ajuste a dados | MLE, AIC/AICc/BIC, pesos de Akaike, K-S e Anderson-Darling com **p-valor por bootstrap paramétrico** |
| Exportação | CSV das iterações, relatório Excel, especificação JSON reprodutível |

**O app não vem com números pré-preenchidos.** Todo valor exibido vem de algo
que você digitou ou de um arquivo que você carregou.

---

## Instalação da versão completa

```bash
git clone https://github.com/jmagomez/mcrisk-dashboard.git
cd mcrisk-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Requer Python 3.10+ e **Streamlit 1.51+** (o app usa `width="stretch"` em
`st.plotly_chart`, que só existe a partir dessa versão). Abre em
`http://localhost:8501`.

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

Para dependência de cauda em vez de reordenação por postos, acrescente
`dependence="t"` e `copula_df=5.0` à `SimulationSpec` — a mesma matriz de
correlação, com extremos que ocorrem juntos.

---

## Três decisões metodológicas, com o efeito medido

Este repositório evita afirmações não verificadas. As escolhas abaixo foram
medidas por testes automatizados, e os números são reprodutíveis com `pytest`.

### 1. Correção arcsin no Iman-Conover

O algoritmo original controla a correlação de **Pearson** dos escores normais,
mas o alvo declarado é o **Spearman**. Para escores normais vale
`rho_S = (6/π)·arcsin(rho_P/2)`, então miramos `rho_P = 2·sin(π·rho_S/6)`.

| | Erro médio absoluto vs. alvo |
|---|---|
| Com correção (padrão) | **0,0010** |
| Sem correção | 0,0144 |

### 2. p-valor por bootstrap paramétrico no ajuste

Quando os parâmetros são estimados dos mesmos dados, a distribuição nula
tabelada do K-S deixa de valer. Sob H₀ **verdadeira**, um teste calibrado deve
produzir p-valores ~U(0,1) e rejeitar em ~5% dos casos ao nível de 5%:

| | p-valor médio | Taxa de rejeição a 5% |
|---|---|---|
| K-S ingênuo | 0,77 | **0,0%** |
| Bootstrap paramétrico (implementado) | 0,50 | **8,8%** |

### 3. Erro de simulação sob LHS

O LHS reduz muito o erro do estimador da média — no modelo de teste
(soma de normais, n=400, 200 replicações), o erro médio absoluto foi
**0,0916 com Monte Carlo** contra **0,0016 com LHS** (~56× menor).

Mas isso vem ao custo da independência entre iterações, e o erro padrão
clássico `s/√n` deixa de ser válido (Stein, 1987). O app avisa disso e oferece
**replicações independentes** como forma correta de medir o erro.

---

## Defeitos encontrados em auditoria, e corrigidos

Uma revisão linha a linha e os testes de interface encontraram seis defeitos
reais. Todos têm teste de regressão.

1. **Correlação dividida pela metade, em silêncio.** A tela pedia "preencha
   apenas um triângulo", e o código fazia `(C + C.T)/2` — o que transforma o
   0,8 digitado em 0,4. A simulação rodava, exibia "matriz válida e
   internamente consistente", e usava metade da correlação pedida. Corrigido
   com `mirror_triangle`, que espelha o lado preenchido e **avisa** quando os
   dois lados conflitam.
2. **Duas variáveis idênticas derrubavam a página.** Prévias com a mesma
   distribuição e os mesmos parâmetros geravam gráficos idênticos, e o
   Streamlit derivava o mesmo ID automático para os dois elementos.
3. **O resultado do ajuste sumia da tela.** O bloco dependia do retorno de
   `st.button`, então qualquer rerun o apagava — inclusive o do próprio
   seletor "Inspecionar ajuste", que ficava inutilizável.
4. **`achieved_spearman` devolvia matriz 1×1 com exatamente duas variáveis**,
   porque `scipy.stats.spearmanr` retorna escalar nesse caso. Passou
   despercebido porque todos os testes anteriores usavam três variáveis.
5. **Na versão web, não dava para digitar mais de um caractere por campo.**
   Cada evento `input` reconstruía a lista de variáveis com `innerHTML`,
   destruindo o próprio campo em uso.
6. **Parâmetros com valor padrão apareciam preenchidos mas não eram
   registrados.** O campo `lambda` da PERT mostrava `4` e o app dizia
   "parâmetro ausente: lam". Como PERT é a distribuição padrão, isso travava
   a primeira coisa que qualquer pessoa tenta.

O primeiro é o mais instrutivo: não quebrava nada. Só produzia números
plausíveis e errados — exatamente o modo de falha que o `LIMITATIONS.md`
argumenta ser o mais perigoso em análise de risco.

Os defeitos 5 e 6 foram relatados por usuário, não pelos testes. Eram da
versão de navegador, que na época não tinha suíte automatizada. Essa lacuna
está fechada — ver abaixo.

---

## Testes

### Motor e interface Streamlit — `pytest`

```bash
pytest -q            # 425 casos, a partir de 250 funções de teste
```

As duas contagens respondem a perguntas diferentes: **250** é quanto código de
teste existe para ler e manter; **425** é quantas situações são efetivamente
verificadas, porque `@pytest.mark.parametrize` multiplica uma função em vários
casos. Dois casos ficam fora da execução padrão por serem marcados `slow`
(1M de iterações e Iman-Conover 500k × 6); rode-os com `pytest -m slow`.

**347 de motor** — momentos teóricos vs. amostrais das 21 distribuições,
monotonicidade das ppf, preservação exata das marginais sob Iman-Conover,
recuperação da correlação alvo, dependência de cauda das cópulas conferida
contra a fórmula fechada, ganho de variância do LHS, ausência de viés,
cobertura empírica dos intervalos de confiança, recuperação de parâmetros no
ajuste, calibração dos testes de aderência, parâmetros degenerados e matrizes
impossíveis, forma do crescimento do custo, e 15 vetores de ataque contra o
avaliador de fórmulas.

**52 de interface** (`tests/test_ui.py`) — dirigem o app de ponta a ponta com
`streamlit.testing.v1.AppTest` e conferem os números exibidos na tela contra a
solução analítica.

**26 de auditoria** (`tests/test_auditoria.py`) — os defeitos 1 a 4 acima,
cada um verificado contra a versão defeituosa antes da correção.

### Versão de navegador — `Playwright`

```bash
cd tests-e2e && npm install && npx playwright install --with-deps chromium && npm test
```

**13 testes de navegador** (`tests-e2e/dashboard.spec.js`), rodando contra a
árvore de trabalho num servidor local. Cobrem digitação, prévia das marginais,
resultados contra solução analítica, correlação, validação de entradas,
ajuste a dados, importação e exportação.

A regra que define esta suíte: **`pressSequentially`, nunca `fill`**, em
qualquer campo cujo comportamento a cada tecla importe. Os defeitos 5 e 6
escaparam porque a verificação anterior definia `.value` e disparava um único
evento — o que nunca reproduz digitação. Um evento não reconstrói o DOM no
meio da palavra; treze eventos reconstroem.

---

## Estrutura

```
mcrisk/
  distributions.py   registro de distribuições, ppf, momentos teóricos
  sampling.py        Monte Carlo e Latin Hypercube
  correlation.py     Iman-Conover, espelhamento, reparo PSD
  copula.py          cópulas Gaussiana e t, dependência de cauda
  scenarios.py       cenários condicionais e de estresse
  formula.py         avaliador seguro de fórmulas (AST + lista branca)
  engine.py          orquestração da simulação
  summary.py         estatísticas de saída e erro de simulação
  sensitivity.py     índices de sensibilidade e tornado
  fitting.py         MLE, AIC/BIC, K-S e A-D com bootstrap
app.py               interface Streamlit (completa)
index.html           interface de navegador (Pyodide), publicada no GitHub Pages
tests/               425 casos de motor e da interface Streamlit
tests-e2e/           13 testes de navegador (Playwright)
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
