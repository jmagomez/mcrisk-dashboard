# Metodologia

Descrição completa dos métodos implementados, das escolhas feitas e das
justificativas. Cada afirmação empírica neste documento corresponde a um teste
automatizado reprodutível com `pytest`.

---

## 1. Arquitetura da simulação

```
   n × k valores em (0,1)          [sampling.py]
              ↓
   ppf de cada marginal            [distributions.py]
              ↓
   reordenação Iman-Conover        [correlation.py]   (opcional)
              ↓
   avaliação vetorizada da fórmula [formula.py]
              ↓
   estatísticas e diagnósticos     [summary.py, sensitivity.py]
```

**Decisão central: tudo passa por transformada inversa.** Cada marginal é
gerada aplicando sua função quantil `F⁻¹(u)` a valores uniformes. Isso tem três
consequências que sustentam o resto do desenho:

1. O esquema de amostragem (MC, LHS) fica **completamente separado** das
   distribuições — trocar um não afeta o outro.
2. Como `F⁻¹` é monótona não decrescente, a **estrutura de postos** do cubo
   unitário é preservada nas marginais. É isso que faz Iman-Conover funcionar.
3. O custo é que distribuições sem quantil fechado ficam mais lentas.

Teste: `test_distributions.py::test_ppf_e_monotona_nao_decrescente`

---

## 2. Amostragem

### Monte Carlo simples

`u_ij ~ U(0,1)` independentes. As iterações são i.i.d., o TCL vale e o erro
padrão da média é `s/√n`.

### Latin Hypercube Sampling (LHS)

Para cada dimensão `j` independentemente, o intervalo (0,1) é dividido em `n`
estratos de igual probabilidade; sorteia-se um ponto em cada estrato e a ordem
é permutada aleatoriamente:

```
u_ij = (π_j(i) + ξ_ij) / n,    ξ_ij ~ U(0,1)
```

com `π_j` uma permutação aleatória **independente por dimensão** — usar a mesma
permutação em todas as colunas introduziria correlação espúria. Há teste
específico para esse bug: `test_sampling.py::test_colunas_do_lhs_sao_permutadas_independentemente`.

**Ganho medido.** Para a média de uma PERT com n=500, ao longo de 300
replicações, a variância do estimador sob LHS ficou abaixo de 25% da variância
sob MC. No nível da simulação completa (soma de normais, n=400, 200
replicações), o erro médio absoluto foi 0,0916 (MC) contra 0,0016 (LHS).

**Custo.** As iterações deixam de ser independentes. Consequências tratadas
explicitamente no código:

- `s/√n` deixa de ser estimador válido do erro da média (Stein, 1987). Sob LHS
  ele tende a **superestimar** o erro real.
- O gráfico de convergência (média acumulada) perde interpretação formal,
  porque a ordem das iterações é arbitrária. Fica como diagnóstico qualitativo.
- A forma correta de medir o erro é por **replicações independentes**: cada
  simulação completa, com semente distinta, é uma observação i.i.d. da
  estatística de interesse (`summary.replicate_summary`).

**Referências:** McKay, Beckman & Conover (1979); Stein (1987); Helton & Davis
(2003).

---

## 3. Correlação entre entradas — Iman-Conover

### Algoritmo

Dada uma amostra `X` (n × k) e uma matriz alvo `C` de correlação de Spearman:

1. Monta `M` (n × k), cada coluna uma permutação aleatória dos escores de
   van der Waerden `a_i = Φ⁻¹(i/(n+1))`.
2. `E = corr(M)`; Cholesky `E = F Fᵀ`.
3. Cholesky `C = P Pᵀ`.
4. `M* = M (F⁻¹)ᵀ Pᵀ`, de modo que `corr(M*) ≈ C`.
5. Reordena cada coluna de `X` para que seus postos coincidam com os de `M*`.

Como o passo 5 **apenas reordena**, as marginais são preservadas exatamente —
o conjunto de valores de cada coluna é idêntico ao original. Essa é a
propriedade que torna o método *distribution-free*, e ela é verificada por
igualdade exata de arrays ordenados em
`test_correlation.py::test_marginais_sao_preservadas_exatamente`.

### A correção arcsin

Os passos 2–4 controlam a correlação de **Pearson** dos escores normais, mas o
alvo declarado é o **Spearman** da amostra final. Para a normal bivariada vale
a relação de Kruskal (1958):

```
rho_S = (6/π) · arcsin(rho_P / 2)     ⟺     rho_P = 2 · sin(π · rho_S / 6)
```

Mirar `rho_P = C` diretamente produz Spearman final sistematicamente **abaixo**
do alvo. Aplicamos a inversa antes da fatoração de Cholesky.

**Medição (10 replicações, n = 20.000, 3 marginais não normais):**

| | Erro médio absoluto vs. alvo |
|---|---|
| Com correção (`spearman_adjust=True`, padrão) | **0,0010** |
| Sem correção | 0,0144 |

O teste correspondente falha se a correção deixar de ajudar — ou seja, o padrão
do código está protegido por evidência, não por opinião.

### Matrizes inconsistentes

Uma matriz de correlação precisa ser positiva semidefinida; nem toda
combinação de coeficientes que "parece razoável" é atingível. Ex.: A e B
fortemente positivos, B e C fortemente positivos, A e C fortemente negativos é
impossível.

O app detecta isso pelo menor autovalor e, se o usuário prosseguir, projeta na
matriz de correlação PSD mais próxima pelo método de projeções alternadas de
**Higham (2002)**, avisando que as correlações efetivas diferirão das pedidas.

### O que este método não dá

Correlação de posto **não determina a distribuição conjunta**. A estrutura
imposta é a induzida por escores normais — na prática próxima de uma cópula
gaussiana, cuja **dependência de cauda é zero**. Ver `LIMITATIONS.md` §3.

---

## 4. Avaliação da fórmula

A fórmula vem de um campo de texto livre. Usar `eval()` cru seria execução de
código arbitrário. A expressão é compilada com `ast.parse` e validada contra
uma lista branca de nós (`BinOp`, `UnaryOp`, `Name`, `Constant`, `Call`,
`Compare`, operadores aritméticos e de comparação) e de funções
(`se`, `min`, `max`, `clip`, `abs`, `exp`, `log`, `sqrt`, `desconto`, …).

Bloqueados na etapa de parsing, antes de qualquer execução: acesso a atributos,
indexação, lambdas, comprehensions, atribuições, literais não numéricos e
qualquer chamada fora da lista branca. A avaliação ocorre com
`__builtins__` vazio.

São 15 vetores de ataque testados em `test_formula.py`, e mais 6 repetidos
pela interface em `test_ui.py` — porque bloquear no motor não basta se a
tela não exibir o erro e continuar permitindo rodar.

A avaliação é **vetorizada**: a expressão é calculada uma vez sobre arrays de
tamanho n, não n vezes sobre escalares.

---

## 5. Estatísticas de saída

- **Percentis:** `numpy.percentile` (interpolação linear entre estatísticas de
  ordem).
- **IC para percentis:** não paramétrico, via estatísticas de ordem. O número
  de observações abaixo do quantil populacional segue Binomial(n, p); os
  limites são as estatísticas de ordem correspondentes (Conover, 1999). Não
  assume forma da distribuição, mas assume iterações i.i.d.
- **IC para a média:** normal via TCL. Inválido sob LHS e sob variância
  infinita.
- **VaR:** percentil α (ou 1−α, conforme a orientação da perda).
- **CVaR / Expected Shortfall:** média condicional além do VaR.

O app reporta VaR e CVaR juntos porque o VaR **não é uma medida coerente de
risco** — não satisfaz subaditividade (Artzner et al., 1999).

**Validação por cobertura empírica:** os testes verificam que ~95% dos
intervalos de 95% contêm o valor verdadeiro, ao longo de 200 replicações
independentes, tanto para a média quanto para o percentil 95.

---

## 6. Sensibilidade

Dois índices, ambos calculados sobre a amostra já existente (sem avaliações
adicionais do modelo):

1. **Correlação de posto (Spearman)** entre cada entrada e a saída. Índice
   *marginal*: ignora que as entradas possam estar correlacionadas entre si.
2. **Coeficiente de regressão de posto padronizado (SRRC).** Regressão linear
   múltipla dos postos da saída sobre os postos das entradas, padronizados.
   Índice *condicional*: controla as demais entradas.

Com entradas independentes os dois praticamente coincidem. Quando divergem, a
divergência é informação sobre a estrutura de dependência.

### Diagnósticos reportados junto

- **R² da regressão de postos.** Mede quanto do comportamento do modelo esses
  índices monotônicos conseguem explicar. Abaixo de 0,7 o app avisa que o
  tornado pode ordenar mal as variáveis e sugere índices de Sobol. Caso de
  teste explícito: para `a * b` com `a`, `b` simétricos, o R² fica abaixo de
  0,5 e o aviso dispara — verificado tanto no motor quanto na tela.
- **VIF máximo** (fator de inflação de variância). Acima de 10 — regra de bolso
  de Belsley, Kuh & Welsch (1980) — o app avisa que os SRRC ficam instáveis e a
  atribuição de importância entre variáveis colineares é ambígua por
  construção.
- **Variáveis constantes** na amostra são reportadas com sensibilidade 0 e
  listadas explicitamente, em vez de produzirem NaN silencioso.

**Referências:** Helton & Davis (2002); Saltelli & Sobol' (1995); Saltelli et
al. (2008).

---

## 7. Ajuste de distribuições a dados

### Estimação

Máxima verossimilhança via `scipy.stats.<dist>.fit`. Para distribuições de
suporte positivo com dados estritamente positivos, `loc` é fixado em 0 — deixar
`loc` livre tipicamente colapsa o ajuste no mínimo amostral. É uma escolha, e
está documentada como tal.

Candidatas com suporte positivo são automaticamente excluídas quando há dados
não positivos.

### Seleção de modelo

- `AIC  = 2k − 2·logL`  (Akaike, 1974)
- `AICc = AIC + 2k(k+1)/(n−k−1)`  (correção para amostra pequena)
- `BIC  = k·ln(n) − 2·logL`  (Schwarz, 1978)
- **Pesos de Akaike:** `w_i = exp(−Δ_i/2) / Σ exp(−Δ_j/2)`

O ranking usa AICc. Os pesos são apresentados como **evidência relativa dentro
do conjunto testado** — se todas as candidatas forem ruins, o peso alto da
primeira não significa que ela seja boa (Burnham & Anderson, 2002, §2.9). O app
diz isso na interface.

### Aderência — e o ponto metodológico principal

Estatísticas calculadas:

- **Kolmogorov-Smirnov:** `D = max|F_n(x) − F(x)|`
- **Anderson-Darling:** `A² = −n − (1/n)·Σ(2i−1)[ln F(x_i) + ln(1 − F(x_{n+1−i}))]`

O A² pesa mais as caudas, o que é preferível em análise de risco: é justamente
na cauda que o modelo importa.

**O problema.** Quando os parâmetros são estimados a partir dos **mesmos dados**
usados no teste, as distribuições nulas tabeladas de K-S e A-D deixam de valer.
O ajuste "puxa" a distribuição na direção da amostra, reduzindo a estatística e
inflando o p-valor.

**A solução implementada.** p-valor por **bootstrap paramétrico** (procedimento
de Lilliefors generalizado): simula B amostras do modelo ajustado, **reajusta o
modelo em cada uma** e compara a estatística observada com a distribuição
empírica resultante. O reajuste dentro do laço é essencial — é ele que
reproduz o encolhimento causado pela estimação. Usa-se o estimador com correção
`(#{≥ obs} + 1)/(B + 1)`, que evita p-valor exatamente zero (Davison & Hinkley,
1997).

**Medição.** Sob H₀ verdadeira (dados realmente normais, ajustando normal), um
teste calibrado deve produzir p-valores ~U(0,1) e rejeitar em ~5% dos casos ao
nível de 5%:

| | p-valor médio | Rejeição a 5% |
|---|---|---|
| K-S ingênuo | 0,77 | **0,0%** |
| Bootstrap paramétrico | 0,50 | **8,8%** |

O teste ingênuo praticamente nunca rejeita — aceita ajustes ruins. Por isso o
app **não reporta** o p-valor assintótico do K-S, mesmo sendo trivial de
calcular e comum em ferramentas comerciais.

Poder do teste também verificado: dados exponenciais ajustados por uma normal
são rejeitados com p < 0,05.

**Referências:** Lilliefors (1967); Stephens (1974); Babu & Rao (2004);
Davison & Hinkley (1997).

### Q-Q plot

Posições de plotagem de Hazen, `p_i = (i − 0,5)/n`. A interface instrui a olhar
especialmente as **pontas**: desvios no centro quase não afetam a decisão,
desvios na cauda mudam completamente o VaR.

---

## 8. Reprodutibilidade

Toda a aleatoriedade vem de um único `numpy.random.Generator` (PCG64)
construído a partir da semente informada. Mesma semente + mesma especificação
produz resultado idêntico bit a bit — há teste com `np.array_equal` no motor e,
na interface, um teste que confere que duas execuções exibem exatamente os
mesmos números na tela.

As replicações usam sementes `seed + r·7919` (primo) para afastar os fluxos.

A exportação em JSON registra distribuições, parâmetros, matriz de correlação,
fórmula, método de amostragem, número de iterações e semente — o suficiente
para um terceiro reproduzir o resultado. Não registra as versões de NumPy/SciPy;
fixe-as via `requirements.txt` se precisar de reprodutibilidade estrita entre
máquinas.

---

## 9. Cobertura de testes

210 testes, em duas camadas.

### Camada 1 — motor (168 testes)

| Arquivo | O que garante |
|---|---|
| `test_distributions.py` | Momentos amostrais vs. teóricos para as 21 distribuições; monotonicidade das ppf; consistência entre as duas parametrizações da lognormal; fórmula clássica da média PERT; rejeição de parâmetros inválidos; variância infinita detectada na Pareto |
| `test_sampling.py` | Ocupação de todos os estratos sob LHS; independência entre colunas; redução de variância medida; ausência de viés; reprodutibilidade |
| `test_correlation.py` | Preservação exata das marginais; recuperação do alvo; superioridade medida da correção arcsin; detecção e reparo de matrizes não-PSD |
| `test_formula.py` | Correção aritmética; 15 vetores de ataque bloqueados; sanitização de nomes |
| `test_engine_and_stats.py` | Soma de normais vs. solução analítica; efeito da correlação na variância; **cobertura empírica** dos ICs; ganho do LHS; identificação da variável dominante com contribuições de variância conhecidas; disparo dos avisos de R² baixo e VIF alto |
| `test_fitting.py` | Recuperação de parâmetros; AIC prefere o modelo gerador; K-S e A-D conferidos contra o SciPy; **calibração do p-valor** sob H₀; poder do teste |

### Camada 2 — interface (42 testes)

`test_ui.py` dirige o aplicativo de ponta a ponta com
`streamlit.testing.v1.AppTest`: clica em "Adicionar variável", preenche
rótulos e parâmetros, escreve a fórmula, clica em "Rodar simulação" e lê as
métricas renderizadas. Garante que:

- os números **exibidos na tela** batem com a solução analítica (média,
  desvio, P5/P50/P95, VaR, CVaR), e não apenas os retornados pelo motor;
- as ressalvas metodológicas aparecem para o usuário — o aviso de que `s/√n`
  não vale sob LHS, o aviso de Sobol quando o R² é baixo, o aviso de que a
  reamostragem empírica não extrapola;
- entradas inválidas produzem mensagem de erro e **bloqueiam** o botão de
  rodar, em vez de derrubar a página;
- tentativas de injeção de código na fórmula são barradas na interface;
- os três botões de exportação aparecem após a simulação.

**Por que essa camada existe.** Um teste que apenas carregava o app vazio
passava. Ao dirigir a interface de verdade, os testes encontraram um
travamento real: duas variáveis com a mesma distribuição e os mesmos
parâmetros produziam gráficos de prévia idênticos, o Streamlit derivava o
mesmo ID automático para os dois elementos e levantava
`StreamlitDuplicateElementId`, derrubando a página. O cenário é banal — dois
custos iguais, duas atividades iguais. O teste de regressão
(`test_duas_variaveis_identicas_nao_derrubam_o_app`) foi verificado contra a
versão anterior do código, onde falha.

---

## Referências completas

Ver a seção "Referências" da aba 6 do aplicativo, ou o `README.md`. As
principais:

- Akaike, H. (1974). IEEE Trans. Automatic Control 19(6):716-723.
- Artzner, P. et al. (1999). *Coherent Measures of Risk*. Math. Finance 9(3):203-228.
- Babu, G.J. & Rao, C.R. (2004). *Goodness-of-fit tests when parameters are estimated*. Sankhyā 66(1):63-74.
- Belsley, D.A., Kuh, E. & Welsch, R.E. (1980). *Regression Diagnostics*. Wiley.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*, 2ª ed. Springer.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Conover, W.J. (1999). *Practical Nonparametric Statistics*, 3ª ed. Wiley.
- Davison, A.C. & Hinkley, D.V. (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Helton, J.C. & Davis, F.J. (2002). *Illustration of Sampling-Based Methods for Uncertainty and Sensitivity Analysis*. Risk Analysis 22(3):591-622.
- Helton, J.C. & Davis, F.J. (2003). *Latin hypercube sampling and the propagation of uncertainty*. Reliab. Eng. Syst. Saf. 81(1):23-69.
- Higham, N.J. (2002). *Computing the nearest correlation matrix*. IMA J. Numer. Anal. 22(3):329-343.
- Iman, R.L. & Conover, W.J. (1982). *A distribution-free approach to inducing rank correlation among input variables*. Comm. Statist. Simulation Comput. 11(3):311-334.
- Kruskal, W.H. (1958). *Ordinal Measures of Association*. JASA 53:814-861.
- Lilliefors, H.W. (1967). JASA 62(318):399-402.
- Malcolm, D.G. et al. (1959). *Application of a Technique for R&D Program Evaluation*. Operations Research 7(5):646-669.
- McKay, M.D., Beckman, R.J. & Conover, W.J. (1979). Technometrics 21(2):239-245.
- Saltelli, A. & Sobol', I.M. (1995). Reliab. Eng. Syst. Saf. 50(3):225-239.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Schwarz, G. (1978). Annals of Statistics 6(2):461-464.
- Stein, M. (1987). Technometrics 29(2):143-151.
- Stephens, M.A. (1974). JASA 69(347):730-737.
- Vose, D. (2008). *Risk Analysis: A Quantitative Guide*, 3ª ed. Wiley.
