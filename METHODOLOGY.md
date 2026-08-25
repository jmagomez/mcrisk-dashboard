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
   dependência: Iman-Conover       [correlation.py]   (opcional)
   ou uma das cinco cópulas        [copula.py]
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

O mesmo pacote `mcrisk/` alimenta as duas interfaces: o app Streamlit
(`app.py`) e a versão de navegador (`index.html`, via Pyodide/WebAssembly).
Nenhuma linha do motor é reescrita em JavaScript — precisamente para que os
números das duas não possam divergir da suíte de testes.

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

### Como uma matriz preenchida pela metade é lida

A interface pede que o usuário preencha **apenas um dos triângulos**. Isso
levanta a questão de como completar o outro lado, e a resposta ingênua está
errada de um jeito perigoso.

Tirar a média com o lado vazio — `(C + C.T)/2` — divide por dois toda
correlação digitada: quem pede 0,8 recebe 0,4. A simulação roda, a matriz passa
em todas as validações (continua simétrica, diagonal 1, positiva semidefinida),
e o resultado sai plausível e errado. Nada avisa.

`correlation.mirror_triangle` aplica, para cada par (i, j):

| Situação | Resultado |
|---|---|
| só um lado preenchido | espelha o lado preenchido |
| os dois iguais | usa o valor |
| os dois preenchidos e diferentes | usa o de cima e **reporta conflito** |
| nenhum preenchido | 0 |

Um zero explícito é indistinguível de "não preenchi" numa grade numérica. A
escolha é tratar zero como "não preenchi", que é o caso comum; quem quiser
fixar zero de verdade preenche os dois lados com zero, e nenhum conflito é
reportado. A ambiguidade genuína — dois valores diferentes — nunca é resolvida
em silêncio.

### Matrizes inconsistentes

Uma matriz de correlação precisa ser positiva semidefinida; nem toda
combinação de coeficientes que "parece razoável" é atingível. Ex.: A e B
fortemente positivos, B e C fortemente positivos, A e C fortemente negativos é
impossível.

O app detecta isso pelo menor autovalor e, se o usuário prosseguir, projeta na
matriz de correlação PSD mais próxima pelo método de projeções alternadas de
**Higham (2002)**, avisando que as correlações efetivas diferirão das pedidas.

### Verificação do que saiu

Depois de simular, a aba de resultados mostra uma tabela **pedida vs. obtida**
por par de variáveis. Iman-Conover atinge o alvo de forma aproximada, e o
reparo PSD pode afastar ainda mais o resultado do pedido — o usuário precisa
poder conferir, não confiar.

### O que este método não dá

Correlação de posto **não determina a distribuição conjunta**. A estrutura
imposta é a induzida por escores normais — na prática próxima de uma cópula
gaussiana, cuja **dependência de cauda é zero**. Quem precisa de extremos
simultâneos escolhe uma cópula na aba de correlação: a **t** para cauda
simétrica nos dois lados, **Clayton** para só a inferior, **Gumbel** para só a
superior. O que elas resolvem e o que continua por conta do usuário está em
`LIMITATIONS.md` §3, e o erro de calibração de cada uma no `BENCHMARK.md`.

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

### Uma armadilha do SciPy

`scipy.stats.spearmanr` devolve uma **matriz** quando recebe três ou mais
colunas, e um **escalar** quando recebe exatamente duas. `achieved_spearman`
normaliza os dois casos para uma matriz k × k. Sem isso, qualquer chamador que
indexe `[i, j]` quebra no caso de duas variáveis — que foi exatamente o que
aconteceu quando a tabela "pedida vs. obtida" foi implementada. O defeito
sobreviveu à suíte original porque todos os testes de correlação usavam três
variáveis.

---

## 6. Sensibilidade

Quatro métodos, todos calculados sobre a amostra já existente (nenhum exige
avaliações adicionais do modelo). Não existe o "melhor": eles respondem a
perguntas diferentes e discordam quando o modelo tem estrutura que um deles
não enxerga — e a discordância é o achado.

Os dois primeiros são baseados em posto:

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

### Os outros dois métodos

3. **Mudança na estatística da saída.** Ordena as iterações pelo valor da
   entrada, divide em faixas **equiprováveis** (por contagem, não por largura —
   dividir por largura esvaziaria as faixas de cauda em qualquer entrada
   assimétrica) e mede a amplitude da estatística da saída entre as faixas.
   Não supõe forma nenhuma, e é o **único dos quatro que enxerga relação não
   monótona**. Medido no repositório com `y = a²`: Spearman e SRRC devolvem
   −0,0022 para a única variável que importa; o swing condicional devolve
   3,2344 contra 0,0811 das irrelevantes. Em troca é marginal — não controla
   as demais entradas — e depende da escolha do número de faixas.

4. **Contribuição para a variância**, por soma de quadrados **sequencial**.
   Responde outra pergunta: que fração da variância da saída cada entrada
   explica. Seleção para a frente; o incremento de R² de cada entrada é a
   contribuição dela. Duas propriedades que o resultado carrega explicitamente,
   porque são onde este método engana: as frações somam o **R² total**, não
   1 — o que sobra não pertence a entrada nenhuma (com `y = a·b`, medido:
   99,98% não explicada) — e, com entradas correlacionadas, quem entra antes
   na regressão fica com a parte compartilhada, o que torna a atribuição
   ambígua por construção. A ordem de entrada é reportada.

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

544 casos de teste, a partir de 336 funções, em três camadas.

As duas contagens medem coisas diferentes e as duas são úteis: **336** é quanto
código de teste existe para ler e manter; **544** é quantas situações são de
fato verificadas, porque `@pytest.mark.parametrize` multiplica uma função em
vários casos — a varredura sobre as 21 distribuições, por exemplo, é uma função
só e 21 casos. Dois casos ficam fora da execução padrão por serem marcados
`slow`; rode-os com `pytest -m slow`.

### Camada 1 — motor (448 casos)

| Arquivo | O que garante |
|---|---|
| `test_distributions.py` | Momentos amostrais vs. teóricos para as 21 distribuições; monotonicidade das ppf; consistência entre as duas parametrizações da lognormal; fórmula clássica da média PERT; rejeição de parâmetros inválidos; variância infinita detectada na Pareto |
| `test_sampling.py` | Ocupação de todos os estratos sob LHS; independência entre colunas; redução de variância medida; ausência de viés; reprodutibilidade |
| `test_correlation.py` | Preservação exata das marginais; recuperação do alvo; superioridade medida da correção arcsin; detecção e reparo de matrizes não-PSD |
| `test_formula.py` | Correção aritmética; 15 vetores de ataque bloqueados; sanitização de nomes |
| `test_engine_and_stats.py` | Soma de normais vs. solução analítica; efeito da correlação na variância; **cobertura empírica** dos ICs; ganho do LHS; identificação da variável dominante com contribuições de variância conhecidas; disparo dos avisos de R² baixo e VIF alto |
| `test_fitting.py` | Recuperação de parâmetros; AIC prefere o modelo gerador; K-S e A-D conferidos contra o SciPy; **calibração do p-valor** sob H₀; poder do teste |
| `test_copula_cenarios.py` | Dependência de cauda da cópula t conferida contra a fórmula fechada λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))); λ = 0 na Gaussiana; conversão Spearman→Pearson; cenário condicional preserva as probabilidades do modelo; estresse re-simula e reporta delta |
| `test_validacao_estatistica.py` | Testes de aderência das marginais geradas contra a distribuição alvo; calibração sob H₀; comparação de momentos de ordem alta |
| `test_robustez.py` | Parâmetros degenerados recusados **com mensagem**; matrizes impossíveis detectadas e o reparo **anunciado**; fórmulas patológicas; dados malformados; varredura aleatória de 80 modelos |
| `test_desempenho.py` | Forma do crescimento do custo em N e em k (não o tempo absoluto); memória da amostra; escala de 1M de iterações |
| `test_metodos_risco.py` | Contribuição para a variância recupera a decomposição exata 90/10/0; o método condicional enxerga `y = a²` onde os de posto devolvem −0,002; significância de cenário troca de sinal com a cauda; projeção de convergência bate com a fórmula fechada; τ de Kendall das três arquimedianas; ajuste de cópula recupera a família geradora; preditiva com incerteza de parâmetro contra a solução bayesiana exata |

### Camada 2 — interface (70 testes)

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
- o esquema de dependência está visível, tem Iman-Conover como padrão, e a
  cópula t exibe o coeficiente de dependência de cauda contra o zero da
  Gaussiana;
- a tabela de correlação **obtida** mostra o que de fato vigorou na simulação,
  conferida numericamente contra o alvo;
- o cenário de estresse reporta um delta que bate com a solução analítica;
- os três botões de exportação aparecem após a simulação.

### Camada 3 — auditoria (26 testes)

`test_auditoria.py` cobre quatro defeitos encontrados numa revisão linha a
linha do código. Cada um foi executado **contra a versão defeituosa** antes da
correção, para confirmar que o teste realmente pega o problema. Um teste de bug
que já passa no código defeituoso não vale nada.

| Defeito | Como se manifestava | Por que passou despercebido |
|---|---|---|
| Correlação dividida por dois | `(C + C.T)/2` sobre uma matriz preenchida em um triângulo só. Pedir 0,8 usava 0,4 | Não quebrava nada. A matriz continuava válida e o app exibia "internamente consistente" |
| Duas variáveis idênticas derrubavam a página | Prévias iguais → mesmo ID automático no Streamlit → `StreamlitDuplicateElementId` | O teste anterior só carregava o app vazio |
| Resultado do ajuste sumia da tela | Bloco dependia do retorno de `st.button`; qualquer rerun o apagava, inclusive o do próprio seletor "Inspecionar ajuste" | Nenhum teste interagia com a tela depois de clicar em "Ajustar" |
| `achieved_spearman` devolvia matriz 1×1 | `scipy.stats.spearmanr` retorna escalar com exatamente duas colunas | Todos os testes de correlação usavam três variáveis |

O primeiro é o mais instrutivo, e é a razão de esta camada existir. Os outros
três quebram de forma visível — alguém percebe. O primeiro produzia números
plausíveis e errados, em silêncio, exatamente o modo de falha que o
`LIMITATIONS.md` argumenta ser o mais perigoso em análise de risco. Foi
encontrado lendo o código e conferindo o que a legenda da tela prometia contra
o que a linha fazia, não rodando o app.

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
- Embrechts, P., McNeil, A. & Straumann, D. (2002). *Correlation and dependence in risk management: properties and pitfalls*. In: *Risk Management: Value at Risk and Beyond*, Cambridge University Press, 176-223.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Helton, J.C. & Davis, F.J. (2002). *Illustration of Sampling-Based Methods for Uncertainty and Sensitivity Analysis*. Risk Analysis 22(3):591-622.
- Helton, J.C. & Davis, F.J. (2003). *Latin hypercube sampling and the propagation of uncertainty*. Reliab. Eng. Syst. Saf. 81(1):23-69.
- Higham, N.J. (2002). *Computing the nearest correlation matrix*. IMA J. Numer. Anal. 22(3):329-343.
- Iman, R.L. & Conover, W.J. (1982). *A distribution-free approach to inducing rank correlation among input variables*. Comm. Statist. Simulation Comput. 11(3):311-334.
- Kruskal, W.H. (1958). *Ordinal Measures of Association*. JASA 53:814-861.
- Lilliefors, H.W. (1967). JASA 62(318):399-402.
- Malcolm, D.G. et al. (1959). *Application of a Technique for R&D Program Evaluation*. Operations Research 7(5):646-669.
- Marshall, A.W. & Olkin, I. (1988). *Families of Multivariate Distributions*. JASA 83(403):834-841.
- McKay, M.D., Beckman, R.J. & Conover, W.J. (1979). Technometrics 21(2):239-245.
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*, ed. revisada. Princeton University Press.
- Nelsen, R.B. (2006). *An Introduction to Copulas*, 2ª ed. Springer.
- Saltelli, A. & Sobol', I.M. (1995). Reliab. Eng. Syst. Saf. 50(3):225-239.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Schwarz, G. (1978). Annals of Statistics 6(2):461-464.
- Stein, M. (1987). Technometrics 29(2):143-151.
- Stephens, M.A. (1974). JASA 69(347):730-737.
- Vose, D. (2008). *Risk Analysis: A Quantitative Guide*, 3ª ed. Wiley.
