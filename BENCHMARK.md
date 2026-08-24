# Qualidade, velocidade e confiabilidade — medido

Nenhum número deste documento foi digitado à mão. Todos saem de
`tools/benchmark.py`, que pode ser executado por qualquer pessoa que clone o
repositório:

```bash
python tools/benchmark.py > medicoes.json
```

**Ambiente da execução registrada abaixo.** Python 3.10.12, NumPy 2.2.6,
SciPy 1.15.3, Linux x86-64. Os tempos absolutos dependem da máquina e **não**
devem ser comparados entre máquinas diferentes — o que se transporta é a
*razão* entre eles, que é o que as tabelas destacam.

---

## 1. Qualidade — erro contra a solução fechada

O modelo de teste é a soma de três N(10, 2) independentes, cuja saída é
exatamente N(30, 2√3). Nenhum valor esperado abaixo vem de simulação.

**200.000 iterações, semente 12345:**

| Estatística | Exato | Monte Carlo | erro | Latin Hypercube | erro |
|---|---|---|---|---|---|
| Média | 30,0000 | 29,9949 | 0,017% | 30,0000 | **0,000%** |
| Desvio-padrão | 3,4641 | 3,4571 | 0,203% | 3,4591 | 0,144% |
| P5 | 24,3021 | 24,3169 | 0,061% | 24,3003 | **0,007%** |
| P95 | 35,6979 | 35,6869 | 0,031% | 35,6776 | 0,057% |
| P99 | 38,0587 | 38,0290 | 0,078% | 38,0447 | 0,037% |
| Semi-desvio | 2,4495 | 2,4447 | 0,196% | 2,4477 | 0,073% |
| Desvio absoluto médio | 2,7639 | 2,7607 | 0,119% | 2,7566 | 0,266% |
| Curtose (Pearson) | 3,0000 | 2,9855 | 0,484% | 3,0163 | 0,543% |

O erro relativo fica **abaixo de 0,6% em todas as estatísticas**, e abaixo de
0,08% nas medidas de posição. A curtose é a de pior desempenho, como esperado:
é o quarto momento, e momentos altos convergem devagar.

### Contribuição para a variância

Com `y = 3a + b + 0·c` e entradas independentes, a decomposição exata é
90% / 10% / 0%. Medido com 100.000 iterações:

| Variável | Exato | Obtido | Erro absoluto |
|---|---|---|---|
| a | 0,9000 | 0,8998 | 0,0002 |
| b | 0,1000 | 0,1002 | 0,0002 |
| c | 0,0000 | 0,0000 | 0,0000 |

### O ponto cego que o quarto método cobre

Este é o número que justifica ter mais de um método de sensibilidade. Com
`y = a²`, a variável `a` é a **única** que importa — e os índices baseados em
posto não a enxergam, porque a relação não é monótona:

| Método | Valor para `a` | Enxerga? |
|---|---|---|
| Correlação de posto (Spearman) | −0,0022 | **não** |
| Regressão de posto (SRRC) | −0,0022 | **não** |
| Mudança na estatística da saída (swing) | 3,2344 | **sim** |
| — mesma medida nas variáveis irrelevantes | 0,0811 | (contraste de 40×) |

Um dashboard com apenas os dois primeiros métodos reportaria, com números
plausíveis e ordenação limpa, que nenhuma entrada importa.

### Projeção de iterações para convergência

Contra a fórmula fechada `n = (z·s / (tol·|média|))²`:

| Tolerância | Fórmula fechada | Projetado | Erro |
|---|---|---|---|
| 5% | 62 | 62 | 0 |
| 1% | 1.540 | 1.540 | 0 |
| 0,5% | 6.162 | 6.162 | 0 |

### Cópulas: parâmetro pedido contra dependência obtida

100.000 iterações. `τ` é calibrado por forma fechada; `ρ` de Spearman é
calibrado por interpolação de uma grade medida com semente fixa — e por isso
tem erro próprio, que aparece aqui.

| Família | Alvo | τ obtido | erro τ | ρ obtido | erro ρ |
|---|---|---|---|---|---|
| Clayton | 0,30 | 0,2985 | 0,0015 | 0,3053 | 0,0053 |
| Clayton | 0,50 | 0,5023 | 0,0023 | 0,5061 | 0,0061 |
| Clayton | 0,70 | 0,6991 | 0,0009 | 0,7002 | 0,0002 |
| Gumbel | 0,30 | 0,2993 | 0,0007 | 0,2974 | 0,0026 |
| Gumbel | 0,50 | 0,5020 | 0,0020 | 0,5022 | 0,0022 |
| Gumbel | 0,70 | 0,6992 | 0,0008 | 0,6997 | 0,0003 |
| Frank | 0,30 | 0,3025 | 0,0025 | 0,3001 | 0,0001 |
| Frank | 0,50 | 0,4981 | 0,0019 | 0,5065 | 0,0065 |
| Frank | 0,70 | 0,6999 | 0,0001 | 0,7077 | 0,0077 |

**Leitura:** o erro na calibração por τ fica em 0,0025 no pior caso; o erro na
calibração por ρ chega a 0,0077, cerca de três vezes maior. A diferença tem
causa conhecida: τ é convertido em θ por fórmula fechada (bisseção sobre a
função de Debye no caso da Frank), enquanto ρ é convertido por interpolação de
uma grade simulada, que carrega erro de amostragem próprio. Quem precisa do
alvo exato deve informar τ diretamente; quem usa a grade da interface, que é em
ρ, deve ler a tabela de **correlação obtida** e não a pedida.

### Dependência de cauda: fórmula fechada contra amostra

O mesmo experimento, com o λ empírico estimado no quantil 0,99:

| Família | τ | λ inferior teórico | empírico | λ superior teórico | empírico |
|---|---|---|---|---|---|
| Clayton | 0,30 | 0,445 | 0,441 | 0,000 | 0,019 |
| Clayton | 0,50 | 0,707 | 0,711 | 0,000 | 0,028 |
| Clayton | 0,70 | 0,862 | 0,855 | 0,000 | 0,056 |
| Gumbel | 0,30 | 0,000 | 0,057 | 0,375 | 0,392 |
| Gumbel | 0,50 | 0,000 | 0,143 | 0,586 | 0,586 |
| Gumbel | 0,70 | 0,000 | 0,327 | 0,769 | 0,765 |
| Frank | 0,30 | 0,000 | 0,030 | 0,000 | 0,034 |
| Frank | 0,50 | 0,000 | 0,056 | 0,000 | 0,053 |
| Frank | 0,70 | 0,000 | 0,112 | 0,000 | 0,111 |

Nos lados onde λ é positivo, o estimador empírico bate com a fórmula fechada
dentro de 0,02. Nos lados onde λ é **zero**, o estimador devolve valores
positivos que crescem com τ — 0,327 para a Gumbel com τ = 0,70. Isso é viés de
amostra finita, não falha da implementação: o estimador usa apenas 1% da
amostra e mede dependência de cauda *assintótica* num quantil finito. É a razão
de a implementação reportar as duas colunas, e de o teste da suíte verificar o
**contraste** entre as caudas em vez de exigir zero exato.

### Ajuste de cópula: a família geradora é recuperada?

Gera-se de uma família com τ = 0,45, ajustam-se as três, e verifica-se se o
AIC escolhe a correta. 20 repetições independentes por família, n = 4.000:

| Família geradora | Acertos | Taxa |
|---|---|---|
| Clayton | 20/20 | 100% |
| Gumbel | 20/20 | 100% |
| Frank | 20/20 | 100% |

Com n = 4.000 a discriminação é perfeita. Isso **não** se transporta para
amostras pequenas: a diferença entre as famílias está na cauda, que é onde há
menos observações.

### Incerteza de parâmetro contra a solução bayesiana exata

Para a normal com priori de Jeffreys, a preditiva é uma *t* de Student com
n−1 graus de liberdade e variância `s²·(1+1/n)·(n−1)/(n−3)`. Esta tabela é a
que mais mudou a implementação:

| n de dados | Var. pontual | Bootstrap **cru** | Bootstrap **corrigido** | Var. exata | distância fechada |
|---|---|---|---|---|---|
| 20 | 0,7037 | 0,7084 | 0,8123 | 0,8693 | 66% |
| 40 | 0,8043 | **0,7970** | 0,8720 | 0,8912 | 78% |
| 100 | 0,7515 | **0,7494** | 0,7718 | 0,7825 | 66% |
| 1.000 | 0,9937 | **0,9922** | 0,9951 | 0,9976 | 37% |

**O achado.** Reamostrar as réplicas do estimador de máxima verossimilhança
sem correção **estreita** a preditiva em três dos quatro casos — o oposto do
que a funcionalidade promete. A razão é exata e não é acidente de semente: o
MLE de um parâmetro de escala é viesado para baixo, e esse viés cancela quase
exatamente o alargamento que a incerteza de locação deveria produzir. Refletir
as réplicas em torno da estimativa (bootstrap básico) corrige o viés de
primeira ordem e fecha entre 37% e 78% da distância até o valor exato.

O que a última linha diz é igualmente importante: com 1.000 observações a
diferença entre ignorar e propagar a incerteza de parâmetro é de 0,4% na
variância. **Com dados suficientes, ignorá-la passa a ser defensável** — e o
número está aqui para que essa decisão seja tomada com evidência.

---

## 2. Velocidade

### Custo por esquema de dependência

200.000 iterações, 3 variáveis normais, na mesma máquina:

| Esquema | Segundos | µs/iteração | Razão vs. sem dependência |
|---|---|---|---|
| Sem dependência (LHS puro) | 0,033 | 0,164 | 1,00× |
| Cópula Gaussiana | 0,043 | 0,214 | 1,30× |
| Cópula Gumbel | 0,044 | 0,222 | 1,35× |
| Cópula Frank | 0,046 | 0,230 | 1,40× |
| Cópula Clayton | 0,048 | 0,242 | 1,47× |
| Iman-Conover | 0,094 | 0,469 | 2,85× |
| Cópula t de Student | 0,123 | 0,615 | 3,74× |

**Resultado contra-intuitivo, e verificável:** as cópulas arquimedianas são
cerca de **duas vezes mais rápidas que o Iman-Conover**. O Iman-Conover
precisa ordenar cada coluna duas vezes (`O(n log n)` por variável); a
amostragem por *frailty* de Marshall-Olkin é `O(n·d)` sem nenhuma ordenação.
A t é a mais cara porque soma uma qui-quadrado por iteração ao custo da
Gaussiana.

Nenhuma dessas diferenças é decisiva na prática: **o esquema mais caro custa
0,6 µs por iteração**, ou 0,6 segundos por milhão. A escolha entre eles deve
ser metodológica, não de desempenho.

### Custo das análises pós-simulação

Sobre uma amostra de 200.000 iterações e 3 variáveis:

| Análise | Segundos |
|---|---|
| Estatísticas descritivas completas | 0,021 |
| Correlação obtida (matriz k × k de Spearman) | 0,027 |
| Contribuição para a variância (soma de quadrados sequencial) | 0,045 |
| Sensibilidade condicional (10 faixas) | 0,063 |
| Convergência (3 estatísticas × 40 pontos de teste) | 0,107 |
| Sensibilidade: posto + SRRC | 0,117 |
| Ajuste de 3 cópulas por MLE (n = 5.000) | 0,011 |
| Ajuste de 4 famílias marginais (n = 500) | 0,007 |
| Incerteza de parâmetro, 200 réplicas de bootstrap (n = 500) | 0,007 |

Toda a bateria de diagnósticos custa menos de 0,4 s sobre 200.000 iterações —
menos que a própria simulação em qualquer esquema com dependência.

### Escala

| Iterações | Segundos | µs/iteração |
|---|---|---|
| 25.000 | 0,0045 | 0,179 |
| 50.000 | 0,0072 | 0,145 |
| 100.000 | 0,0145 | 0,145 |
| 200.000 | 0,0333 | 0,166 |
| 400.000 | 0,0851 | 0,213 |

Multiplicar o tamanho por 16 multiplicou o tempo por 19: o crescimento é
**linear**, não quadrático. O custo por iteração varia entre 0,145 e 0,213 µs
ao longo de toda a faixa, e a subida nas duas pontas tem causas diferentes —
em 25.000 o tempo fixo de montagem ainda pesa, e em 400.000 a matriz de
entradas deixa de caber no cache de último nível.

---

## 3. Confiabilidade

### Cobertura empírica dos intervalos de confiança

300 simulações independentes de 4.000 iterações cada. Um intervalo de 95%
honesto contém o valor verdadeiro em ~95% das rodadas:

| Método | Cobertura do IC da média | Cobertura do IC do P95 | Nominal |
|---|---|---|---|
| Monte Carlo | **95,7%** | **96,0%** | 95% |
| Latin Hypercube | **100,0%** | 98,0% | 95% |

Sob Monte Carlo a calibração é boa. Sob LHS a cobertura vai a 100%, o que
**não** é uma boa notícia: é a confirmação empírica de que `s/√n` deixa de ser
válido quando as iterações não são independentes, e que o erro é
sistematicamente **superestimado** (Stein, 1987). O app avisa disso na tela; a
medida acima é a evidência.

### O critério de convergência cumpre o que promete?

200 simulações independentes. Em cada uma, encontra-se o ponto em que o
critério declara convergência a 2% com 95% de confiança, e verifica-se se a
estimativa **naquele ponto** está de fato dentro de 2% do valor verdadeiro:

| Tolerância declarada | Rodadas | Convergiram | Dentro da tolerância | Mediana de iterações |
|---|---|---|---|---|
| 2% (95% de confiança) | 200 | 200 | **100%** | 250 |

O critério é **conservador**, não exatamente calibrado: promete 95% e entrega
100%. Duas causas, ambas conhecidas e nenhuma delas defeito — o teste ocorre
em pontos discretos (a cada 250 iterações aqui), o que faz o `n` efetivo
ultrapassar o mínimo necessário; e o próprio IC é de dois lados sobre uma
estatística cuja distribuição é bem comportada. Errar para o lado de rodar
iterações a mais é a direção certa do erro.

### Reprodutibilidade

Mesma semente, mesma especificação, duas execuções, comparação **bit a bit**
com `np.array_equal`:

| Esquema | Idêntico bit a bit |
|---|---|
| Iman-Conover | sim |
| Cópula Gaussiana | sim |
| Cópula t | sim |
| Cópula Clayton | sim |
| Cópula Gumbel | sim |
| Cópula Frank | sim |

### Dependência efetiva contra a pedida

ρ de Spearman alvo = 0,60 em todos os pares, 100.000 iterações, 3 variáveis:

| Esquema | Erro máximo | Erro médio |
|---|---|---|
| Iman-Conover | 0,0012 | 0,0010 |
| Cópula Gaussiana | 0,0037 | 0,0029 |
| Cópula Frank | 0,0046 | 0,0026 |
| Cópula Gumbel | 0,0054 | 0,0032 |
| Cópula Clayton | 0,0074 | 0,0049 |
| Cópula t | 0,0098 | 0,0095 |

O Iman-Conover é o mais preciso — ele mira o Spearman diretamente. As cópulas
pagam entre 3× e 8× mais erro na correlação **em troca** de controlar a
estrutura de cauda, que o Iman-Conover não controla. É a troca central da
ferramenta, e agora ela está quantificada.

### Ganho de variância do Latin Hypercube

200 replicações independentes de 2.000 iterações, modelo aditivo:

| Método | Erro médio absoluto da média |
|---|---|
| Monte Carlo | 0,0658 |
| Latin Hypercube | 0,0005 |
| **Razão** | **141×** |

O ganho é grande porque o modelo é uma soma — o caso mais favorável ao LHS.
Para modelos fortemente não lineares e interativos a vantagem encolhe, e o
número acima **não** deve ser extrapolado.

---

## 4. Cobertura de testes

| Camada | Casos |
|---|---|
| Motor (11 arquivos) | 448 |
| Interface Streamlit (`test_ui.py`) | 70 |
| Auditoria de defeitos (`test_auditoria.py`) | 26 |
| **Total** | **544 casos, a partir de 336 funções** |

Tempo de execução da suíte completa nesta máquina: **1m35s** para o motor e
**2m29s** para a interface, cerca de 4 minutos no total. A interface é a parte
lenta porque cada teste sobe o aplicativo inteiro e o exercita de ponta a
ponta.

---

## O que este documento NÃO mede

- **Acurácia das premissas.** Todos os números acima são sobre erro de
  amostragem e correção de implementação — a menor das fontes de erro em
  análise de risco. Ver `LIMITATIONS.md` seção 1.
- **Comparação com ferramentas comerciais.** Não há aqui nenhuma medição de
  @RISK ou ModelRisk. As funcionalidades foram implementadas a partir da
  documentação pública de cada uma, e o que se compara é o resultado contra a
  **solução matemática fechada**, não contra o número que outro programa
  produz.
- **Desempenho em modelos grandes e ramificados.** O modelo de teste tem três
  variáveis e uma fórmula aditiva. Modelos com dezenas de entradas e fórmulas
  profundas têm outro perfil de custo.
