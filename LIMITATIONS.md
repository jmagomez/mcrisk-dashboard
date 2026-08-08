# Limitações e modos de falha

Este documento existe porque software de análise de risco costuma ser vendido
pela lista de recursos, e quase nunca pela lista do que ele não resolve. A
seção mais importante é a primeira.

---

## 1. A maior fonte de erro não está no código

O motor deste repositório é testado e reprodutível. Isso não torna os
resultados confiáveis — porque o erro dominante em análise quantitativa de
risco quase sempre está nas **premissas de entrada**, não na aritmética.

Ordem de grandeza aproximada das fontes de erro, da maior para a menor:

1. **Escolha errada da distribuição** (ex.: usar Normal para algo com cauda
   pesada). Pode errar a probabilidade de um evento extremo por ordens de
   magnitude.
2. **Parâmetros elicitados mal.** A literatura de julgamento sob incerteza é
   consistente: especialistas produzem intervalos sistematicamente estreitos
   demais (excesso de confiança). Se o seu "máximo" é na verdade um P90, o
   modelo subestima a cauda inteira.
3. **Dependências ignoradas ou mal especificadas.** Tratar entradas
   correlacionadas como independentes distorce a variância da saída.
4. **Estrutura do modelo.** A fórmula pode simplesmente não representar o
   negócio.
5. **Erro de amostragem de Monte Carlo.** É o único item que mais iterações
   resolvem — e é tipicamente o menor de todos.

**Consequência prática:** aumentar de 10.000 para 1.000.000 de iterações
melhora o item 5 e não faz absolutamente nada pelos itens 1 a 4. Um resultado
com barra de erro estreita pode estar precisamente errado.

Leitura recomendada: Savage (2009), *The Flaw of Averages*; Saltelli et al.
(2020), *Five ways to ensure that models serve society*, Nature 582:482-484.

---

## 2. O que este app NÃO faz

Comparado ao @RISK e a ferramentas equivalentes, faltam aqui:

- **Índices de sensibilidade globais (Sobol).** Só há índices baseados em
  correlação/regressão de postos, válidos apenas sob monotonicidade. Índices de
  Sobol exigem desenho amostral próprio e avaliações adicionais do modelo.
- **Cópulas.** A dependência é imposta só via estrutura de postos
  (Iman-Conover). Não há cópula de Clayton, Gumbel ou t, e portanto **não há
  como modelar dependência de cauda** — o fenômeno em que variáveis se tornam
  mais correlacionadas justamente em cenários extremos. Isso importa muito em
  risco financeiro e de crédito.
- **Processos estocásticos e séries temporais.** Cada iteração é um sorteio
  independente no tempo. Não há movimento browniano, reversão à média,
  autocorrelação, sazonalidade ou GARCH.
- **Otimização sob incerteza** (equivalente ao RISKOptimizer).
- **Integração com Excel.** O modelo é uma fórmula digitada, não uma planilha
  com dependências entre células.
- **Inferência bayesiana.** O ajuste é por máxima verossimilhança, com
  parâmetros pontuais; a incerteza *sobre os parâmetros* não é propagada.
- **Simulação hierárquica / multinível**, análise de decisão em árvore,
  e análise de cronograma com dependências entre atividades (rede PERT/CPM).
- **Distribuições multivariadas nativas** (Normal multivariada, Dirichlet).

---

## 3. Limitações específicas de cada componente

### Amostragem

- **LHS quebra a independência entre iterações.** O erro padrão `s/√n` e os
  intervalos de confiança baseados nele deixam de ser válidos (Stein, 1987).
  O app avisa e oferece replicações independentes — mas o padrão está
  desligado, porque custa tempo. Se você reportar um IC sob LHS sem ativar
  replicações, o número exibido é apenas indicativo.
- **O ganho do LHS é maior para funções aproximadamente aditivas** das
  entradas. Para modelos fortemente não lineares e interativos, a vantagem
  encolhe.
- **LHS estratifica marginais, não o espaço conjunto.** Não é amostragem de
  baixa discrepância (Sobol/Halton) e não tem as garantias correspondentes.
- `lhs_median` (mediana do estrato) reduz variância, mas **subestima a
  variabilidade dentro do estrato** e não deve ser usado quando a cauda
  importa.

### Correlação (Iman-Conover)

- **Correlação de posto não determina a distribuição conjunta.** Infinitas
  cópulas produzem o mesmo Spearman. A estrutura efetivamente imposta é a
  induzida por escores normais — na prática, próxima de uma cópula gaussiana,
  que tem **dependência de cauda zero**. Se o seu risco é "tudo dá errado ao
  mesmo tempo", este método não o captura.
- **A correlação obtida é aproximada**, não exata. O erro medido é da ordem de
  0,001–0,02 dependendo de n e da matriz alvo.
- **Matrizes não positivas semidefinidas são reparadas silenciosamente** pelo
  motor (com aviso na interface). Isso significa que as correlações efetivas
  diferem das pedidas. Sempre confira a matriz obtida.
- **Correlação imposta não é mecanismo causal.** Se duas variáveis se movem
  juntas porque ambas dependem do câmbio, o correto é modelar o câmbio como
  variável, não impor uma correlação entre elas.

### Fórmula de saída

- Uma única expressão escalar por simulação. Não há células intermediárias,
  múltiplas saídas simultâneas, iteração temporal nem lógica condicional
  complexa além de `se(cond, a, b)`.
- Divisões por zero e logaritmos de números não positivos produzem valores não
  finitos. Eles são **descartados** das estatísticas, o que enviesa o resultado
  se a fração não for desprizível. O app informa a fração.

### Sensibilidade

- **Correlação de posto e SRRC só são válidos sob monotonicidade.** Para um
  modelo como `a * b` com `a`, `b` simétricos em torno de zero, ambos os
  índices dão ≈ 0 para variáveis que claramente importam. O app reporta o R² da
  regressão de postos e avisa quando ele é baixo — leia esse aviso.
- **Com entradas correlacionadas, a atribuição de importância é ambígua por
  construção.** Não existe forma única de dividir o crédito entre variáveis
  colineares. O app reporta o VIF máximo e avisa acima de 10.
- A coluna "contribuição para a variância" (SRRC²) só faz sentido com R² alto e
  entradas aproximadamente independentes.

### Ajuste de distribuições

- **AIC/BIC apenas ordenam candidatas.** Uma lista de modelos ruins ainda
  produz um vencedor. O peso de Akaike é evidência *relativa* dentro do
  conjunto testado (Burnham & Anderson, 2002, §2.9).
- **Testes de aderência têm pouco poder em amostras pequenas.** Com n < 30,
  quase nada é rejeitado — isso não é evidência de bom ajuste, é falta de
  informação. O app avisa.
- **Ajustar e depois simular ignora a incerteza dos parâmetros.** Os parâmetros
  estimados entram na simulação como se fossem conhecidos, o que subestima a
  incerteza total. A correção seria bootstrap ou tratamento bayesiano.
- **Selecionar a melhor distribuição e usar só ela ignora a incerteza de
  modelo.** Quando o peso de Akaike da primeira colocada é baixo, o app sugere
  rodar com mais de uma candidata e comparar.
- O ajuste é sempre **univariado e marginal**; não ajusta estrutura de
  dependência a partir de dados.
- Para distribuições de suporte positivo, `loc` é fixado em 0 quando os dados
  são positivos. É uma escolha defensável, mas é uma escolha — pode não ser
  adequada a dados com deslocamento real.

### Reamostragem empírica

- **Nunca gera valores fora do mínimo e do máximo observados.** Para análise de
  cauda isso subestima o risco extremo por construção. Se o pior caso histórico
  é o pior caso possível no seu modelo, o modelo está errado.

### Medidas de risco

- **VaR não é uma medida coerente de risco** — não é subaditivo, ou seja, o VaR
  de uma carteira pode ser maior que a soma dos VaR individuais (Artzner et
  al., 1999). Leia sempre junto com o CVaR.
- Estimativas de percentis extremos (P1, P99) e de CVaR têm muito mais erro de
  amostragem que a média, porque dependem de poucas observações na cauda. Os
  intervalos de confiança exibidos tornam isso visível — observe a largura.

---

## 4. Precisão numérica

- Distribuições de **variância infinita** (t com gl ≤ 2, Pareto com b ≤ 2) não
  têm média amostral que estabilize da forma usual; o TCL não se aplica e os
  intervalos de confiança da média são inválidos. O app permite usá-las, com
  aviso na descrição da distribuição.
- Quantis muito extremos de suportes ilimitados podem gerar overflow. Valores
  não finitos são detectados e reportados, não silenciados.
- A reprodutibilidade bit a bit depende da versão do NumPy e do SciPy. A
  especificação exportada em JSON registra a semente, mas não as versões —
  fixe-as via `requirements.txt` se precisar de reprodutibilidade estrita.

---

## 5. Segurança

O avaliador de fórmulas usa `ast` com lista branca de nós e funções, e é
testado contra 15 vetores de ataque conhecidos (acesso a `__class__`,
`__import__`, `open`, comprehensions, lambdas, indexação, etc.). Ainda assim:

- Não foi auditado por terceiros.
- **Não exponha este app publicamente sem revisão de segurança.** Ele foi
  pensado para uso local ou em rede interna confiável.
- Arquivos CSV enviados são lidos com `pandas.read_csv`, que tem sua própria
  superfície de ataque com arquivos malformados.

---

## 6. Quando NÃO usar este app

- Quando houver exigência regulatória de ferramenta validada (modelos
  regulatórios de capital, submissões a agências). Este software não tem
  validação formal nem trilha de auditoria certificada.
- Quando o risco relevante for de **dependência de cauda** ou **contágio** —
  faltam cópulas apropriadas.
- Quando o problema for essencialmente **temporal** (evolução de preços,
  filas, confiabilidade ao longo do tempo).
- Quando você não tiver base empírica nem elicitação estruturada para as
  distribuições. Nesse caso, o resultado é uma opinião com aparência de
  precisão — e a aparência de precisão é o risco.

---

## Referências desta seção

- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). *Coherent Measures
  of Risk*. Mathematical Finance 9(3):203-228.
- Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel
  Inference*, 2ª ed., Springer.
- Embrechts, P., McNeil, A. & Straumann, D. (2002). *Correlation and Dependence
  in Risk Management: Properties and Pitfalls*. In: Risk Management: Value at
  Risk and Beyond, Cambridge University Press.
- Saltelli, A. et al. (2020). *Five ways to ensure that models serve society: a
  manifesto*. Nature 582:482-484.
- Savage, S.L. (2009). *The Flaw of Averages*, Wiley.
- Stein, M. (1987). *Large Sample Properties of Simulations Using Latin
  Hypercube Sampling*. Technometrics 29(2):143-151.
- Taleb, N.N. (2007). *The Black Swan*, Random House.
