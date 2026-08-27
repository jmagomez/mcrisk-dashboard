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

- **Índices de sensibilidade globais (Sobol).** Dos quatro métodos
  implementados, três supõem monotonicidade e o quarto (mudança na estatística
  da saída) a dispensa, mas é marginal — não controla as demais entradas.
  Nenhum deles decompõe a variância em efeitos principais e de interação, que é
  o que os índices de Sobol dão. Eles exigem desenho amostral próprio e
  avaliações adicionais do modelo, e por isso não cabem na arquitetura atual,
  que calcula tudo sobre a amostra já existente.
- **Cópula ajustada e usada no mesmo fluxo.** Há cinco famílias — Gaussiana, t,
  Clayton, Gumbel e Frank — cobrindo cauda simétrica, só inferior, só superior
  e nenhuma. E há `mcrisk.copula.fit_copula`, que ajusta as três arquimedianas
  a dados por pseudo-verossimilhança e as ordena por AIC. O que **não** existe
  é a ligação entre as duas coisas: o ajuste não alimenta o seletor da
  interface, e a cópula usada na simulação continua sendo escolhida a mão. Os
  graus de liberdade da t são informados, não estimados. Faltam também as
  famílias de dois parâmetros (BB1, BB7) e as vines, que permitiriam estrutura
  de dependência heterogênea entre pares — as três arquimedianas aqui são
  **permutáveis**. Ver §3.
- **Processos estocásticos e séries temporais.** Cada iteração é um sorteio
  independente no tempo. Não há movimento browniano, reversão à média,
  autocorrelação, sazonalidade ou GARCH.
- **Otimização sob incerteza** (equivalente ao RISKOptimizer).
- **Integração com Excel.** O modelo é uma fórmula digitada, não uma planilha
  com dependências entre células.
- **Inferência bayesiana propriamente dita.** A incerteza sobre os parâmetros
  passou a ser propagada (por bootstrap, não por posteriori) e as candidatas
  podem ser combinadas por peso de Akaike, o que aproxima a média de modelos
  bayesiana. Mas não há priori, não há posteriori e não há MCMC: nada aqui
  permite incorporar conhecimento prévio, nem obter a distribuição a
  posteriori de um parâmetro.
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

### Dependência por postos (Iman-Conover)

- **Correlação de posto não determina a distribuição conjunta.** Infinitas
  cópulas produzem o mesmo Spearman. A estrutura efetivamente imposta é a
  induzida por escores normais — na prática, próxima de uma cópula gaussiana,
  que tem **dependência de cauda zero**. Se o seu risco é "tudo dá errado ao
  mesmo tempo", este método não o captura — use a cópula t, com as ressalvas
  logo abaixo.
- **A correlação obtida é aproximada**, não exata. O erro medido é da ordem de
  0,001–0,02 dependendo de n e da matriz alvo.
- **Matrizes não positivas semidefinidas são reparadas** pelo motor, com aviso
  na interface. Isso significa que as correlações efetivas diferem das pedidas.
  Sempre confira a tabela de correlação **obtida**.
- **Correlação imposta não é mecanismo causal.** Se duas variáveis se movem
  juntas porque ambas dependem do câmbio, o correto é modelar o câmbio como
  variável, não impor uma correlação entre elas.

### Cópulas: o que elas resolvem e o que continua por sua conta

A cópula t admite eventos extremos simultâneos que a Gaussiana não produz —
coeficiente de dependência de cauda
λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) > 0, contra λ = 0 na Gaussiana para
qualquer ρ < 1. Isso corrige uma subestimação real de risco de cauda, mas não
transforma a ferramenta em modelo calibrado.

- **A escolha é sua, não dos dados.** Os graus de liberdade da t são um
  parâmetro que você informa, e a família usada na simulação é a que você marca
  no seletor. `mcrisk.copula.fit_copula` ajusta as três arquimedianas a um par
  de séries e as ordena por AIC, mas o resultado **não** alimenta o seletor:
  quem quiser usá-lo precisa ler o número e escolher a mão. Ligar as duas
  pontas exigiria série conjunta longa para cada par do modelo, e automatizar
  isso sem a série daria a impressão de calibração onde não há.
- **A estratificação do LHS deixa de valer.** Com cópula, o cubo unitário passa
  a vir da própria cópula, não do desenho estratificado. Para o mesmo número de
  iterações, o erro de simulação tende a ser **maior** que no modo
  Iman-Conover. O app avisa isso na tela.
- **As marginais deixam de ser preservadas exatamente.** Iman-Conover apenas
  reordena os valores sorteados; a cópula os regenera. As marginais continuam
  corretas em distribuição, mas não valor a valor.
- **A conversão Spearman → Pearson vira aproximação.** `2·sen(π·ρₛ/6)` é exata
  para a cópula Gaussiana; para a t, o ρ de Spearman também depende de ν, sem
  forma fechada elementar. O desvio é da ordem de 0,02 para ν = 3 e ρ = 0,5.
  Leia a tabela de correlação **obtida**, não a pedida.
- **A t é radialmente simétrica.** Ela impõe a mesma dependência na cauda
  superior e na inferior. Para assimetria de cauda há **Clayton** (λ inferior
  = 2^(−1/θ), λ superior = 0) e **Gumbel** (o contrário), e **Frank** como
  controle: mesma família arquimediana, dependência de cauda zero dos dois
  lados. Se trocar Frank por Gaussiana muda o resultado, o que mudou não foi
  a cauda.
- **As três arquimedianas são PERMUTÁVEIS.** Um único parâmetro governa todos
  os pares. Não existe "A e B muito dependentes, C pouco": uma matriz
  heterogênea é achatada no ρ médio, e a interface avisa quanto de dispersão
  foi descartado. Com dependências que diferem entre pares, a escolha honesta
  continua sendo Gaussiana ou t, que aceitam a matriz inteira.
- **A calibração por ρ de Spearman é numérica, não fechada.** Não há fórmula
  elementar ligando ρ ao θ dessas famílias. A grade é simulada com semente
  fixa e invertida por interpolação: determinística, mas com erro próprio de
  até 0,0077 (contra 0,0025 da calibração por τ de Kendall, que tem forma
  fechada). Os números estão no `BENCHMARK.md`.

### Cenários: estresse não tem probabilidade

Um cenário de **estresse** troca a distribuição de uma entrada e re-simula. O
resultado vale para aquele mundo hipotético e **não** carrega a probabilidade
do modelo original. Ele não é um percentil da distribuição base, e apresentá-lo
como se fosse é o erro mais comum nesta área — a diferença entre "há 5% de
chance disso" e "se isso acontecer, o resultado é este".

A análise **condicional** é outra coisa: recorta iterações que já existem, com
as probabilidades do próprio modelo. O preço é amostral — condicionar em cauda
estreita deixa poucas iterações, e a ação de recortar não gera informação nova.
A ferramenta avisa abaixo de 200 iterações restantes; abaixo disso as
estatísticas do cenário são ruído.

### Fórmula de saída

- Uma única expressão escalar por simulação. Não há células intermediárias,
  múltiplas saídas simultâneas, iteração temporal nem lógica condicional
  complexa além de `se(cond, a, b)`.
- Divisões por zero e logaritmos de números não positivos produzem valores não
  finitos. Eles são **descartados** das estatísticas, o que enviesa o resultado
  se a fração não for desprezível. O app informa a fração.

### Sensibilidade

- **Dois dos quatro métodos só são válidos sob monotonicidade.** Para um
  modelo como `a * b` com `a`, `b` simétricos em torno de zero, a correlação de
  posto e o SRRC dão ≈ 0 para variáveis que claramente importam. Medido no
  repositório com `y = a²`: ambos devolvem −0,0022 para a única variável que
  importa. O app reporta o R² da regressão de postos e avisa quando ele é
  baixo — e o método de **mudança na estatística da saída** existe justamente
  para esse caso, devolvendo swing 3,2344 no mesmo modelo.
- **O método que dispensa monotonicidade é marginal.** Ele não controla as
  demais entradas, e depende da escolha do número de faixas: poucas suavizam
  demais, muitas deixam cada faixa com poucas iterações e o swing passa a medir
  ruído. A função avisa quando as faixas ficam pequenas.
- **Com entradas correlacionadas, a atribuição de importância é ambígua por
  construção.** Não existe forma única de dividir o crédito entre variáveis
  colineares. O app reporta o VIF máximo e avisa acima de 10. Na contribuição
  para a variância a ambiguidade tem forma específica: a soma de quadrados é
  **sequencial**, então quem entra antes na regressão fica com a parte
  compartilhada. A ordem de entrada é reportada para que isso seja visível.
- **As frações de variância somam o R², não 100%.** O que sobra não pertence a
  entrada nenhuma. Com `y = a·b`, medido: 99,98% não explicada. Ler as frações
  como se cobrissem o modelo inteiro é o erro que este método convida a
  cometer.
- **Nenhum dos quatro decompõe a variância em efeitos principais e de
  interação.** Isso são os índices de Sobol, que não estão implementados.

### Ajuste de distribuições

- **AIC/BIC apenas ordenam candidatas.** Uma lista de modelos ruins ainda
  produz um vencedor. O peso de Akaike é evidência *relativa* dentro do
  conjunto testado (Burnham & Anderson, 2002, §2.9).
- **Testes de aderência têm pouco poder em amostras pequenas.** Com n < 30,
  quase nada é rejeitado — isso não é evidência de bom ajuste, é falta de
  informação. O app avisa.
- **A incerteza de parâmetro é propagável, mas não por padrão.** O caminho
  existe (`parameter_uncertainty` + `simular_com_incerteza`, e o painel da aba
  5), e transcrever só os parâmetros pontuais para a aba 1 continua sendo o que
  a maioria vai fazer. Nesse caminho, a simulação herda uma precisão que os
  dados não sustentam. Quanto isso importa depende de n: medido para a normal,
  o efeito na variância cai de ~16% com n = 20 para 0,4% com n = 1.000. Com
  dados suficientes, ignorar passa a ser defensável — o `BENCHMARK.md` traz a
  tabela para que a decisão seja tomada com evidência.
- **O bootstrap mede a incerteza CONDICIONAL a esta amostra.** Ele reamostra o
  que existe e não consegue inventar a informação que a amostra não tem. Com n
  pequeno o intervalo sai otimista, e a função avisa abaixo de 30 observações.
- **A média de modelos combina candidatas; não conserta um conjunto em que
  nenhuma serve.** Se todas forem rejeitadas pelo teste K-S, misturá-las produz
  uma mistura igualmente inadequada, e o app diz isso.
- **A cópula ajustada não é usada na simulação.** `fit_copula` existe e é
  bivariado; o ajuste marginal continua univariado. Não há caminho automático
  do ajuste de dependência para o modelo — ver §2.
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
- Quando a estrutura de dependência precisar ser **estimada e usada**, não
  escolhida. O ajuste de cópula existe, mas não alimenta a simulação; os graus
  de liberdade da t são informados a mão; e as arquimedianas são permutáveis,
  o que impede dependência heterogênea entre pares.
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
- Demarta, S. & McNeil, A.J. (2005). *The t Copula and Related Copulas*.
  International Statistical Review 73(1):111-129.
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
- Embrechts, P., McNeil, A. & Straumann, D. (2002). *Correlation and Dependence
  in Risk Management: Properties and Pitfalls*. In: Risk Management: Value at
  Risk and Beyond, Cambridge University Press.
- Genest, C. & Rivest, L.-P. (1993). *Statistical Inference Procedures for
  Bivariate Archimedean Copulas*. JASA 88(423):1034-1043.
- Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
- Marshall, A.W. & Olkin, I. (1988). *Families of Multivariate Distributions*.
  JASA 83(403):834-841.
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management:
  Concepts, Techniques and Tools*, ed. revisada. Princeton University Press.
- Nelsen, R.B. (2006). *An Introduction to Copulas*, 2ª ed., Springer.
- Saltelli, A. et al. (2020). *Five ways to ensure that models serve society: a
  manifesto*. Nature 582:482-484.
- Savage, S.L. (2009). *The Flaw of Averages*, Wiley.
- Stein, M. (1987). *Large Sample Properties of Simulations Using Latin
  Hypercube Sampling*. Technometrics 29(2):143-151.
- Taleb, N.N. (2007). *The Black Swan*, Random House.
