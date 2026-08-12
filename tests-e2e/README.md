# Testes de navegador (Playwright)

Cobrem `index.html` — a versao do dashboard que roda no navegador via
Pyodide. O motor Python e a interface Streamlit sao cobertos por `pytest`
em `tests/`; esta suite fecha a unica parte do projeto que nao tinha teste
automatizado.

## Por que ela existe

Todos os defeitos relatados por usuario apareceram aqui, e nenhum foi pego
pela verificacao manual anterior:

| Defeito | Por que passou |
|---|---|
| Nao dava para digitar mais de um caractere por campo | A verificacao definia `.value` e disparava um unico evento `input`. Digitacao real sao N eventos sequenciais no mesmo elemento; com um so, o DOM nao chega a ser reconstruido no meio |
| `lambda` da PERT aparecia preenchido mas era tratado como ausente | Os testes sempre digitavam em todos os campos, entao o valor padrao nunca ficava so na tela |
| A previa da marginal ficava em branco apos adicionar outra variavel | Nenhuma verificacao adicionava uma segunda variavel depois de conferir a primeira |
| Com a CDN bloqueada, a tela mostrava `loadPyodide is not defined` | Nenhuma verificacao tentava abrir a pagina sem rede |

Por isso a regra do `ajudantes.js`: **`pressSequentially`, nunca `fill`**, em
qualquer campo cujo comportamento a cada tecla importe.

## Os dois arquivos de teste

| Arquivo | Precisa de rede? | O que cobre |
|---|---|---|
| `dashboard.spec.js` | **sim** | 13 testes: digitacao, previa, resultados contra solucao analitica, correlacao, validacao, ajuste a dados, importacao e exportacao |
| `offline.spec.js` | **nao** | 2 testes: o que a pagina mostra quando as CDNs estao inacessiveis |

## Rodar localmente

```bash
cd tests-e2e
npm install
npx playwright install --with-deps chromium
npm test
```

O `playwright.config.js` sobe um `python3 -m http.server` na raiz do
repositorio, entao os testes rodam contra a arvore de trabalho — nao contra
o que ja esta publicado no GitHub Pages.

Para ver o navegador: `npm run test:headed`.
Para abrir o relatorio depois: `npm run relatorio`.

## Rodar atras de um proxy restritivo

Se a rede bloqueia o host de download do navegador, aponte para um Chromium
que ja exista na maquina:

```bash
CHROMIUM_PATH=/caminho/para/chromium npx playwright test offline.spec.js
```

`offline.spec.js` roda inteiro nessas condicoes, porque testa justamente o
caminho sem rede. Ja o `dashboard.spec.js` precisa das CDNs: sem elas o
Pyodide nao inicializa e nao ha o que simular. As rodas de NumPy e SciPy
para WebAssembly nao existem no npm nem no PyPI — vem junto da distribuicao
do Pyodide —, entao nao ha como servi-las localmente sem baixar a
distribuicao inteira.

## O que esperar de tempo

A pagina baixa Pyodide, NumPy e SciPy (~30 MB) antes de ficar utilizavel.
A suite roda em serie, em um worker so, reaproveitando o cache HTTP do
mesmo contexto: o primeiro teste paga o download, os demais pagam so a
reinicializacao. Conte alguns minutos no total.

## O que NAO esta coberto

- Outros navegadores alem do Chromium.
- Layout responsivo e aparencia.
- Rede lenta (so o caso extremo de CDN indisponivel).
