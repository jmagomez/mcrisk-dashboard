// Testes de navegador do dashboard web (index.html).
//
// POR QUE ESTA SUITE EXISTE
// -------------------------
// O motor Python tem 236 testes. A interface Streamlit tem 42. A interface de
// navegador nao tinha nenhum — e foi exatamente ali que todos os defeitos
// relatados por usuario apareceram:
//
//   * nao dava para digitar mais de um caractere por campo;
//   * o lambda da PERT aparecia preenchido e era tratado como ausente;
//   * a previa da marginal ficava em branco apos adicionar outra variavel.
//
// Os dois primeiros passaram por uma verificacao manual feita em JavaScript,
// que definia `.value` e disparava UM evento. Isso nunca reproduz digitacao.
// Aqui usamos `pressSequentially`, que emite eventos de teclado de verdade,
// um por caractere — a unica forma de pegar esse tipo de defeito.
//
// Onde ha solucao analitica, o valor esperado vem da teoria, nao de uma
// rodada anterior do proprio app.

const { test, expect } = require("@playwright/test");
const {
  abrirApp,
  numero,
  cartao,
  definirVariavel,
  definirFormula,
  rodar,
  metrica,
  definirCorrelacao,
  pearsonDeSpearman,
  normaisDeterministicas,
} = require("./ajudantes");

// Uma pagina compartilhada entre os testes: assim o Pyodide e os pacotes ficam
// no cache HTTP do contexto e cada teste paga so a reinicializacao, nao o
// download de ~30 MB.
//
// NAO usamos `mode: "serial"`. Ele pula todos os testes seguintes assim que um
// falha — justamente na execucao em que mais se precisa do panorama completo.
// O isolamento entre testes vem do `page.reload()` no `beforeEach`, que nao
// depende de o teste anterior ter passado.

let page;

test.beforeAll(async ({ browser }) => {
  page = await browser.newPage();
  await abrirApp(page);
});

test.afterAll(async () => {
  if (page) await page.close();
});

test.beforeEach(async () => {
  await page.reload();
  await expect(page.locator("#app")).toBeVisible({ timeout: 300_000 });
  await expect(page.locator(".var")).toHaveCount(1);
});

// ---------------------------------------------------------------------------
// 1. Estado inicial
// ---------------------------------------------------------------------------

test("a interface carrega e nao traz numeros pre-preenchidos", async () => {
  await expect(page.locator(".abas button")).toHaveCount(6);
  await expect(page.locator("#metricas .metrica")).toHaveCount(0);
  await expect(cartao(page, 0).locator('input[data-campo="label"]')).toHaveValue("");
});

// ---------------------------------------------------------------------------
// 2. Digitacao real — o defeito relatado
// ---------------------------------------------------------------------------

test("digitar preserva todos os caracteres e nao rouba o foco", async () => {
  const c = cartao(page, 0);

  // (a) rotulo, 13 caracteres
  const rotulo = c.locator('input[data-campo="label"]');
  await rotulo.click();
  await rotulo.pressSequentially("custo da obra", { delay: 15 });
  await expect(rotulo).toHaveValue("custo da obra");

  // O foco tem que continuar no mesmo campo depois da ultima tecla. Se o app
  // reconstroi o DOM a cada evento, este e o assert que denuncia.
  expect(
    await page.evaluate(() => document.activeElement?.dataset?.campo || null)
  ).toBe("label");

  // (b) o nome derivado para a formula aparece
  await expect(c.locator("p.dica").first()).toContainText("custo_da_obra");

  // (c) parametro numerico com decimal
  const minimo = c.locator('input[data-param="minimo"]');
  await minimo.click();
  await minimo.pressSequentially("1234.5", { delay: 15 });
  await expect(minimo).toHaveValue("1234.5");

  // (d) formula
  await definirFormula(page, "custo_da_obra * 2");
  await expect(page.locator("#formula")).toHaveValue("custo_da_obra * 2");
});

// ---------------------------------------------------------------------------
// 3. Parametros com valor padrao
// ---------------------------------------------------------------------------

test("parametro com valor padrao conta como preenchido", async () => {
  // PERT e a distribuicao padrao e traz lambda = 4 no campo. Se o modelo so
  // registrasse o valor apos um evento `input`, quem preenchesse apenas
  // min/moda/max veria "parametro ausente: lam" olhando para um campo com 4.
  await definirVariavel(page, 0, "custo", "pert", {
    minimo: 100,
    moda: 150,
    maximo: 400,
  });
  await expect(page.locator("#msg-1")).toBeEmpty();
  await expect(page.locator("#varmsg")).toContainText("1 variável(is) pronta(s)");

  await definirFormula(page, "custo");
  await rodar(page);
  // Media teorica da PERT = (min + 4*moda + max) / 6
  expect(await metrica(page, "Média")).toBeCloseTo((100 + 4 * 150 + 400) / 6, 0);
});

// ---------------------------------------------------------------------------
// 4. Previa da marginal
// ---------------------------------------------------------------------------

test("a previa aparece e sobrevive a adicao de outra variavel", async () => {
  await definirVariavel(page, 0, "x", "normal", { mu: 10, sigma: 2 });
  await expect(page.locator("#prev-1 .js-plotly-plot")).toBeVisible();
  await expect(page.locator("#prevt-1")).toContainText("media");

  // Regressao: adicionar a segunda variavel recria o DOM. O cache de desenho
  // continuava batendo e a previa da primeira ficava em branco para sempre.
  await definirVariavel(page, 1, "y", "normal", { mu: 5, sigma: 1 });
  await expect(page.locator("#prev-1 .js-plotly-plot")).toBeVisible();
  await expect(page.locator("#prev-2 .js-plotly-plot")).toBeVisible();
});

// ---------------------------------------------------------------------------
// 5. Resultados contra solucao analitica
// ---------------------------------------------------------------------------

test("soma de normais reproduz media, desvio, percentis e caudas teoricos", async () => {
  await page.locator("#iteracoes").fill("40000");
  await definirVariavel(page, 0, "x", "normal", { mu: 10, sigma: 2 });
  await definirVariavel(page, 1, "y", "normal", { mu: 5, sigma: 1 });
  await definirFormula(page, "x + y");
  await rodar(page);

  // x + y ~ N(15, sqrt(5))
  const dp = Math.sqrt(5);
  expect(await metrica(page, "Média")).toBeCloseTo(15, 1);
  expect(await metrica(page, "Desvio-padrão")).toBeCloseTo(dp, 1);

  // Percentis exibidos na tabela, contra os quantis da normal.
  const linhaP5 = page.locator("#tabperc tr", { hasText: /^P5/ }).first();
  const valorP5 = numero(await linhaP5.locator("td").nth(1).innerText());
  expect(valorP5).toBeCloseTo(15 - 1.6448536 * dp, 0);

  // VaR 95% e o percentil 5; o CVaR e a media alem dele, logo mais extremo.
  const varr = numero(await page.locator("#valvar").innerText());
  const cvar = numero(await page.locator("#valcvar").innerText());
  expect(varr).toBeCloseTo(15 - 1.6448536 * dp, 0);
  expect(cvar).toBeLessThan(varr);

  // Sob LHS o app precisa avisar que s/sqrt(n) nao vale.
  await expect(page.locator("#avisoerro")).toContainText("Latin Hypercube");
});

// ---------------------------------------------------------------------------
// 6. Correlacao — o defeito mais grave ja encontrado no projeto
// ---------------------------------------------------------------------------

test("correlacao preenchida em um triangulo so nao e dividida pela metade", async () => {
  // A tela manda preencher um triangulo so. Uma "simetrizacao" por media com
  // o lado vazio transformaria 0,8 em 0,4 — sem erro, sem aviso, so numeros
  // errados. Var(x+y) denuncia a diferenca de forma mensuravel.
  await page.locator("#iteracoes").fill("40000");
  await definirVariavel(page, 0, "x", "normal", { mu: 10, sigma: 2 });
  await definirVariavel(page, 1, "y", "normal", { mu: 5, sigma: 1 });
  await definirCorrelacao(page, 0, 1, 0.8);
  await definirFormula(page, "x + y");
  await rodar(page);

  const desvioCerto = Math.sqrt(5 + 2 * pearsonDeSpearman(0.8) * 2 * 1);
  const desvioMetade = Math.sqrt(5 + 2 * pearsonDeSpearman(0.4) * 2 * 1);
  const obtido = await metrica(page, "Desvio-padrão");

  expect(Math.abs(obtido - desvioCerto)).toBeLessThan(
    Math.abs(obtido - desvioMetade)
  );
  expect(obtido).toBeCloseTo(desvioCerto, 1);

  // A tabela pedida vs. obtida precisa existir e bater com o alvo.
  await expect(page.locator("#seccorrobt")).toBeVisible();
  const linha = page.locator("#tabcorr tr").nth(1);
  expect(numero(await linha.locator("td").nth(1).innerText())).toBeCloseTo(0.8, 2);
  expect(numero(await linha.locator("td").nth(2).innerText())).toBeCloseTo(0.8, 1);
});

test("valores conflitantes nos dois triangulos geram aviso", async () => {
  await definirVariavel(page, 0, "x", "normal", { mu: 0, sigma: 1 });
  await definirVariavel(page, 1, "y", "normal", { mu: 0, sigma: 1 });
  await definirCorrelacao(page, 0, 1, 0.8);
  await definirCorrelacao(page, 1, 0, 0.2);
  await expect(page.locator("#corrmsg")).toContainText("conflitantes");
});

// ---------------------------------------------------------------------------
// 7. Validacao de entradas
// ---------------------------------------------------------------------------

test("rotulos duplicados sao sinalizados e bloqueiam a simulacao", async () => {
  await definirVariavel(page, 0, "custo", "normal", { mu: 100, sigma: 10 });
  await definirVariavel(page, 1, "custo", "normal", { mu: 50, sigma: 5 });
  await expect(page.locator("#varmsg")).toContainText("Rótulos repetidos");
  await page.locator('.abas button[data-aba="modelo"]').click();
  await expect(page.locator("#rodar")).toBeDisabled();
});

test("parametro invalido e injecao de codigo sao barrados", async () => {
  // (a) sigma <= 0 nao e uma distribuicao valida
  await definirVariavel(page, 0, "x", "normal", { mu: 0, sigma: -1 });
  await expect(page.locator("#msg-1")).toContainText("sigma deve ser > 0");

  // (b) formula com tentativa de execucao de codigo
  await cartao(page, 0).locator('input[data-param="sigma"]').fill("1");
  await definirFormula(page, "__import__('os').system('ls')");
  await expect(page.locator("#formsg .erro")).toBeVisible();
  await expect(page.locator("#rodar")).toBeDisabled();
});

// ---------------------------------------------------------------------------
// 8. Ajuste de distribuicoes a dados
// ---------------------------------------------------------------------------

test("o ajuste elege a familia correta e recupera os parametros", async () => {
  const dados = normaisDeterministicas(300, 50, 8, 42);
  await page.locator('.abas button[data-aba="ajuste"]').click();
  await page.locator("#fitdados").fill(dados.map((v) => v.toFixed(4)).join(", "));
  await expect(page.locator("#fitdesc")).toContainText("n");
  await page.locator("#fitboot").selectOption("100");

  await page.locator("#ajustar").click();
  await expect(page.locator("#fitres")).toBeVisible({ timeout: 180_000 });

  // Dados vindos de uma Normal: a Normal deve vencer o ranking por AICc.
  const primeira = page.locator("#tabfit tr").nth(1);
  await expect(primeira.locator("td").first()).toHaveText("normal");

  // E os parametros ajustados devem estar proximos dos que geraram os dados.
  // O paragrafo tem dois <code> (os parametros e a mencao a scipy.stats);
  // queremos o primeiro. E o separador e ", " porque em pt-BR a virgula
  // tambem e o separador decimal.
  await expect(page.locator("#fitparams")).toContainText("normal");
  const texto = await page.locator("#fitparams code").first().innerText();
  const partes = texto.split(", ").map((s) => numero(s));
  expect(partes[0]).toBeCloseTo(50, 0);
  expect(partes[1]).toBeCloseTo(8, 0);

  // Sob H0 verdadeira o bootstrap nao deve rejeitar.
  const pKS = numero(await primeira.locator("td").nth(8).innerText());
  expect(pKS).toBeGreaterThan(0.05);
});

// ---------------------------------------------------------------------------
// 9. Importar especificacao
// ---------------------------------------------------------------------------

test("importar JSON carrega variaveis, formula, configuracao e correlacao", async () => {
  const spec = {
    iteracoes: 9000,
    metodo: "mc",
    semente: 321,
    formula: "a + b",
    variaveis: [
      { rotulo: "a", distribuicao: "normal", parametros: { mu: 10, sigma: 2 } },
      { rotulo: "b", distribuicao: "normal", parametros: { mu: 5, sigma: 1 } },
    ],
    correlacao: [
      [1, 0.7],
      [0.7, 1],
    ],
  };
  await page.setInputFiles("#arqjson", {
    name: "modelo.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(spec)),
  });

  await expect(page.locator("#importmsg")).toContainText("Modelo carregado");
  await expect(page.locator(".var")).toHaveCount(2);
  await expect(page.locator("#formula")).toHaveValue("a + b");
  await expect(page.locator("#iteracoes")).toHaveValue("9000");
  await expect(page.locator("#metodo")).toHaveValue("mc");
  await expect(page.locator("#seed")).toHaveValue("321");

  // A tela pede um triangulo so, entao a importacao nao deve poluir o outro.
  await page.locator('.abas button[data-aba="corr"]').click();
  await expect(page.locator('input[data-corr="0_1"]')).toHaveValue("0.7");
  await expect(page.locator('input[data-corr="1_0"]')).toHaveValue("");

  // E o modelo importado tem que rodar de verdade.
  await rodar(page);
  expect(await metrica(page, "Média")).toBeCloseTo(15, 0);
});

// ---------------------------------------------------------------------------
// 10. Replicacoes independentes
// ---------------------------------------------------------------------------

test("replicacoes independentes trocam o erro padrao pelo estimador valido", async () => {
  await page.locator("#iteracoes").fill("5000");
  await page.locator("#usarreplicas").check();
  await page.locator("#nreplicas").fill("5");
  await definirVariavel(page, 0, "x", "normal", { mu: 10, sigma: 2 });
  await definirVariavel(page, 1, "y", "normal", { mu: 5, sigma: 1 });
  await definirFormula(page, "x + y");
  await rodar(page);

  // Com replicacoes o painel troca s/sqrt(n) pelo erro entre replicas, e o
  // aviso de LHS deixa de fazer sentido.
  await expect(page.locator("#metricaserro")).toContainText("Replicações");
  await expect(page.locator("#metricaserro")).toContainText("Erro padrão (válido)");
  await expect(page.locator("#avisoerro")).not.toContainText("Latin Hypercube");
  expect(await metrica(page, "Média")).toBeCloseTo(15, 0);
});

// ---------------------------------------------------------------------------
// 11. Exportacao
// ---------------------------------------------------------------------------

test("exportar CSV e JSON produz arquivos com o conteudo esperado", async () => {
  await page.locator("#iteracoes").fill("2000");
  await definirVariavel(page, 0, "x", "normal", { mu: 10, sigma: 2 });
  await definirFormula(page, "x * 2");
  await rodar(page);

  const baixaCsv = page.waitForEvent("download");
  await page.locator("#baixarcsv").click();
  const csv = await baixaCsv;
  expect(csv.suggestedFilename()).toBe("simulacao_iteracoes.csv");

  const baixaJson = page.waitForEvent("download");
  await page.locator("#baixarjson").click();
  const json = await baixaJson;
  expect(json.suggestedFilename()).toBe("modelo.json");

  // O JSON precisa registrar a semente: e o que torna o resultado reproduzivel.
  const fs = require("fs");
  const caminho = await json.path();
  const spec = JSON.parse(fs.readFileSync(caminho, "utf8"));
  expect(spec.semente).toBe(12345);
  expect(spec.formula).toBe("x * 2");
  expect(spec.variaveis).toHaveLength(1);
});
