// Funcoes de apoio aos testes de navegador.
//
// REGRA CENTRAL DESTE ARQUIVO: digitar com `pressSequentially`, nunca com
// `fill`, em qualquer campo cujo comportamento a cada tecla importe.
//
// `fill` define o valor de uma vez e dispara um unico evento. Foi exatamente
// assim que a verificacao anterior deste dashboard foi feita, e foi por isso
// que passou um defeito em que o app so aceitava UM caractere por campo: cada
// evento `input` reconstruia o DOM e destruia o proprio campo em uso. Um
// unico evento nunca reproduz isso; N eventos sequenciais reproduzem.

const { expect } = require("@playwright/test");

const ESPERA_APP = 300_000; // Pyodide + NumPy + SciPy sao ~30 MB

/** Abre a pagina e espera a interface ficar utilizavel. */
async function abrirApp(page) {
  await page.goto("/index.html");
  await expect(page.locator("#app")).toBeVisible({ timeout: ESPERA_APP });
  await expect(page.locator("#carregando")).toBeHidden();
  // A primeira variavel e criada pelo proprio app ao terminar de carregar.
  await expect(page.locator(".var")).toHaveCount(1);
}

/** Converte um numero formatado em pt-BR de volta para float. */
function numero(texto) {
  const t = String(texto == null ? "" : texto).trim();
  if (t === "" || t === "—") return NaN;
  if (/e[+-]?\d+$/i.test(t)) return Number(t.replace(",", "."));
  return Number(t.replace(/\./g, "").replace(",", "."));
}

/** Cartao da i-esima variavel (0-based). */
function cartao(page, i) {
  return page.locator(".var").nth(i);
}

/**
 * Preenche uma variavel pela interface, digitando de verdade.
 * Cria o cartao se ele ainda nao existir.
 */
async function definirVariavel(page, i, rotulo, dist, params) {
  const existentes = await page.locator(".var").count();
  if (i >= existentes) {
    for (let k = existentes; k <= i; k++) {
      await page.locator("#addvar").click();
    }
    await expect(page.locator(".var")).toHaveCount(i + 1);
  }
  const c = cartao(page, i);

  const campoRotulo = c.locator('input[data-campo="label"]');
  await campoRotulo.click();
  await campoRotulo.pressSequentially(rotulo, { delay: 12 });

  if (dist) {
    await c.locator('select[data-campo="dist"]').selectOption(dist);
  }

  for (const [nome, valor] of Object.entries(params || {})) {
    const campo = c.locator(`input[data-param="${nome}"]`);
    await campo.click();
    // Limpa o que o app tiver colocado como padrao antes de digitar.
    await campo.press("ControlOrMeta+a");
    await campo.pressSequentially(String(valor), { delay: 12 });
  }
}

/** Escreve a formula de saida, tecla a tecla. */
async function definirFormula(page, expr) {
  await page.locator('.abas button[data-aba="modelo"]').click();
  const campo = page.locator("#formula");
  await campo.click();
  await campo.press("ControlOrMeta+a");
  await campo.pressSequentially(expr, { delay: 12 });
}

/** Roda a simulacao e espera o resultado aparecer. */
async function rodar(page) {
  await page.locator('.abas button[data-aba="modelo"]').click();
  const botao = page.locator("#rodar");
  await expect(botao).toBeEnabled();
  await botao.click();
  await expect(page.locator("#status")).toContainText("Concluído", {
    timeout: 180_000,
  });
  await page.locator('.abas button[data-aba="result"]').click();
  await expect(page.locator("#conteudores")).toBeVisible();
}

/** Le uma metrica do painel de resultados pelo rotulo exibido. */
async function metrica(page, rotulo) {
  const bloco = page
    .locator("#metricas .metrica", { has: page.locator(".rot", { hasText: new RegExp(`^${rotulo}$`) }) })
    .first();
  return numero(await bloco.locator(".val").innerText());
}

/** Marca a caixa de aplicar correlacao e preenche uma celula da grade. */
async function definirCorrelacao(page, i, j, valor) {
  await page.locator('.abas button[data-aba="corr"]').click();
  const caixa = page.locator("#usarcorr");
  await expect(caixa).toBeVisible();
  if (!(await caixa.isChecked())) await caixa.check();
  const celula = page.locator(`input[data-corr="${i}_${j}"]`);
  await celula.click();
  await celula.press("ControlOrMeta+a");
  await celula.pressSequentially(String(valor), { delay: 12 });
}

/**
 * Correlacao de Pearson equivalente a um Spearman alvo, para a normal
 * bivariada: rho_P = 2 sin(pi rho_S / 6). E a relacao que o proprio motor
 * aplica (Kruskal, 1958), entao o valor esperado nos testes vem da teoria,
 * nao de uma rodada anterior do app.
 */
function pearsonDeSpearman(rhoS) {
  return 2 * Math.sin((Math.PI * rhoS) / 6);
}

/** Amostra deterministica ~ Normal(mu, sigma), sem depender de RNG do runner. */
function normaisDeterministicas(n, mu, sigma, semente) {
  let s = semente >>> 0;
  const rnd = () => {
    s = (1103515245 * s + 12345) % 2147483648;
    return s / 2147483648;
  };
  const out = [];
  for (let i = 0; i < n; i++) {
    const u1 = Math.max(rnd(), 1e-12);
    const u2 = rnd();
    out.push(mu + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
  }
  return out;
}

module.exports = {
  ESPERA_APP,
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
};
