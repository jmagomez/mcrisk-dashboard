// O que a pagina faz quando a rede bloqueia as CDNs.
//
// Este arquivo e proposital e util: ele NAO precisa de internet, porque testa
// exatamente o caminho em que a internet falhou. Roda em qualquer maquina,
// inclusive atras de um proxy restritivo — que foi onde o defeito abaixo
// apareceu.
//
// O defeito que ele guarda: com as CDNs bloqueadas a pagina exibia
// "Falha ao carregar: loadPyodide is not defined" — o nome de uma variavel de
// JavaScript, que nao diz nada a quem esta olhando a tela — e a barra de
// progresso continuava animando, sugerindo que ainda estava carregando.
//
// Verificado contra a versao anterior do index.html, onde falha com
// exatamente aquela mensagem.

const { test, expect } = require("@playwright/test");

test("com as CDNs bloqueadas, a pagina explica a causa e o que fazer", async ({ page }) => {
  await page.route("**cdn.jsdelivr.net/**", (r) => r.abort());
  await page.route("**cdn.plot.ly/**", (r) => r.abort());
  await page.goto("/index.html");

  const passo = page.locator("#passo");
  await expect(passo.locator(".erro")).toBeVisible({ timeout: 60_000 });

  // Nomeia o que faltou, em vez de vazar o nome de uma variavel interna.
  await expect(passo).toContainText("Não consegui baixar");
  await expect(passo).toContainText("cdn.jsdelivr.net");
  await expect(passo).not.toContainText("loadPyodide");

  // Diz a causa provavel e oferece uma saida.
  await expect(passo).toContainText("proxy");
  await expect(passo.locator("a")).toHaveAttribute("href", /mcrisk-dashboard/);

  // E para de fingir que ainda esta carregando.
  await expect(page.locator("#barra")).toBeHidden();
  await expect(page.locator("#app")).toBeHidden();
});

test("a mensagem nomeia os dois dominios que falharam", async ({ page }) => {
  await page.route("**cdn.plot.ly/**", (r) => r.abort());
  await page.route("**cdn.jsdelivr.net/**", (r) => r.abort());
  await page.goto("/index.html");
  const passo = page.locator("#passo");
  await expect(passo).toContainText("cdn.plot.ly", { timeout: 60_000 });
  await expect(passo).toContainText("cdn.jsdelivr.net");
});
