// Configuracao do Playwright para o dashboard de navegador.
//
// O `index.html` carrega Pyodide, NumPy e SciPy de CDN (~30 MB) e so entao
// habilita a interface. Isso domina o tempo de cada teste, e e a razao dos
// timeouts generosos e da execucao em um worker so: rodar em paralelo faria
// cada worker baixar tudo de novo.

const { defineConfig, devices } = require("@playwright/test");

module.exports = defineConfig({
  testDir: ".",
  timeout: 240_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never", outputFolder: "relatorio" }]]
    : [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "retain-on-failure",
    video: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // Serve a raiz do repositorio: o teste roda contra os arquivos da arvore de
  // trabalho, nao contra o que ja esta publicado no GitHub Pages.
  webServer: {
    command: "python3 -m http.server 8000 --bind 127.0.0.1 --directory ..",
    url: "http://127.0.0.1:8000/index.html",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
