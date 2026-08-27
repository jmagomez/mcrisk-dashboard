"""
O manifesto de arquivos do dashboard de navegador tem que bater com o pacote.

POR QUE ESTE ARQUIVO EXISTE
---------------------------
`index.html` nao empacota o Python: ele baixa `mcrisk/*.py` um a um, por uma
lista fixa (`PY_ARQUIVOS`), e escreve no sistema de arquivos virtual do
Pyodide. `mcrisk/__init__.py` importa o pacote INTEIRO, entao basta um modulo
de fora da lista para a pagina nao abrir.

Foi o que aconteceu: `copula`, `convergence` e `scenarios` entraram no pacote
e a lista nao acompanhou. E a mensagem que chegava na tela nao ajudava em
nada --

    ImportError: cannot import name 'copula' from partially initialized
    module 'mcrisk' (most likely due to a circular import)

-- porque quando falta um submodulo o CPython engole o ModuleNotFoundError e
culpa uma importacao circular que nao existe. O grafo de imports do pacote e
aciclico; o problema era um arquivo que nunca foi baixado.

A suite Python nao pegava isso (importa do disco, onde nada falta) e a suite
de navegador so pega com rede, baixando ~30 MB de CDN. Este teste pega em
milissegundos e sem rede, que e o que faz dele o lugar certo para a guarda.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
INDEX = RAIZ / "index.html"
PACOTE = RAIZ / "mcrisk"


def _manifesto() -> list[str]:
    """Le a lista PY_ARQUIVOS de index.html."""
    texto = INDEX.read_text(encoding="utf-8")
    m = re.search(r"const PY_ARQUIVOS\s*=\s*(\[.*?\]);", texto, re.S)
    assert m, "nao achei a declaracao de PY_ARQUIVOS em index.html"
    return json.loads(re.sub(r"\s+", "", m.group(1)))


def _modulos_do_pacote() -> set[str]:
    return {p.name for p in PACOTE.glob("*.py")}


def _importados_pelo_init() -> set[str]:
    """Os submodulos que `mcrisk/__init__.py` importa na carga."""
    arvore = ast.parse((PACOTE / "__init__.py").read_text(encoding="utf-8"))
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.ImportFrom) and no.level == 1 and no.module is None:
            nomes.update(alias.name for alias in no.names)
    return nomes


def test_manifesto_cobre_todo_o_pacote():
    faltando = _modulos_do_pacote() - set(_manifesto())
    assert not faltando, (
        "estes modulos existem em mcrisk/ mas index.html nunca os baixa: "
        + ", ".join(sorted(faltando))
        + ". Como __init__.py importa o pacote inteiro, a pagina nao abre. "
        "Acrescente-os a PY_ARQUIVOS."
    )


def test_manifesto_nao_pede_arquivo_inexistente():
    sobrando = set(_manifesto()) - _modulos_do_pacote()
    assert not sobrando, (
        "PY_ARQUIVOS pede arquivos que nao existem em mcrisk/: "
        + ", ".join(sorted(sobrando))
        + ". O fetch devolve 404 e a pagina nao abre."
    )


def test_tudo_que_o_init_importa_esta_no_manifesto():
    """A condicao que realmente derruba a pagina, checada direto na fonte."""
    disponiveis = {n[:-3] for n in _manifesto()}
    faltando = _importados_pelo_init() - disponiveis
    assert not faltando, (
        "__init__.py importa " + ", ".join(sorted(faltando))
        + ", que PY_ARQUIVOS nao baixa. A pagina falha com um ImportError que "
        "culpa uma importacao circular inexistente."
    )


def test_o_pacote_nao_tem_importacao_circular():
    """
    A mensagem do defeito acusava um ciclo. Nao havia -- e este teste registra
    isso, para a proxima pessoa nao sair procurando o ciclo errado.
    """
    grafo: dict[str, set[str]] = {}
    for caminho in PACOTE.glob("*.py"):
        if caminho.name == "__init__.py":
            continue
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        alvos: set[str] = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.ImportFrom) and no.level == 1:
                if no.module is None:  # from . import x
                    alvos.update(a.name for a in no.names)
                else:  # from .x import y
                    alvos.add(no.module.split(".")[0])
        grafo[caminho.stem] = alvos & {p.stem for p in PACOTE.glob("*.py")}

    visitando: set[str] = set()
    concluidos: set[str] = set()

    def ciclo_a_partir_de(no: str, caminho: list[str]) -> list[str] | None:
        if no in visitando:
            return caminho[caminho.index(no):] + [no]
        if no in concluidos:
            return None
        visitando.add(no)
        for vizinho in sorted(grafo.get(no, ())):
            achado = ciclo_a_partir_de(vizinho, caminho + [no])
            if achado:
                return achado
        visitando.discard(no)
        concluidos.add(no)
        return None

    for modulo in sorted(grafo):
        achado = ciclo_a_partir_de(modulo, [])
        assert achado is None, "importacao circular: " + " -> ".join(achado)
