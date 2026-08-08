"""
Avaliacao segura de formulas de saida.

O usuario digita algo como `receita - custo_fixo - custo_var * volume`.
Isso PRECISA ser avaliado sem `eval()` cru: uma string arbitraria vinda de
um campo de texto e um vetor de execucao de codigo. Aqui a formula e
compilada com o modulo `ast` e so um subconjunto explicito de nos e
funcoes e aceito.

O que e permitido:
  - nomes de variaveis previamente definidas;
  - literais numericos;
  - + - * / // % ** e unario -/+;
  - comparacoes (< <= > >= == !=) e operadores logicos elementwise & | ~;
  - chamadas a funcoes da lista branca abaixo.

O que NAO e permitido (levanta erro): atributos, indexacao, lambdas,
comprehensions, atribuicoes, imports, chamadas a qualquer outro nome.

As operacoes sao vetorizadas em numpy: a formula e avaliada UMA vez sobre
arrays de tamanho n, nao n vezes sobre escalares.
"""

from __future__ import annotations

import ast
from typing import Dict, List

import numpy as np


def _desconto(taxa, periodo):
    """Fator de desconto 1 / (1 + taxa)^periodo, vetorizado.

    Nao ha uma funcao `vpl(...)` de conveniencia porque ela exigiria montar
    uma matriz periodo-a-periodo dentro da formula, o que a linguagem de
    expressao nao suporta. Escreva o VPL de forma explicita, por exemplo:
        -invest + fluxo1*desconto(taxa,1) + fluxo2*desconto(taxa,2)
    """
    return 1.0 / (1.0 + np.asarray(taxa, dtype=float)) ** np.asarray(
        periodo, dtype=float
    )


ALLOWED_FUNCS = {
    # elementares
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "sign": np.sign,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "clip": np.clip,
    # agregacao elementwise entre variaveis
    "min": np.minimum,
    "max": np.maximum,
    # condicional (equivalente ao SE do Excel)
    "se": np.where,
    "where": np.where,
    # financeiro
    "desconto": _desconto,
    # trigonometricas, ocasionalmente uteis
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}

FUNC_HELP = {
    "se(cond, a, b)": "condicional vetorizado, equivalente ao SE do Excel",
    "min(a, b) / max(a, b)": "minimo/maximo elemento a elemento",
    "clip(x, lo, hi)": "limita x ao intervalo [lo, hi]",
    "abs, exp, log, log10, sqrt, sign": "funcoes elementares",
    "round, floor, ceil": "arredondamentos",
    "desconto(taxa, periodo)": "fator 1/(1+taxa)^periodo, para montar VPL explicitamente",
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    # operadores
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)


class FormulaError(ValueError):
    """Erro de sintaxe, de nome ou de construcao nao permitida."""


class Formula:
    """Formula compilada e validada, pronta para avaliacao vetorizada."""

    def __init__(self, expr: str, allowed_names: List[str]):
        self.expr = (expr or "").strip()
        self.allowed_names = list(allowed_names)
        if not self.expr:
            raise FormulaError("formula vazia")
        try:
            self.tree = ast.parse(self.expr, mode="eval")
        except SyntaxError as e:
            raise FormulaError(f"erro de sintaxe: {e.msg}") from e
        self.used_names = self._validate()

    # -- validacao ----------------------------------------------------------
    def _validate(self) -> List[str]:
        used: List[str] = []
        allowed = set(self.allowed_names)
        for node in ast.walk(self.tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise FormulaError(
                    f"construcao nao permitida na formula: {type(node).__name__}"
                )
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise FormulaError("so sao permitidas chamadas de funcao simples")
                if node.func.id not in ALLOWED_FUNCS:
                    raise FormulaError(
                        f"funcao nao permitida: {node.func.id}. "
                        f"Disponiveis: {', '.join(sorted(ALLOWED_FUNCS))}"
                    )
                if node.keywords:
                    raise FormulaError("argumentos nomeados nao sao suportados")
            elif isinstance(node, ast.Name):
                if node.id in ALLOWED_FUNCS:
                    continue
                if node.id not in allowed:
                    raise FormulaError(
                        f"variavel desconhecida: '{node.id}'. "
                        f"Definidas: {', '.join(sorted(allowed)) or '(nenhuma)'}"
                    )
                if node.id not in used:
                    used.append(node.id)
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, bool)):
                    raise FormulaError("apenas literais numericos sao permitidos")
        return used

    # -- avaliacao ----------------------------------------------------------
    def evaluate(self, variables: Dict[str, np.ndarray]) -> np.ndarray:
        missing = [n for n in self.used_names if n not in variables]
        if missing:
            raise FormulaError(f"variaveis sem valores: {', '.join(missing)}")
        env: Dict[str, object] = dict(ALLOWED_FUNCS)
        env.update({k: np.asarray(v) for k, v in variables.items()})
        code = compile(self.tree, filename="<formula>", mode="eval")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = eval(code, {"__builtins__": {}}, env)  # noqa: S307
        arr = np.asarray(result, dtype=float)
        if arr.ndim == 0:
            n = max((np.size(v) for v in variables.values()), default=1)
            arr = np.full(n, float(arr))
        return arr

    def __repr__(self) -> str:  # pragma: no cover
        return f"Formula({self.expr!r}, usa={self.used_names})"


def validate_formula(expr: str, allowed_names: List[str]) -> tuple[bool, str]:
    """Conveniencia para a UI: (ok, mensagem)."""
    try:
        f = Formula(expr, allowed_names)
    except FormulaError as e:
        return False, str(e)
    if not f.used_names:
        return (
            True,
            "Formula valida, mas nao usa nenhuma variavel aleatoria: o "
            "resultado sera constante e a simulacao nao acrescenta informacao.",
        )
    return True, f"Formula valida. Usa: {', '.join(f.used_names)}"


def sanitize_variable_name(name: str) -> str:
    """Converte um rotulo livre em identificador Python valido."""
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")  # colapsa sequencias de underscore
    if not s:
        s = "var"
    if s[0].isdigit():
        s = "v_" + s
    return s
