"""
Verificacao do avaliador de formulas.

Metade destes testes e de SEGURANCA: a formula vem de um campo de texto,
e se `eval` cru fosse usado uma string maliciosa executaria codigo
arbitrario. Cada teste abaixo representa um vetor de ataque que deve ser
bloqueado na etapa de parsing, antes de qualquer execucao.
"""

import numpy as np
import pytest

from mcrisk.formula import (
    Formula,
    FormulaError,
    sanitize_variable_name,
    validate_formula,
)

VARS = ["a", "b", "c"]
DADOS = {
    "a": np.array([1.0, 2.0, 3.0, 4.0]),
    "b": np.array([10.0, 10.0, 10.0, 10.0]),
    "c": np.array([0.5, 0.5, 2.0, 2.0]),
}


# --------------------------------------------------------------------------
# Comportamento correto
# --------------------------------------------------------------------------


def test_aritmetica_basica():
    f = Formula("a * b - c", VARS)
    assert np.allclose(f.evaluate(DADOS), [9.5, 19.5, 28.0, 38.0])


def test_precedencia_e_parenteses():
    assert np.allclose(
        Formula("(a + b) * c", VARS).evaluate(DADOS), [5.5, 6.0, 26.0, 28.0]
    )


def test_potencia_e_unario():
    assert np.allclose(Formula("-a ** 2", VARS).evaluate(DADOS), [-1, -4, -9, -16])


def test_condicional_se():
    f = Formula("se(a > 2, b, 0)", VARS)
    assert np.allclose(f.evaluate(DADOS), [0.0, 0.0, 10.0, 10.0])


def test_funcoes_elementares():
    assert np.allclose(Formula("sqrt(a)", VARS).evaluate(DADOS), np.sqrt(DADOS["a"]))
    assert np.allclose(Formula("abs(-a)", VARS).evaluate(DADOS), DADOS["a"])
    assert np.allclose(Formula("max(a, c)", VARS).evaluate(DADOS), [1, 2, 3, 4])
    assert np.allclose(Formula("min(a, c)", VARS).evaluate(DADOS), [0.5, 0.5, 2, 2])
    assert np.allclose(Formula("clip(a, 2, 3)", VARS).evaluate(DADOS), [2, 2, 3, 3])


def test_operadores_logicos_elementwise():
    f = Formula("se((a > 1) & (c < 1), 1, 0)", VARS)
    assert np.allclose(f.evaluate(DADOS), [0, 1, 0, 0])


def test_fator_de_desconto():
    f = Formula("desconto(c, 2)", VARS)
    assert np.allclose(f.evaluate(DADOS), 1.0 / (1.0 + DADOS["c"]) ** 2)


def test_formula_constante_e_expandida_para_o_tamanho_da_amostra():
    f = Formula("42", VARS)
    out = f.evaluate(DADOS)
    assert out.shape == (4,) and np.all(out == 42.0)


def test_used_names_lista_apenas_o_que_e_usado():
    assert Formula("a + a * 2", VARS).used_names == ["a"]
    assert set(Formula("a + b", VARS).used_names) == {"a", "b"}


def test_divisao_por_zero_produz_nao_finito_sem_levantar_excecao():
    """A simulacao nao deve abortar; os valores nao finitos sao tratados depois."""
    dados = {"a": np.array([1.0, 2.0]), "b": np.array([0.0, 2.0]), "c": np.zeros(2)}
    out = Formula("a / b", VARS).evaluate(dados)
    assert not np.isfinite(out[0]) and np.isfinite(out[1])


# --------------------------------------------------------------------------
# Seguranca: cada teste e um vetor de ataque
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicioso",
    [
        "__import__('os').system('ls')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('x=1')",
        "a.__class__",
        "[x for x in range(10)]",
        "lambda: 1",
        "{'k': 1}",
        "a[0]",
        "globals()",
        "print(1)",
        "compile('1','','eval')",
        "getattr(a, 'shape')",
        "'texto'",
    ],
)
def test_construcoes_perigosas_sao_bloqueadas(malicioso):
    with pytest.raises(FormulaError):
        Formula(malicioso, VARS)


def test_variavel_desconhecida_e_rejeitada():
    with pytest.raises(FormulaError) as e:
        Formula("a + z", VARS)
    assert "z" in str(e.value)


def test_funcao_nao_permitida_e_rejeitada():
    with pytest.raises(FormulaError):
        Formula("sum(a)", VARS)


def test_sintaxe_invalida_e_rejeitada():
    with pytest.raises(FormulaError):
        Formula("a +* b", VARS)


def test_formula_vazia_e_rejeitada():
    with pytest.raises(FormulaError):
        Formula("   ", VARS)


def test_argumentos_nomeados_sao_rejeitados():
    with pytest.raises(FormulaError):
        Formula("clip(a, a_min=1, a_max=2)", VARS)


def test_atribuicao_e_rejeitada():
    with pytest.raises(FormulaError):
        Formula("a = 1", VARS)


# --------------------------------------------------------------------------
# Utilitarios
# --------------------------------------------------------------------------


def test_validate_formula_avisa_sobre_formula_sem_variaveis():
    ok, msg = validate_formula("1 + 1", VARS)
    assert ok and "constante" in msg


def test_validate_formula_retorna_erro_legivel():
    ok, msg = validate_formula("a + nao_existe", VARS)
    assert not ok and "nao_existe" in msg


@pytest.mark.parametrize(
    "entrada,esperado",
    [
        ("Preço de Venda", "Preco_de_Venda"),
        ("custo (R$)", "custo_R"),
        ("2024 receita", "v_2024_receita"),
        ("", "var"),
        ("Índice-Ação", "Indice_Acao"),
    ],
)
def test_sanitize_variable_name(entrada, esperado):
    assert sanitize_variable_name(entrada) == esperado


def test_nome_sanitizado_e_identificador_valido():
    for s in ["Preço!!", "123", "  ", "a-b-c", "ãéî"]:
        assert sanitize_variable_name(s).isidentifier()
