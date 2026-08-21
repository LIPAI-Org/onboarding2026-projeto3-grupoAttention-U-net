"""Persistência e consolidação de resultados experimentais de segmentação."""

from __future__ import annotations

import csv
import math
import numbers
import tempfile
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from src.utils.paths import (
    PATH_TABELA_COMPLETA,
    PATH_TABELA_CONSOLIDADA
)

COLUNAS_EXECUCOES = (
    "arquitetura",
    "dataset",
    "modo_treinamento",
    "loss",
    "augmentation",
    "seed",
    "mdice",
    "miou",
    "dice_classe_1",
    "iou_classe_1",
    "precision_classe_1",
    "recall_classe_1",
)

COLUNAS_CONFIGURACAO = COLUNAS_EXECUCOES[:5]
COLUNAS_METRICAS = COLUNAS_EXECUCOES[6:]
SEEDS_ESPERADAS = (42, 123, 2025)
COLUNAS_CONSOLIDADOS = (
    *COLUNAS_CONFIGURACAO,
    *(coluna for metrica in COLUNAS_METRICAS for coluna in (f"{metrica}_media", f"{metrica}_std")),
)

def _validar_texto(nome: str, valor: Any) -> str:
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{nome} deve ser uma string não vazia.")
    return valor.strip()


def _validar_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed deve ser um número inteiro.")
    return int(seed)


def _validar_metrica(nome: str, valor: Any) -> float:
    if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
        raise TypeError(f"{nome} deve ser numérica.")
    valor_float = float(valor)
    if not math.isfinite(valor_float):
        raise ValueError(f"{nome} não pode ser NaN ou infinito.")
    if not 0.0 <= valor_float <= 1.0:
        raise ValueError(f"{nome} deve estar no intervalo [0, 1].")
    return valor_float


def _ler_csv(caminho: Path, colunas_esperadas: Iterable[str]) -> list[dict[str, str]]:
    if not caminho.exists():
        return []

    with caminho.open("r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        if leitor.fieldnames != list(colunas_esperadas):
            raise ValueError(
                f"Cabeçalho inválido em {caminho}. Esperado: {', '.join(colunas_esperadas)}."
            )
        return list(leitor)


def _escrever_csv(caminho: Path, colunas: Iterable[str], linhas: Iterable[Mapping[str, Any]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(colunas))
        escritor.writeheader()
        escritor.writerows(linhas)


def adicionar_resultado(
    *,
    arquitetura: str,
    dataset: str,
    modo_treinamento: str,
    loss: str,
    augmentation: str,
    seed: int,
    mdice: float,
    miou: float,
    dice_classe_1: float,
    iou_classe_1: float,
    precision_classe_1: float,
    recall_classe_1: float,
) -> None:
    """ Retorna ValueError caso já tenha executado antes, tratar! """
    linha: dict[str, Any] = {
        "arquitetura": _validar_texto("arquitetura", arquitetura),
        "dataset": _validar_texto("dataset", dataset),
        "modo_treinamento": _validar_texto("modo_treinamento", modo_treinamento),
        "loss": _validar_texto("loss", loss),
        "augmentation": _validar_texto("augmentation", augmentation),
        "seed": _validar_seed(seed),
        "mdice": _validar_metrica("mdice", mdice),
        "miou": _validar_metrica("miou", miou),
        "dice_classe_1": _validar_metrica("dice_classe_1", dice_classe_1),
        "iou_classe_1": _validar_metrica("iou_classe_1", iou_classe_1),
        "precision_classe_1": _validar_metrica("precision_classe_1", precision_classe_1),
        "recall_classe_1": _validar_metrica("recall_classe_1", recall_classe_1),
    }
    caminho = PATH_TABELA_COMPLETA
    linhas = _ler_csv(caminho, COLUNAS_EXECUCOES)
    chave = tuple(str(linha[coluna]) for coluna in (*COLUNAS_CONFIGURACAO, "seed"))

    for existente in linhas:
        chave_existente = tuple(existente[coluna] for coluna in (*COLUNAS_CONFIGURACAO, "seed"))
        if chave_existente == chave:
            raise ValueError("Esta execução já está registrada para a mesma configuração e seed.")

    linhas.append(linha)
    _escrever_csv(caminho, COLUNAS_EXECUCOES, linhas)


def consolidar_resultados() -> None:
    """ Retorna ValueError se uma seed encontrada for diferente das esperadas """
    linhas = _ler_csv(PATH_TABELA_COMPLETA, COLUNAS_EXECUCOES)
    grupos: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for linha in linhas:
        chave = tuple(linha[coluna] for coluna in COLUNAS_CONFIGURACAO)
        grupos.setdefault(chave, []).append(linha)

    consolidados: list[dict[str, float | str]] = []
    for chave, execucoes in grupos.items():
        seeds = [_validar_seed(int(execucao["seed"])) for execucao in execucoes]
        if len(execucoes) != len(SEEDS_ESPERADAS) or set(seeds) != set(SEEDS_ESPERADAS):
            raise ValueError(
                f"A configuração {chave} deve conter exatamente as seeds {SEEDS_ESPERADAS}; encontradas: {sorted(seeds)}."
            )

        linha_consolidada: dict[str, float | str] = dict(zip(COLUNAS_CONFIGURACAO, chave))
        for metrica in COLUNAS_METRICAS:
            valores = [_validar_metrica(metrica, float(execucao[metrica])) for execucao in execucoes]
            linha_consolidada[f"{metrica}_media"] = mean(valores)
            linha_consolidada[f"{metrica}_std"] = stdev(valores)
        consolidados.append(linha_consolidada)

    consolidados.sort(key=lambda linha: tuple(str(linha[coluna]) for coluna in COLUNAS_CONFIGURACAO))
    _escrever_csv(PATH_TABELA_CONSOLIDADA, COLUNAS_CONSOLIDADOS, consolidados)

# Script de teste
# def executar_testes_basicos() -> None:
#     with tempfile.TemporaryDirectory() as diretorio_temporario:
#         diretorio = Path(diretorio_temporario)
#         execucoes = diretorio / "execucoes.csv"
#         consolidados = diretorio / "consolidados.csv"
#         base = {
#             "arquitetura": "Attention U-Net",
#             "dataset": "OEDB",
#             "modo_treinamento": "do zero",
#             "loss": "BCE",
#             "augmentation": "sim",
#         }
#         for seed, mdice, miou, dice, iou, precision, recall in (
#             (42, 0.80, 0.70, 0.79, 0.65, 0.82, 0.77),
#             (123, 0.82, 0.72, 0.81, 0.67, 0.84, 0.79),
#             (2025, 0.84, 0.74, 0.83, 0.69, 0.86, 0.81),
#         ):
#             adicionar_resultado(
#                 **base, seed=seed, mdice=mdice, miou=miou, dice_classe_1=dice,
#                 iou_classe_1=iou, precision_classe_1=precision, recall_classe_1=recall,
#                 caminho_execucoes=execucoes,
#             )

#         try:
#             adicionar_resultado(
#                 **base, seed=42, mdice=0.80, miou=0.70, dice_classe_1=0.79,
#                 iou_classe_1=0.65, precision_classe_1=0.82, recall_classe_1=0.77,
#                 caminho_execucoes=execucoes,
#             )
#         except ValueError:
#             pass
#         else:
#             raise AssertionError("A duplicação deveria ter sido impedida.")

#         consolidar_resultados(caminho_execucoes=execucoes, caminho_consolidados=consolidados)
#         resultado = _ler_csv(consolidados, COLUNAS_CONSOLIDADOS)
#         assert len(resultado) == 1
#         assert "seed" not in resultado[0]
#         assert "iou_classe_1_media" in resultado[0]
#         assert math.isclose(float(resultado[0]["mdice_media"]), 0.82)
#         assert math.isclose(float(resultado[0]["mdice_std"]), 0.02)
