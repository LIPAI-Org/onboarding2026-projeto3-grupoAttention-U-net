"""Persistência e consolidação de resultados experimentais de segmentação."""

from __future__ import annotations

import csv
import math
import numbers
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

import torch.nn as nn

from src.utils.paths import (
    PATH_TABELA_CONSOLIDADA,
    PATH_TABELA_COMPLETA
)
from configs.basicas import (
    TAM_PATCH,
    TAM_BATCH,
    NUM_EPOCAS
)
from src.utils.graficos_globais import calcular_numero_parametros, calcular_gflops

COLUNAS_PLANILHA = (
    "repetition", "seed", "dataset", "task", "model", "encoder",
    "training mode", "augmentation", "loss", "input_size", "epochs",
    "batch_size", "dice_background_test", "dice_foreground_test",
    "mDice_test", "iou_background_test", "iou_foreground_test",
    "mIoU_test", "precision_foreground_test", "recall_foreground_test",
    "num_params", "trainable_params", "gflops", "best_epoch", "val_mDice_best"
)

# Colunas usadas para agrupar as seeds (o que define uma configuração única)
COLUNAS_CONFIGURACAO = (
    "dataset", "task", "model", "encoder", "training mode", 
    "augmentation", "loss", "input_size", "epochs", "batch_size",
    "num_params", "trainable_params", "gflops"
)

# Colunas sobre as quais serão calculadas a média e o desvio padrão
COLUNAS_METRICAS = (
    "dice_background_test", "dice_foreground_test", "mDice_test", 
    "iou_background_test", "iou_foreground_test", "mIoU_test", 
    "precision_foreground_test", "recall_foreground_test", 
    "best_epoch", "val_mDice_best"
)

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
    
    campos_livres_de_limite = ("epoch", "param", "gflops", "batch", "size")
    if not any(campo in nome.lower() for campo in campos_livres_de_limite):
        if not 0.0 <= valor_float <= 1.0:
            raise ValueError(f"{nome} ({valor_float}) deve estar no intervalo [0, 1].")
            
    return valor_float


def _ler_csv(caminho: Path, colunas_esperadas: Iterable[str]) -> list[dict[str, str]]:
    caminho = Path(caminho)
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
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(colunas))
        escritor.writeheader()
        escritor.writerows(linhas)


def consolidar_resultados() -> None:
    linhas = _ler_csv(PATH_TABELA_COMPLETA, COLUNAS_PLANILHA)
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


def adicionar_resultado_completo(
    *,
    modelo: nn.Module,
    dataset: str,
    modo_treinamento: str,
    loss: str,
    augmentation: str,
    seed: int,
    dice_background_test: float,
    dice_foreground_test: float,
    mDice_test: float,
    iou_background_test: float,
    iou_foreground_test: float,
    mIoU_test: float,
    precision_foreground_test: float,
    recall_foreground_test: float,
    best_epoch: int,
    val_mDice_best: float,
    task: str = "segmentacao_binaria"
) -> None:
    nome_modelo = modelo.__class__.__name__
    if nome_modelo == "UNetClassica":
        encoder = "resnet34"
    elif nome_modelo == "AttentionUNet":
        encoder = "custom_from_scratch"
    else:
        encoder = "desconhecido"

    trainable_params = calcular_numero_parametros(modelo)
    num_params = sum(p.numel() for p in modelo.parameters())
    
    tamanho_entrada = (1, 3, TAM_PATCH, TAM_PATCH)
    gflops = calcular_gflops(modelo, tamanho_entrada=tamanho_entrada)

    caminho = Path(PATH_TABELA_COMPLETA)
    linhas = _ler_csv(caminho, COLUNAS_PLANILHA) if caminho.exists() else []

    chave_busca = (str(seed), dataset, nome_modelo, modo_treinamento, augmentation, loss)
    repetition = 1
    for linha in linhas:
        chave_linha = (
            linha["seed"], linha["dataset"], linha["model"], 
            linha["training mode"], linha["augmentation"], linha["loss"]
        )
        if chave_linha == chave_busca:
            repetition += 1

    nova_linha: dict[str, Any] = {
        "repetition": repetition,
        "seed": _validar_seed(seed),
        "dataset": _validar_texto("dataset", dataset),
        "task": _validar_texto("task", task),
        "model": nome_modelo,
        "encoder": encoder,
        "training mode": _validar_texto("modo_treinamento", modo_treinamento),
        "augmentation": _validar_texto("augmentation", augmentation),
        "loss": _validar_texto("loss", loss),
        "input_size": f"{TAM_PATCH}x{TAM_PATCH}",
        "epochs": int(NUM_EPOCAS),
        "batch_size": int(TAM_BATCH),
        "dice_background_test": _validar_metrica("dice_background_test", dice_background_test),
        "dice_foreground_test": _validar_metrica("dice_foreground_test", dice_foreground_test),
        "mDice_test": _validar_metrica("mDice_test", mDice_test),
        "iou_background_test": _validar_metrica("iou_background_test", iou_background_test),
        "iou_foreground_test": _validar_metrica("iou_foreground_test", iou_foreground_test),
        "mIoU_test": _validar_metrica("mIoU_test", mIoU_test),
        "precision_foreground_test": _validar_metrica("precision_foreground_test", precision_foreground_test),
        "recall_foreground_test": _validar_metrica("recall_foreground_test", recall_foreground_test),
        "num_params": num_params,
        "trainable_params": trainable_params,
        "gflops": round(float(gflops), 4),
        "best_epoch": int(best_epoch),
        "val_mDice_best": _validar_metrica("val_mDice_best", val_mDice_best),
    }

    linhas.append(nova_linha)
    _escrever_csv(caminho, COLUNAS_PLANILHA, linhas)