"""Gráficos comparativos globais para os resultados de segmentação."""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils.paths import (
    PATH_GRAFICOS_GLOBAIS_MDICE,
    PATH_GRAFICOS_GLOBAIS_MIOU,
    PATH_GRAFICOS_GLOBAIS_GFLOPS_PARAMETROS,
    PATH_TABELA_CONSOLIDADA,
)

COLUNAS_CONFIGURACAO = (
    "arquitetura",
    "dataset",
    "modo_treinamento",
    "loss",
    "augmentation",
)
COLUNAS_METRICAS = (
    "mdice",
    "miou",
    "dice_classe_1",
    "iou_classe_1",
    "precision_classe_1",
    "recall_classe_1",
)
COLUNAS_CONSOLIDADOS = (
    *COLUNAS_CONFIGURACAO,
    *(coluna for metrica in COLUNAS_METRICAS for coluna in (f"{metrica}_media", f"{metrica}_std")),
)


def _slug(texto: str) -> str:
    resultado = re.sub(r"[^A-Za-z0-9_-]+", "_", texto.strip())
    return resultado.strip("_") or "dataset"


def _validar_valor_metrica(nome: str, valor: Any) -> float:
    try:
        valor_float = float(valor)
    except (TypeError, ValueError) as erro:
        raise ValueError(f"{nome} deve ser numérico.") from erro
    if not math.isfinite(valor_float) or not 0.0 <= valor_float <= 1.0:
        raise ValueError(f"{nome} deve ser finito e pertencer ao intervalo [0, 1].")
    return valor_float


def ler_resultados_consolidados(caminho: str | Path | None = None) -> list[dict[str, str]]:
    caminho_csv = Path(caminho) if caminho is not None else Path(PATH_TABELA_CONSOLIDADA)
    if not caminho_csv.is_file():
        raise FileNotFoundError(f"CSV consolidado não encontrado: {caminho_csv}")

    with caminho_csv.open("r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        colunas_ausentes = set(COLUNAS_CONSOLIDADOS) - set(leitor.fieldnames or ())
        if colunas_ausentes:
            raise ValueError(f"CSV consolidado sem as colunas: {', '.join(sorted(colunas_ausentes))}.")
        linhas = list(leitor)
    if not linhas:
        raise ValueError("CSV consolidado não possui resultados para plotar.")
    return linhas


def identificar_datasets(resultados: Sequence[Mapping[str, Any]]) -> list[str]:
    datasets = {str(linha.get("dataset", "")).strip() for linha in resultados}
    datasets.discard("")
    if not datasets:
        raise ValueError("Nenhum dataset válido foi encontrado nos resultados.")
    return sorted(datasets)


def _rotulo_configuracao(linha: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            str(linha["arquitetura"]),
            str(linha["modo_treinamento"]),
            f"{linha['loss']} | {linha['augmentation']}",
        )
    )


def _gerar_grafico_metrica(
    resultados: Sequence[Mapping[str, Any]],
    metrica: str,
    titulo: str,
    diretorio_saida: Path,
) -> dict[str, Path]:
    col_media, col_std = f"{metrica}_media", f"{metrica}_std"
    arquivos: dict[str, Path] = {}

    for dataset in identificar_datasets(resultados):
        linhas = [linha for linha in resultados if str(linha["dataset"]).strip() == dataset]
        medias = [_validar_valor_metrica(col_media, linha.get(col_media)) for linha in linhas]
        desvios = [_validar_valor_metrica(col_std, linha.get(col_std)) for linha in linhas]
        rotulos = [_rotulo_configuracao(linha) for linha in linhas]
        
        max_por_arq = {}
        for linha, media in zip(linhas, medias):
            arq = str(linha.get("arquitetura", "Desconhecida")).strip()
            if arq not in max_por_arq or media > max_por_arq[arq]:
                max_por_arq[arq] = media

        largura = max(9.0, len(linhas) * 2.15)
        tamanho_fonte_rotulos = 9 if len(linhas) <= 6 else 8 if len(linhas) <= 12 else 7

        figura, eixo = plt.subplots(figsize=(largura, 7.0))
        posicoes = range(len(linhas))
        eixo.bar(posicoes, medias, yerr=desvios, capsize=5)
        
        cores = plt.cm.tab10.colors
        for i, (arq, valor_max) in enumerate(max_por_arq.items()):
            cor = cores[i % len(cores)]
            eixo.axhline(
                y=valor_max, 
                color=cor, 
                linestyle="--", 
                linewidth=1.5, 
                alpha=0.8, 
                label=f"Máx {arq}: {valor_max:.3f}"
            )

        eixo.set_title(f"{titulo} - Dataset {dataset}")
        eixo.set_xlabel("Configuração experimental")
        eixo.set_ylabel(titulo)
        eixo.set_ylim(0.0, 1.0)
        eixo.set_xticks(list(posicoes), rotulos, rotation=28, ha="right", fontsize=tamanho_fonte_rotulos)
        eixo.grid(axis="y", linestyle="--", alpha=0.45)
        eixo.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        eixo.margins(x=0.02)
        figura.tight_layout()

        destino = diretorio_saida / f"{metrica}_{_slug(dataset)}.png"
        destino.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(destino, dpi=300, bbox_inches="tight")
        plt.close(figura)
        arquivos[dataset] = destino
        
    return arquivos


def gerar_grafico_mdice(
    resultados: Sequence[Mapping[str, Any]] | None = None,
    *,
    caminho_consolidado: str | Path | None = None,
    diretorio_saida: str | Path | None = None,
) -> dict[str, Path]:
    linhas = list(resultados) if resultados is not None else ler_resultados_consolidados(caminho_consolidado)
    saida = Path(diretorio_saida) if diretorio_saida is not None else Path(PATH_GRAFICOS_GLOBAIS_MDICE)
    return _gerar_grafico_metrica(linhas, "mdice", "mDice", saida)


def gerar_grafico_miou(
    resultados: Sequence[Mapping[str, Any]] | None = None,
    *,
    caminho_consolidado: str | Path | None = None,
    diretorio_saida: str | Path | None = None,
) -> dict[str, Path]:
    linhas = list(resultados) if resultados is not None else ler_resultados_consolidados(caminho_consolidado)
    saida = Path(diretorio_saida) if diretorio_saida is not None else Path(PATH_GRAFICOS_GLOBAIS_MIOU)
    return _gerar_grafico_metrica(linhas, "miou", "mIoU", saida)


def calcular_numero_parametros(modelo: nn.Module) -> int:
    if not isinstance(modelo, nn.Module):
        raise TypeError("modelo deve ser uma instância de torch.nn.Module.")
    return sum(parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad)


def _validar_modelos(modelos: Mapping[str, nn.Module]) -> None:
    if not isinstance(modelos, Mapping) or not modelos:
        raise ValueError("modelos deve ser um mapeamento não vazio de nomes para modelos.")
    if any(not isinstance(nome, str) or not nome.strip() for nome in modelos):
        raise ValueError("Cada arquitetura deve possuir um nome não vazio.")
    if any(not isinstance(modelo, nn.Module) for modelo in modelos.values()):
        raise TypeError("Todos os valores de modelos devem ser torch.nn.Module.")


def _gerar_grafico_barras(
    valores: Mapping[str, float],
    titulo: str,
    eixo_y: str,
    destino: Path,
) -> Path:
    figura, eixo = plt.subplots(figsize=(max(7.0, len(valores) * 2.2), 5.5))
    nomes, numeros = list(valores.keys()), list(valores.values())
    barras = eixo.bar(nomes, numeros)
    eixo.set_title(titulo)
    eixo.set_xlabel("Arquitetura")
    eixo.set_ylabel(eixo_y)
    eixo.grid(axis="y", linestyle="--", alpha=0.45)
    for barra, valor in zip(barras, numeros):
        eixo.annotate(f"{valor:.2f}", (barra.get_x() + barra.get_width() / 2, valor),
                      xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(figura)
    return destino


def gerar_grafico_parametros(
    modelos: Mapping[str, nn.Module], *, diretorio_saida: str | Path | None = None
) -> Path:
    _validar_modelos(modelos)
    valores_milhoes = {nome: calcular_numero_parametros(modelo) / 1_000_000 for nome, modelo in modelos.items()}
    saida = Path(diretorio_saida) if diretorio_saida is not None else Path(PATH_GRAFICOS_GLOBAIS_GFLOPS_PARAMETROS)
    destino = saida / "numero_parametros.png"
    return _gerar_grafico_barras(valores_milhoes, "Número de parâmetros treináveis", "Parâmetros treináveis (M)", destino)


def _validar_tamanho_entrada(tamanho_entrada: Sequence[int]) -> tuple[int, int, int, int]:
    if len(tamanho_entrada) != 4 or any(isinstance(valor, bool) or not isinstance(valor, int) or valor <= 0 for valor in tamanho_entrada):
        raise ValueError("tamanho_entrada deve ter quatro inteiros positivos: [N, C, H, W].")
    return tuple(tamanho_entrada)  # type: ignore[return-value]


def calcular_gflops(
    modelo: nn.Module,
    tamanho_entrada: Sequence[int] = (1, 3, 256, 256),
) -> float:
    if not isinstance(modelo, nn.Module):
        raise TypeError("modelo deve ser uma instância de torch.nn.Module.")
    tamanho = _validar_tamanho_entrada(tamanho_entrada)
    try:
        from fvcore.nn import FlopCountAnalysis
    except ModuleNotFoundError as erro:
        raise RuntimeError("A biblioteca 'fvcore' é necessária para calcular GFLOPs.") from erro

    tensor_referencia = next(modelo.parameters(), next(modelo.buffers(), None))
    device = tensor_referencia.device if tensor_referencia is not None else torch.device("cpu")
    entrada = torch.zeros(tamanho, device=device)
    estava_em_treino = modelo.training
    try:
        modelo.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(modelo, entrada).total()
    finally:
        modelo.train(estava_em_treino)
    return float(flops) / 1_000_000_000


def gerar_grafico_gflops(
    modelos: Mapping[str, nn.Module],
    *,
    tamanho_entrada: Sequence[int] = (1, 3, 256, 256),
    diretorio_saida: str | Path | None = None,
) -> Path:
    _validar_modelos(modelos)
    tamanho = _validar_tamanho_entrada(tamanho_entrada)
    valores = {nome: calcular_gflops(modelo, tamanho) for nome, modelo in modelos.items()}
    saida = Path(diretorio_saida) if diretorio_saida is not None else Path(PATH_GRAFICOS_GLOBAIS_GFLOPS_PARAMETROS)
    destino = saida / f"gflops_{tamanho[2]}x{tamanho[3]}.png"
    return _gerar_grafico_barras(valores, f"Complexidade computacional ({tamanho[2]} x {tamanho[3]} RGB)", "GFLOPs", destino)


def gerar_graficos_globais(
    modelos: Mapping[str, nn.Module],
    *,
    caminho_consolidado: str | Path | None = None,
    tamanho_entrada: Sequence[int] = (1, 3, 256, 256),
) -> dict[str, Any]:
    """
    Gera todos os gráficos globais disponíveis.
    """
    resultados = ler_resultados_consolidados(caminho_consolidado)
    return {
        "mdice": gerar_grafico_mdice(resultados),
        "miou": gerar_grafico_miou(resultados),
        "parametros": gerar_grafico_parametros(modelos),
        "gflops": gerar_grafico_gflops(modelos, tamanho_entrada=tamanho_entrada),
    }
