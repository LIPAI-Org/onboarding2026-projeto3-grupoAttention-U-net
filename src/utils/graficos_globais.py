"""Gráficos comparativos globais para os resultados de segmentação."""

from __future__ import annotations

import csv
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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


def _diretorios_padrao() -> tuple[Path, Path]:
    try:
        from src.utils.paths import PATH_PLOTS, PATH_RESULTS

        return Path(PATH_RESULTS) / "metrics", Path(PATH_PLOTS) / "graficos_globais"
    except ModuleNotFoundError:
        raiz = Path(__file__).resolve().parents[2]
        return raiz / "results" / "metrics", raiz / "results" / "plots" / "graficos_globais"


def _caminho_consolidado(caminho: str | Path | None) -> Path:
    return Path(caminho) if caminho is not None else _diretorios_padrao()[0] / "resultados_consolidados.csv"


def _diretorio_saida(diretorio: str | Path | None) -> Path:
    return Path(diretorio) if diretorio is not None else _diretorios_padrao()[1]


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
    caminho_csv = _caminho_consolidado(caminho)
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
        largura = max(9.0, len(linhas) * 2.15)
        tamanho_fonte_rotulos = 9 if len(linhas) <= 6 else 8 if len(linhas) <= 12 else 7

        figura, eixo = plt.subplots(figsize=(largura, 7.0))
        posicoes = range(len(linhas))
        eixo.bar(posicoes, medias, yerr=desvios, capsize=5)
        eixo.set_title(f"{titulo} - Dataset {dataset}")
        eixo.set_xlabel("Configuração experimental")
        eixo.set_ylabel(titulo)
        eixo.set_ylim(0.0, 1.0)
        eixo.set_xticks(list(posicoes), rotulos, rotation=28, ha="right", fontsize=tamanho_fonte_rotulos)
        eixo.grid(axis="y", linestyle="--", alpha=0.45)
        eixo.margins(x=0.02)
        figura.tight_layout()

        destino = diretorio_saida / metrica / f"{metrica}_{_slug(dataset)}.png"
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
    return _gerar_grafico_metrica(linhas, "mdice", "mDice", _diretorio_saida(diretorio_saida))


def gerar_grafico_miou(
    resultados: Sequence[Mapping[str, Any]] | None = None,
    *,
    caminho_consolidado: str | Path | None = None,
    diretorio_saida: str | Path | None = None,
) -> dict[str, Path]:
    linhas = list(resultados) if resultados is not None else ler_resultados_consolidados(caminho_consolidado)
    return _gerar_grafico_metrica(linhas, "miou", "mIoU", _diretorio_saida(diretorio_saida))


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
    destino = _diretorio_saida(diretorio_saida) / "parametros" / "numero_parametros.png"
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
        raise RuntimeError(
        ) from erro

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
    destino = _diretorio_saida(diretorio_saida) / "gflops" / "gflops_{}x{}.png".format(tamanho[2], tamanho[3])
    return _gerar_grafico_barras(valores, f"Complexidade computacional ({tamanho[2]} x {tamanho[3]} RGB)", "GFLOPs", destino)


def gerar_graficos_globais(
    *,
    caminho_consolidado: str | Path | None = None,
    diretorio_saida: str | Path | None = None,
) -> dict[str, dict[str, Path]]:
    resultados = ler_resultados_consolidados(caminho_consolidado)
    return {
        "mdice": gerar_grafico_mdice(resultados, diretorio_saida=diretorio_saida),
        "miou": gerar_grafico_miou(resultados, diretorio_saida=diretorio_saida),
    }


def executar_testes_basicos() -> None:
    linhas = [
        {
            "arquitetura": "Modelo A", "dataset": "HE", "modo_treinamento": "do zero", "loss": "BCE", "augmentation": "não",
            "mdice_media": "0.80", "mdice_std": "0.02", "miou_media": "0.70", "miou_std": "0.03",
            "dice_classe_1_media": "0.79", "dice_classe_1_std": "0.02", "iou_classe_1_media": "0.65", "iou_classe_1_std": "0.03",
            "precision_classe_1_media": "0.81", "precision_classe_1_std": "0.02", "recall_classe_1_media": "0.78", "recall_classe_1_std": "0.02",
        }
    ]
    with tempfile.TemporaryDirectory() as diretorio_temporario:
        diretorio = Path(diretorio_temporario)
        csv_teste = diretorio / "consolidados.csv"
        with csv_teste.open("w", newline="", encoding="utf-8") as arquivo:
            escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_CONSOLIDADOS)
            escritor.writeheader()
            escritor.writerows(linhas)

        assert identificar_datasets(ler_resultados_consolidados(csv_teste)) == ["HE"]
        assert (gerar_grafico_mdice(linhas, diretorio_saida=diretorio)["HE"]).is_file()
        assert (gerar_grafico_miou(linhas, diretorio_saida=diretorio)["HE"]).is_file()

        modelo = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        assert calcular_numero_parametros(modelo) == 10
        try:
            calcular_gflops(modelo, tamanho_entrada=(1, 3, 256))
        except ValueError:
            pass
        else:
            raise AssertionError("O tamanho de entrada inválido deveria gerar ValueError.")


if __name__ == "__main__":
    executar_testes_basicos()
    print("Testes básicos concluídos com sucesso.")
