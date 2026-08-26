"""Gera mosaicos qualitativos com as previsões dos melhores modelos finais."""

from __future__ import annotations

import argparse
import os
import sys
import random

from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from configs.basicas import DEVICE, LIMIAR, NORMALIZACAO_MEAN, NORMALIZACAO_STD
from src.data.dataloader import criar_dataloaders
from src.modelos.attention_unet import AttentionUNet
from src.modelos.unet import UNetClassica
from src.utils.paths import (
    PATH_MELHOR_ATUNET_HE,
    PATH_MELHOR_ATUNET_OEDB,
    PATH_MELHOR_UNET_HE,
    PATH_MELHOR_UNET_OEDB,
    PATH_QUALITATIVE_MOSAICOS,
)

SEED = 42
QUANTIDADE_PADRAO = 6


@dataclass(frozen=True)
class ModelosDataset:
    """Checkpoints associados a um dos conjuntos de teste."""

    dataset: str
    checkpoint_unet: str
    checkpoint_atunet: str


MODELOS_POR_DATASET: Sequence[ModelosDataset] = (
    ModelosDataset("HE", PATH_MELHOR_UNET_HE + ".pt", PATH_MELHOR_ATUNET_HE + ".pt"),
    ModelosDataset("OEDB", PATH_MELHOR_UNET_OEDB + ".pt", PATH_MELHOR_ATUNET_OEDB + ".pt"),
)


def carregar_modelo(modelo: nn.Module, caminho_checkpoint: str) -> nn.Module:
    """Carrega um checkpoint no formato ``{'state_dict': ..., 'mdice': ...}``."""
    if not os.path.isfile(caminho_checkpoint):
        raise FileNotFoundError(f"Checkpoint não encontrado: {caminho_checkpoint}")

    checkpoint = torch.load(caminho_checkpoint, map_location=DEVICE)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Formato de checkpoint inválido: {caminho_checkpoint}")

    modelo.load_state_dict(checkpoint["state_dict"])
    return modelo.to(DEVICE).eval()


def carregar_modelos(especificacao: ModelosDataset) -> tuple[nn.Module, nn.Module]:
    """Carrega U-Net e Attention U-Net finais para um dataset."""
    unet = carregar_modelo(
        UNetClassica(usar_pesos_imagenet=False),
        especificacao.checkpoint_unet,
    )
    atunet = carregar_modelo(
        AttentionUNet(),
        especificacao.checkpoint_atunet,
    )
    return unet, atunet


def desnormalizar(imagem: torch.Tensor) -> np.ndarray:
    """Converte tensor normalizado ``C x H x W`` em imagem RGB para exibição."""
    media = torch.tensor(NORMALIZACAO_MEAN, dtype=imagem.dtype).view(3, 1, 1)
    desvio = torch.tensor(NORMALIZACAO_STD, dtype=imagem.dtype).view(3, 1, 1)
    imagem_rgb = (imagem.detach().cpu() * desvio + media).clamp(0, 1)
    return imagem_rgb.permute(1, 2, 0).numpy()


@torch.no_grad()
def prever(modelo: nn.Module, imagem: torch.Tensor) -> np.ndarray:
    """Retorna a máscara binária prevista usando o limiar configurado."""
    logits = modelo(imagem.unsqueeze(0).to(DEVICE))
    tamanho_esperado = (1, 1, *imagem.shape[-2:])

    if tuple(logits.shape) != tamanho_esperado:
        raise ValueError(
            "A predição deve ter formato [1, 1, H, W]; "
            f"recebido {tuple(logits.shape)}."
        )

    probabilidades = torch.sigmoid(logits)
    return (probabilidades.squeeze(0).squeeze(0) >= LIMIAR).cpu().numpy()


def selecionar_indices(total: int, quantidade: int, seed: int) -> list[int]:
    """Seleciona amostras aleatórias reproduzíveis."""
    if total == 0:
        raise ValueError("O conjunto de teste não contém pares válidos de imagem e máscara.")

    return random.Random(seed).sample(range(total), k=min(quantidade, total))


def gerar_mosaico(especificacao: ModelosDataset, quantidade: int, seed: int) -> str:
    """Salva um mosaico do conjunto de teste para um dataset."""
    _, _, dataloader_teste = criar_dataloaders(
        nome_dataset=especificacao.dataset,
        num_workers=0,
        aplicar_aug=False,
    )

    dataset_teste = dataloader_teste.dataset
    indices = selecionar_indices(len(dataset_teste), quantidade, seed)
    unet, atunet = carregar_modelos(especificacao)

    figura, eixos = plt.subplots(
        len(indices),
        4,
        figsize=(14, 3.5 * len(indices)),
        squeeze=False,
    )

    titulos = ("Imagem", "Ground Truth", "U-Net", "Attention U-Net")
    for coluna, titulo in enumerate(titulos):
        eixos[0, coluna].set_title(titulo, fontweight="bold")

    for linha, indice in enumerate(indices):
        imagem, mascara = dataset_teste[indice]

        predicao_unet = prever(unet, imagem)
        predicao_atunet = prever(atunet, imagem)

        paineis = (
            desnormalizar(imagem),
            mascara.squeeze(0).numpy(),
            predicao_unet,
            predicao_atunet,
        )

        for coluna, painel in enumerate(paineis):
            eixo = eixos[linha, coluna]

            if coluna == 0:
                eixo.imshow(painel)
            else:
                eixo.imshow(painel, cmap="gray", vmin=0, vmax=1)

            eixo.axis("off")

        eixos[linha, 0].set_ylabel(f"Teste #{indice}", rotation=90, labelpad=12)

    figura.suptitle(
        f"Segmentação qualitativa — {especificacao.dataset}",
        fontsize=16,
        fontweight="bold",
    )

    figura.tight_layout(rect=(0, 0, 1, 0.97))

    diretorio_saida = os.path.join(
        PATH_QUALITATIVE_MOSAICOS,
        especificacao.dataset,
    )
    os.makedirs(diretorio_saida, exist_ok=True)

    caminho_saida = os.path.join(
        diretorio_saida,
        f"mosaico_{especificacao.dataset.lower()}.png",
    )

    figura.savefig(caminho_saida, dpi=180, bbox_inches="tight")
    plt.close(figura)

    return caminho_saida


def script_gerar_mosaicos() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantidade",
        type=int,
        default=QUANTIDADE_PADRAO,
        help="Número máximo de imagens por dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed da seleção de exemplos.",
    )

    argumentos = parser.parse_args()

    if argumentos.quantidade < 1:
        parser.error("--quantidade deve ser maior que zero.")

    for especificacao in MODELOS_POR_DATASET:
        caminho = gerar_mosaico(
            especificacao,
            argumentos.quantidade,
            argumentos.seed,
        )
        print(f"Mosaico salvo em: {caminho}")


if __name__ == "__main__":
    script_gerar_mosaicos()