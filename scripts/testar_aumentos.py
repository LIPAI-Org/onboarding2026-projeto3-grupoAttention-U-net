"""Gera figuras para inspeção visual da sincronização imagem-máscara."""

from __future__ import annotations

import argparse
import os
import random
from typing import Sequence

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from configs.basicas import TAM_PATCH
from src.data.dataloader import criar_dataloaders
from src.data.transformadas import obter_transformacao_visualizacao
from src.utils.paths import PATH_QUALITATIVE_AUGMENTATIONS

SEED = 42
QUANTIDADE_PADRAO = 3

AUMENTOS: Sequence[tuple[str, str]] = (
    ("horizontal_flip", "Horizontal Flip"),
    ("vertical_flip", "Vertical Flip"),
    ("rotate_90", "Rotação aleatória de 90°"),
)


def carregar_par_original(
    caminho_imagem: str,
    caminho_mascara: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Lê e redimensiona um par preservando seu alinhamento espacial."""
    imagem = np.array(Image.open(caminho_imagem).convert("RGB"))
    mascara = np.array(Image.open(caminho_mascara).convert("L"))

    redimensionar = A.Compose([
        A.Resize(height=TAM_PATCH, width=TAM_PATCH, p=1.0),
    ])

    resultado = redimensionar(image=imagem, mask=mascara)
    return resultado["image"], resultado["mask"]


def selecionar_indices(total: int, quantidade: int, seed: int) -> list[int]:
    """Seleciona pares de teste de modo reproduzível."""
    if total == 0:
        raise ValueError("O conjunto de teste não contém pares válidos de imagem e máscara.")

    return random.Random(seed).sample(range(total), k=min(quantidade, total))


def salvar_verificacao(
    dataset: str,
    indice: int,
    caminho_imagem: str,
    caminho_mascara: str,
) -> str:
    """Cria uma figura com uma linha para cada augmentation geométrica."""
    imagem_original, mascara_original = carregar_par_original(
        caminho_imagem,
        caminho_mascara,
    )

    figura, eixos = plt.subplots(
        len(AUMENTOS),
        4,
        figsize=(14, 3.5 * len(AUMENTOS)),
        squeeze=False,
    )

    titulos = (
        "Imagem original",
        "Máscara original",
        "Imagem aumentada",
        "Máscara aumentada",
    )

    for coluna, titulo in enumerate(titulos):
        eixos[0, coluna].set_title(titulo, fontweight="bold")

    for linha, (nome_aumento, rotulo) in enumerate(AUMENTOS):
        transformacao = obter_transformacao_visualizacao(nome_aumento)

        resultado = transformacao(
            image=imagem_original,
            mask=mascara_original,
        )

        imagem_aumentada = resultado["image"]
        mascara_aumentada = resultado["mask"]

        paineis = (
            imagem_original,
            mascara_original,
            imagem_aumentada,
            mascara_aumentada,
        )

        for coluna, painel in enumerate(paineis):
            eixo = eixos[linha, coluna]

            if coluna in (0, 2):
                eixo.imshow(painel)
            else:
                eixo.imshow(painel, cmap="gray", vmin=0, vmax=255)

            eixo.axis("off")

        eixos[linha, 0].set_ylabel(
            rotulo,
            rotation=90,
            labelpad=12,
        )

    figura.suptitle(
        f"Sincronização das augmentations — {dataset} — teste #{indice}",
        fontsize=15,
        fontweight="bold",
    )

    figura.tight_layout(rect=(0, 0, 1, 0.96))

    diretorio_saida = os.path.join(
        PATH_QUALITATIVE_AUGMENTATIONS,
        dataset,
    )
    os.makedirs(diretorio_saida, exist_ok=True)

    caminho_saida = os.path.join(
        diretorio_saida,
        f"augmentations_teste_{indice}.png",
    )

    figura.savefig(caminho_saida, dpi=180, bbox_inches="tight")
    plt.close(figura)

    return caminho_saida


def gerar_verificacoes(
    dataset: str,
    quantidade: int,
    seed: int,
) -> list[str]:
    """Usa exclusivamente pares do conjunto de teste para gerar as figuras."""
    _, _, dataloader_teste = criar_dataloaders(
        dataset,
        num_workers=0,
        aplicar_aug=False,
    )

    dataset_teste = dataloader_teste.dataset
    caminhos_salvos: list[str] = []

    for indice in selecionar_indices(
        len(dataset_teste),
        quantidade,
        seed,
    ):
        caminho_imagem, caminho_mascara = dataset_teste.amostras[indice]

        caminhos_salvos.append(
            salvar_verificacao(
                dataset,
                indice,
                caminho_imagem,
                caminho_mascara,
            )
        )

    return caminhos_salvos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantidade",
        type=int,
        default=QUANTIDADE_PADRAO,
        help="Número máximo de pares por dataset.",
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

    for dataset in ("HE", "OEDB"):
        for caminho in gerar_verificacoes(
            dataset,
            argumentos.quantidade,
            argumentos.seed,
        ):
            print(f"Verificação salva em: {caminho}")


if __name__ == "__main__":
    main()