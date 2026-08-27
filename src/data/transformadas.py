"""
Lida com a lógica de pré-processamento e data augmentations
"""

from typing import List, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.basicas import TAM_PATCH, NORMALIZACAO_MEAN, NORMALIZACAO_STD


def obter_transformacoes(aplicar_aug: bool) -> A.Compose:
    """
    Pega as transformações, caso utilize.
    """
    passos: List[Any] = [
        A.Resize(height=TAM_PATCH, width=TAM_PATCH, p=1.0)
    ]

    if aplicar_aug:
        passos.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    passos.extend([
        A.Normalize(mean=NORMALIZACAO_MEAN, std=NORMALIZACAO_STD),
        ToTensorV2()
    ])

    return A.Compose(passos)


def obter_transformacao_visualizacao(nome_aumento: str) -> A.Compose:
    """Cria uma transformação geométrica determinística para inspeção visual."""
    aumentos = {
        "horizontal_flip": A.HorizontalFlip(p=1.0),
        "vertical_flip": A.VerticalFlip(p=1.0),
        "rotate_90": A.RandomRotate90(p=1.0),
    }

    try:
        aumento = aumentos[nome_aumento]
    except KeyError as erro:
        raise ValueError(f"Aumento desconhecido: {nome_aumento}") from erro

    return A.Compose([
        A.Resize(height=TAM_PATCH, width=TAM_PATCH, p=1.0),
        aumento,
    ])