"""
Lida com a lógica de pré-processamento e data augmentations
"""

from typing import List, Any

import albumentations as A

from albumentations.pytorch import ToTensorV2

from configs.basicas import TAM_PATCH, NORMALIZACAO_MEAN, NORMALIZACAO_STD

def obter_transformacoes(aplicar_aug: bool) -> A.Compose:
    passos: List[Any] = [
        A.Resize(height=TAM_PATCH, width=TAM_PATCH, always_apply=True)
    ]

    if aplicar_aug:
        passos.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    passos.extend([
        A.Normalize(mean=NORMALIZACAO_MEAN, std=NORMALIZACAO_STD, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    return A.Compose(passos)
