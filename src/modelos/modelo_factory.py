""" Factory de modelos pelo nome """

import torch.nn as nn

from src.modelos.unet import UNetClassica
from src.modelos.attention_unet import AttentionUNet

def pegar_modelo(nome: str) -> nn.Module:
    """
    Três modelos possiveis:

    UNETFS: U-Net classica treinada From Scratch

    UNETPTALL: U-Net classica com uso de pesos da imagenet

    ATUNET: Attention U-Net

    Levanta ValueError caso o nome informado seja diferente.
    """
    if nome.upper() == "UNETFS":
        return UNetClassica(usar_pesos_imagenet=False)
    elif nome.upper() == "UNETPTALL":
        return UNetClassica()
    elif nome.upper() == "ATUNET":
        return AttentionUNet()
    else:
        raise ValueError("Arquitetura desconhecida")
