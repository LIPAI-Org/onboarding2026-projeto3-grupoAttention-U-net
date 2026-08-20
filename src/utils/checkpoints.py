import os

import torch

from src.utils.paths import PATH_MODELOS, PATH_MELHOR_ATUNET, PATH_MELHOR_UNET
from src.modelos.unet import UNetClassica
from src.modelos.attention_unet import AttentionUNet

def salvar_checkpoint(modelo, mdice, nome_arquivo):
    os.makedirs(PATH_MODELOS, exist_ok=True)
    caminho = os.path.join(PATH_MODELOS, f"{nome_arquivo}.pt")
    checkpoint = {'state_dict': modelo.state_dict(), 'mdice': mdice}
    torch.save(checkpoint, caminho)
    caminho_melhor = None
    if isinstance(modelo, AttentionUNet):
        caminho_melhor = PATH_MELHOR_ATUNET
    elif isinstance(modelo, UNetClassica):
        caminho_melhor = PATH_MELHOR_UNET
    if caminho_melhor is not None:
        if not os.path.exists(caminho_melhor):
            torch.save(checkpoint, caminho_melhor)
        else:
            melhor_checkpoint = torch.load(caminho_melhor)
            if mdice > melhor_checkpoint['mdice']:
                torch.save(checkpoint, caminho_melhor)

def carregar_checkpoint(modelo, nome_arquivo, device='cpu'):
    caminho = os.path.join(PATH_MODELOS, f"{nome_arquivo}.pt")
    checkpoint = torch.load(caminho, map_location=device)
    modelo.load_state_dict(checkpoint['state_dict'])
    return modelo, checkpoint['mdice']
