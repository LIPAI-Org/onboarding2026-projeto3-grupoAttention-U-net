"""
Lida com o mapeamento, a leitura e o carregamento dos arquivos
"""

import os

from typing import Tuple, List

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from src.data.transformadas import obter_transformacoes

class HistologiaDataset(Dataset):
    """
    Dataset PyTorch para tarefas de segmentação semântica em imagens
    histológicas.
    """
    def __init__(
        self, 
        diretorio_imagens: str, 
        diretorio_mascaras: str, 
        extensao_imagem: str, 
        aplicar_aug: bool = False
    ) -> None:
        self.diretorio_imagens = diretorio_imagens
        self.diretorio_mascaras = diretorio_mascaras
        self.extensao_imagem = extensao_imagem
        self.transformacoes = obter_transformacoes(aplicar_aug)
        self.amostras = self._mapear_arquivos()

    def _mapear_arquivos(self) -> List[Tuple[str, str]]:
        """
        Mapeia a imagem original com a máscara desta.
        """
        amostras: List[Tuple[str, str]] = []
        arquivos_presentes = os.listdir(self.diretorio_imagens)

        for arquivo in arquivos_presentes:
            if arquivo.startswith("."):
                continue
            if not arquivo.endswith(self.extensao_imagem):
                continue

            nome_base = os.path.splitext(arquivo)[0]
            caminho_imagem = os.path.join(self.diretorio_imagens, arquivo)
            caminho_mascara = os.path.join(self.diretorio_mascaras, f"{nome_base}.png")

            if os.path.isfile(caminho_imagem) and os.path.isfile(caminho_mascara):
                amostras.append((caminho_imagem, caminho_mascara))

        return amostras

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, torch.Tensor]:
        caminho_imagem, caminho_mascara = self.amostras[indice]

        imagem_pil = Image.open(caminho_imagem).convert("RGB")
        mascara_pil = Image.open(caminho_mascara).convert("L")

        imagem_np = np.array(imagem_pil)
        mascara_np = np.array(mascara_pil)

        dados_transformados = self.transformacoes(image=imagem_np, mask=mascara_np)

        imagem_tensor = dados_transformados["image"]
        mascara_tensor = dados_transformados["mask"]

        mascara_tensor = (mascara_tensor > 0).float()
        mascara_tensor = mascara_tensor.unsqueeze(0)

        return imagem_tensor, mascara_tensor
