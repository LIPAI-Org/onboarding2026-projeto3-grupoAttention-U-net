"""
Factory para criar os dataloaders de treino, validação e teste
"""

from typing import Tuple

from torch.utils.data import DataLoader

from src.utils.paths import (
    PATH_SPLIT_TREINO_HE, PATH_SPLIT_VAL_HE, PATH_SPLIT_TESTE_HE,
    PATH_SPLIT_TREINO_OEDB, PATH_SPLIT_VAL_OEDB, PATH_SPLIT_TESTE_OEDB,
    PATH_MASC_HE, PATH_MASC_OEDB
)
from src.data.dataset import HistologiaDataset
from configs.basicas import TAM_BATCH

def criar_dataloaders(
    nome_dataset: str, 
    num_workers: int, 
    aplicar_aug: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dominio = nome_dataset.upper()
    
    if dominio == "HE":
        caminho_treino = PATH_SPLIT_TREINO_HE
        caminho_val = PATH_SPLIT_VAL_HE
        caminho_teste = PATH_SPLIT_TESTE_HE
        caminho_mascara = PATH_MASC_HE
        extensao = ".png"
    elif dominio == "OEDB":
        caminho_treino = PATH_SPLIT_TREINO_OEDB
        caminho_val = PATH_SPLIT_VAL_OEDB
        caminho_teste = PATH_SPLIT_TESTE_OEDB
        caminho_mascara = PATH_MASC_OEDB
        extensao = ".tif"
    else:
        raise ValueError("Identificador de dataset invalido. Escolha 'HE' ou 'OEDB'.")

    dataset_treino = HistologiaDataset(
        diretorio_imagens=caminho_treino,
        diretorio_mascaras=caminho_mascara,
        extensao_imagem=extensao,
        aplicar_aug=aplicar_aug
    )

    dataset_val = HistologiaDataset(
        diretorio_imagens=caminho_val,
        diretorio_mascaras=caminho_mascara,
        extensao_imagem=extensao,
        aplicar_aug=False
    )

    dataset_teste = HistologiaDataset(
        diretorio_imagens=caminho_teste,
        diretorio_mascaras=caminho_mascara,
        extensao_imagem=extensao,
        aplicar_aug=False
    )

    dl_treino = DataLoader(
        dataset_treino, 
        batch_size=TAM_BATCH, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    
    dl_val = DataLoader(
        dataset_val, 
        batch_size=TAM_BATCH, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=False
    )
    
    dl_teste = DataLoader(
        dataset_teste, 
        batch_size=TAM_BATCH, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=False
    )

    return dl_treino, dl_val, dl_teste
