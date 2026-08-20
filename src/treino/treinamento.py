""" Lida com o treinamento """

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from configs.basicas import (
    DEVICE,
    TAM_STEP,
    NUM_EPOCAS,
    OTIMIZADOR,
    REDUTOR_LR,
    GAMMA,
    TAXA_APRENDIZADO,
)
from src.utils.checkpoints import salvar_checkpoint
from src.treino.validacao import validar
from src.utils.seed import fixar_seed
import curvas_treino

def treinar_uma_epoca(
    nome: str,
    modelo: nn.Module,
    dataloader_treino: DataLoader,
    dataloader_val: DataLoader,
    f_loss: Callable,
    melhor_mdice_atual: float,
    otimizador: torch.optim.Optimizer,
) -> Tuple[nn.Module, float, float, float]:

    modelo.train()
    soma_loss_treino = 0.0

    for imagens, mascaras in dataloader_treino:
        imagens = imagens.to(DEVICE)
        mascaras = mascaras.to(DEVICE)

        otimizador.zero_grad()
        saidas = modelo(imagens)
        loss = f_loss(saidas, mascaras)
        loss.backward()
        otimizador.step()

        soma_loss_treino += loss.item()

    loss_treino = soma_loss_treino / len(dataloader_treino)

    loss_val, mdice_val, _ = validar(modelo, dataloader_val, f_loss)
    if mdice_val > melhor_mdice_atual:
        salvar_checkpoint(modelo, mdice_val, nome)

    return modelo, loss_treino, loss_val, mdice_val


def treinar_modelo(
    nome: str,
    modelo: nn.Module,
    dataloader_treino: DataLoader,
    dataloader_val: DataLoader,
    f_loss: Callable,
    seed: int,
) -> Tuple[nn.Module, float, Dict[str, List[float]]]:
    
    fixar_seed(seed)
    modelo = modelo.to(DEVICE)

    otimizador = OTIMIZADOR(modelo.parameters(), lr=TAXA_APRENDIZADO)
    scheduler = REDUTOR_LR(otimizador, step_size=TAM_STEP, gamma=GAMMA)

    melhor_mdice_atual = 0.0
    historico_loss_treino: List[float] = []
    historico_loss_val: List[float] = []
    historico_mdice_val: List[float] = []

    print(f"\n{'='*15} Iniciando Treinamento: {nome} {'='*15}")
    print(f"Dispositivo: {DEVICE} | Total de Épocas: {NUM_EPOCAS}\n")

    for epoca in range(1, NUM_EPOCAS + 1):
        modelo, loss_treino, loss_val, mdice_val = treinar_uma_epoca(
            nome=nome,
            modelo=modelo,
            dataloader_treino=dataloader_treino,
            dataloader_val=dataloader_val,
            f_loss=f_loss,
            melhor_mdice_atual=melhor_mdice_atual,
            otimizador=otimizador,
        )

        if mdice_val > melhor_mdice_atual:
            melhor_mdice_atual = mdice_val

        scheduler.step()

        historico_loss_treino.append(loss_treino)
        historico_loss_val.append(loss_val)
        historico_mdice_val.append(mdice_val)

        lr_atual = otimizador.param_groups[0]["lr"]
        print(
            f"Época [{epoca:02d}/{NUM_EPOCAS:02d}] - "
            f"Loss Treino: {loss_treino:.4f} | "
            f"Loss Val: {loss_val:.4f} | "
            f"mDice Val: {mdice_val:.4f} | "
            f"LR: {lr_atual:.6f}"
        )

    historico = {
        "loss_treino": historico_loss_treino,
        "loss_val": historico_loss_val,
        "mdice_val": historico_mdice_val,
    }

    print(f"\nTreinamento de '{nome}' concluído com sucesso!")
    print(f"Melhor mDice atingido: {melhor_mdice_atual:.4f}\n")

    curvas_treino.plotar_losses(nome, historico_loss_treino, historico_loss_val)
    curvas_treino.plotar_mdice_val(nome, historico_mdice_val)

    return modelo, melhor_mdice_atual, historico