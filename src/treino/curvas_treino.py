""" Lida com as curvas de aprendizado """

import os
import matplotlib.pyplot as plt
from typing import List

from src.utils.paths import (
    PATH_PLOTS_CURVAS_APRENDIZADO_LOSS,
    PATH_PLOTS_CURVAS_APRENDIZADO_MDICE
)

def plotar_losses(nome: str, historico_loss_treino: List[float], historico_loss_val: List[float]):

    plt.figure(figsize=(10, 6))
    
    epocas = range(1, len(historico_loss_treino) + 1)
    
    plt.plot(epocas, historico_loss_treino, label='Loss Treino', color='blue', marker='o')
    plt.plot(epocas, historico_loss_val, label='Loss Validação', color='red', marker='x')
    
    plt.title(f'Curva de Aprendizado (Loss) - {nome}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    caminho_salvamento = os.path.join(PATH_PLOTS_CURVAS_APRENDIZADO_LOSS, f"loss_{nome}.png")
    plt.savefig(caminho_salvamento, bbox_inches='tight')
    plt.close()

def plotar_mdice_val(nome: str, historico_mdice_val: List[float]):

    plt.figure(figsize=(10, 6))
    
    epocas = range(1, len(historico_mdice_val) + 1)
    
    plt.plot(epocas, historico_mdice_val, label='mDice Validação', color='green', marker='s')
    
    plt.title(f'Curva de Aprendizado (mDice) - {nome}')
    plt.xlabel('Época')
    plt.ylabel('mDice')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    caminho_salvamento = os.path.join(PATH_PLOTS_CURVAS_APRENDIZADO_MDICE, f"mdice_{nome}.png") 
    plt.savefig(caminho_salvamento, bbox_inches='tight')
    plt.close()
