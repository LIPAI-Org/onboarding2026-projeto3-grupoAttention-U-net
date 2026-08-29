""" Lógica da etapa de validação """

import torch

from configs.basicas import DEVICE
from src.avaliacao.metricas import calcular_metricas_por_imagem, agregar_metricas

def validar(modelo, dataloader_val, f_loss):
    """Valida o modelo e calcula as métricas de desempenho."""
    modelo.eval()
    todas_metricas_imagens = []
    
    val_loss = 0.0

    with torch.no_grad():
        for inputs, mascaras in dataloader_val:
            inputs = inputs.to(DEVICE)
            mascaras = mascaras.to(DEVICE)

            logits = modelo(inputs)
            loss = f_loss(logits, mascaras)
            val_loss += loss.item() * inputs.size(0)

            metricas_batch = calcular_metricas_por_imagem(logits, mascaras, threshold=0.5)
            todas_metricas_imagens.extend(metricas_batch)

    metricas_globais = agregar_metricas(todas_metricas_imagens)
    
    loss_media = val_loss / len(dataloader_val.dataset)
    mdice_final = metricas_globais["mdice"]
    
    return loss_media, mdice_final, metricas_globais
