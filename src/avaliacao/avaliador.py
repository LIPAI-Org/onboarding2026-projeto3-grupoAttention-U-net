"""Avaliação final de modelos de segmentação binária no conjunto de teste."""

from __future__ import annotations

import numbers
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.avaliacao.metricas import MetricasBinarias, agregar_metricas, calcular_metricas_por_imagem
from configs.basicas import DEVICE, LIMIAR


def _validar_entrada(
    modelo: nn.Module,
    dataloader_teste: DataLoader,
) -> torch.device:
    """Valida o modelo, o DataLoader e as configurações de avaliação."""
    if not isinstance(modelo, nn.Module):
        raise TypeError("modelo deve ser uma instância de torch.nn.Module.")
    if isinstance(LIMIAR, bool) or not isinstance(LIMIAR, numbers.Real):
        raise TypeError("threshold deve ser numérico.")
    if not 0.0 <= float(LIMIAR) <= 1.0:
        raise ValueError("threshold deve estar entre 0 e 1.")
    if not hasattr(dataloader_teste, "__len__") or len(dataloader_teste) == 0:
        raise ValueError("dataloader_teste não pode estar vazio.")

    device_torch = torch.device(DEVICE)
    primeiro_parametro = next(modelo.parameters(), None)
    if primeiro_parametro is not None and primeiro_parametro.device.type != device_torch.type:
        raise ValueError("device deve ser o mesmo device dos parâmetros do modelo.")
    return device_torch


def _validar_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Valida e retorna as imagens e máscaras de um batch."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Cada batch deve conter exatamente (imagens, mascaras).")

    imagens, mascaras = batch
    if not isinstance(imagens, torch.Tensor) or not isinstance(mascaras, torch.Tensor):
        raise TypeError("imagens e mascaras devem ser tensores PyTorch.")
    if imagens.ndim != 4:
        raise ValueError("imagens devem ter formato [N, C, H, W].")
    if mascaras.ndim != 4 or mascaras.shape[1] != 1:
        raise ValueError("mascaras devem ter formato [N, 1, H, W].")
    if imagens.shape[0] == 0 or imagens.shape[0] != mascaras.shape[0]:
        raise ValueError("imagens e mascaras devem possuir o mesmo batch não vazio.")
    if imagens.shape[-2:] != mascaras.shape[-2:]:
        raise ValueError("imagens e mascaras devem possuir a mesma resolução espacial.")
    return imagens, mascaras


@torch.no_grad()
def avaliar_modelo(
    modelo: nn.Module,
    dataloader_teste: DataLoader,
) -> MetricasBinarias:
    """Avalia o modelo no conjunto de teste e retorna as métricas agregadas."""
    device_torch = _validar_entrada(modelo, dataloader_teste)
    modelo.eval()
    metricas_por_imagem: list[MetricasBinarias] = []

    for batch in dataloader_teste:
        imagens, mascaras = _validar_batch(batch)
        imagens = imagens.to(device_torch)
        mascaras = mascaras.to(device_torch)

        logits = modelo(imagens)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("O modelo deve retornar um tensor de logits.")
        if logits.shape != mascaras.shape:
            raise ValueError(
                "logits e mascaras devem ter o mesmo formato; "
                f"recebidos {tuple(logits.shape)} e {tuple(mascaras.shape)}."
            )

        metricas_por_imagem.extend(
            calcular_metricas_por_imagem(logits, mascaras, threshold=float(LIMIAR))
        )

    if not metricas_por_imagem:
        raise ValueError("Nenhuma imagem foi recebida para avaliação.")
    return agregar_metricas(metricas_por_imagem)
