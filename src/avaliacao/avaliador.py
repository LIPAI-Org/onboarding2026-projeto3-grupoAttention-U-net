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
    """Valida os argumentos da avaliação e retorna o device normalizado."""
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
    if primeiro_parametro is not None and primeiro_parametro.device != device_torch:
        raise ValueError("device deve ser o mesmo device dos parâmetros do modelo.")
    return device_torch


def _validar_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Garante que o batch segue o contrato imagem/máscara do DataLoader."""
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
    """Avalia um modelo já selecionado no teste e retorna métricas médias por imagem.

    Não seleciona épocas, modelos ou hiperparâmetros. As fórmulas e a
    agregação das métricas são inteiramente delegadas a ``metricas.py``.
    """
    device_torch = _validar_entrada(modelo, dataloader_teste, DEVICE, LIMIAR)
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

# Script de teste
# def executar_testes_basicos() -> None:
#     """Verifica o fluxo com modelos e tensores artificiais, sem dados reais."""
#     from torch.utils.data import TensorDataset

#     class ModeloPerfeito(nn.Module):
#         def __init__(self) -> None:
#             super().__init__()
#             self.gradientes_ativos_no_forward: bool | None = None

#         def forward(self, imagens: torch.Tensor) -> torch.Tensor:
#             self.gradientes_ativos_no_forward = torch.is_grad_enabled()
#             return imagens[:, :1] * 20.0 - 10.0

#     mascaras = torch.tensor(
#         [
#             [[[0.0, 1.0], [1.0, 0.0]]],
#             [[[1.0, 0.0], [0.0, 1.0]]],
#         ]
#     )
#     imagens = mascaras.repeat(1, 3, 1, 1)
#     dataloader = DataLoader(TensorDataset(imagens, mascaras), batch_size=2)
#     modelo = ModeloPerfeito()
#     resultado = avaliar_modelo(modelo, dataloader, device="cpu", threshold=0.5)

#     assert resultado["classe_0"]["dice"] == 1.0
#     assert resultado["classe_1"]["iou"] == 1.0
#     assert resultado["classe_0"]["precision"] == 1.0
#     assert resultado["classe_1"]["recall"] == 1.0
#     assert resultado["mdice"] == 1.0 and resultado["miou"] == 1.0
#     assert modelo.gradientes_ativos_no_forward is False

#     resultado_limiar_alto = avaliar_modelo(modelo, dataloader, device="cpu", threshold=1.0)
#     assert resultado_limiar_alto["classe_1"]["dice"] == 0.0

#     dataloader_vazio = DataLoader(TensorDataset(imagens[:0], mascaras[:0]), batch_size=1)
#     try:
#         avaliar_modelo(modelo, dataloader_vazio, device="cpu")
#     except ValueError as erro:
#         assert "vazio" in str(erro)
#     else:
#         raise AssertionError("Um DataLoader vazio deveria gerar ValueError.")
