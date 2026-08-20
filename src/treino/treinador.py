"""Rotinas de treinamento para segmentação binária."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from src.avaliacao.metricas import (
    agregar_metricas,
    calcular_metricas_por_imagem,
)


class HistoricoTreinamento(TypedDict):
    perda_treino: list[float]
    perda_validacao: list[float]
    mdice_validacao: list[float]
    melhor_epoca: int
    melhor_mdice_validacao: float


def _preparar_lote(lote: object, device: str | torch.device) -> tuple[Tensor, Tensor]:
    if not isinstance(lote, (tuple, list)) or len(lote) != 2:
        raise ValueError("cada lote deve ser uma tupla (imagens, mascaras).")

    imagens, mascaras = lote
    if not isinstance(imagens, Tensor) or not isinstance(mascaras, Tensor):
        raise TypeError("imagens e mascaras do lote devem ser tensores.")
    if imagens.ndim != 4 or mascaras.ndim != 4 or mascaras.shape[1] != 1:
        raise ValueError("imagens e mascaras devem ter formato [N, C, H, W]; máscara com C=1.")
    if imagens.shape[0] != mascaras.shape[0] or imagens.shape[-2:] != mascaras.shape[-2:]:
        raise ValueError("imagens e mascaras devem ter o mesmo lote e dimensões espaciais.")

    return (
        imagens.to(device=device, dtype=torch.float32, non_blocking=True),
        mascaras.to(device=device, dtype=torch.float32, non_blocking=True),
    )


def _calcular_perda(
    logits: Tensor,
    mascaras: Tensor,
    funcao_perda: nn.Module,
) -> Tensor:
    if logits.shape != mascaras.shape:
        raise ValueError(
            f"{tuple(logits.shape)} e {tuple(mascaras.shape)}."
        )
    perda = funcao_perda(logits, mascaras)
    if perda.ndim != 0:
        raise ValueError("a função de perda deve retornar um escalar.")
    return perda


def treinar_epoca(
    modelo: nn.Module,
    dataloader: Iterable[object],
    funcao_perda: nn.Module,
    otimizador: Optimizer,
    device: str | torch.device,
) -> float:
    modelo.train()
    soma_perdas = 0.0
    total_imagens = 0

    for lote in dataloader:
        imagens, mascaras = _preparar_lote(lote, device)
        otimizador.zero_grad()
        perda = _calcular_perda(modelo(imagens), mascaras, funcao_perda)
        perda.backward()
        otimizador.step()

        tamanho_lote = imagens.shape[0]
        soma_perdas += perda.detach().item() * tamanho_lote
        total_imagens += tamanho_lote

    if total_imagens == 0:
        raise ValueError("o dataloader de treino não pode estar vazio.")
    return soma_perdas / total_imagens


@torch.no_grad()
def _avaliar_validacao(
    modelo: nn.Module,
    dataloader: Iterable[object],
    funcao_perda: nn.Module,
    device: str | torch.device,
    limiar: float | None = None,
) -> tuple[float, float | None]:
    modelo.eval()
    soma_perdas = 0.0
    total_imagens = 0
    metricas_por_imagem = []

    for lote in dataloader:
        imagens, mascaras = _preparar_lote(lote, device)
        logits = modelo(imagens)
        perda = _calcular_perda(logits, mascaras, funcao_perda)
        tamanho_lote = imagens.shape[0]
        soma_perdas += perda.item() * tamanho_lote
        total_imagens += tamanho_lote
        if limiar is not None:
            metricas_por_imagem.extend(
                calcular_metricas_por_imagem(logits, mascaras, threshold=limiar)
            )

    if total_imagens == 0:
        raise ValueError("o dataloader de validação não pode estar vazio.")
    mdice = agregar_metricas(metricas_por_imagem)["mdice"] if limiar is not None else None
    return soma_perdas / total_imagens, mdice


@torch.no_grad()
def avaliar_epoca(
    modelo: nn.Module,
    dataloader: Iterable[object],
    funcao_perda: nn.Module,
    device: str | torch.device,
) -> float:
    perda_validacao, _ = _avaliar_validacao(modelo, dataloader, funcao_perda, device)
    return perda_validacao


def treinar(
    modelo: nn.Module,
    dataloader_treino: Iterable[object],
    dataloader_validacao: Iterable[object],
    funcao_perda: nn.Module,
    otimizador: Optimizer,
    num_epocas: int,
    device: str | torch.device,
    limiar: float = 0.5,
) -> HistoricoTreinamento:
    if num_epocas <= 0:
        raise ValueError("num_epocas deve ser maior que zero.")
    if not 0.0 <= limiar <= 1.0:
        raise ValueError("limiar deve estar entre 0 e 1.")

    modelo.to(device)
    historico: HistoricoTreinamento = {
        "perda_treino": [],
        "perda_validacao": [],
        "mdice_validacao": [],
        "melhor_epoca": 0,
        "melhor_mdice_validacao": float("-inf"),
    }
    melhor_estado: dict[str, Tensor] | None = None

    for epoca in range(num_epocas):
        historico["perda_treino"].append(
            treinar_epoca(modelo, dataloader_treino, funcao_perda, otimizador, device)
        )
        perda_validacao, mdice_validacao = _avaliar_validacao(
            modelo, dataloader_validacao, funcao_perda, device, limiar
        )
        assert mdice_validacao is not None
        historico["perda_validacao"].append(perda_validacao)
        historico["mdice_validacao"].append(mdice_validacao)

        # Empates mantêm a primeira época, garantindo seleção determinística.
        if mdice_validacao > historico["melhor_mdice_validacao"]:
            historico["melhor_mdice_validacao"] = mdice_validacao
            historico["melhor_epoca"] = epoca + 1
            # A cópia no CPU preserva o melhor estado sem reter memória da GPU.
            melhor_estado = {
                nome: tensor.detach().cpu().clone()
                for nome, tensor in modelo.state_dict().items()
            }

    if melhor_estado is None:
        raise RuntimeError("não foi possível selecionar um modelo de validação.")
    modelo.load_state_dict(melhor_estado)
    return historico