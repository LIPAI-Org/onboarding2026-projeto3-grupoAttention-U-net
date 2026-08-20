"""Métricas para segmentação semântica binária."""

from __future__ import annotations

from typing import Literal, TypedDict

import torch


EPSILON = 1e-7
Classe = Literal[0, 1]


class MetricasClasse(TypedDict):
    dice: float
    iou: float
    precision: float
    recall: float


class MetricasBinarias(TypedDict):
    classe_0: MetricasClasse
    classe_1: MetricasClasse
    mdice: float
    miou: float


def _validar_epsilon(epsilon: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon deve ser maior que zero.")


def _validar_mascaras(predicao: torch.Tensor, mascara: torch.Tensor) -> None:
    if predicao.shape != mascara.shape:
        raise ValueError(
            f"{tuple(predicao.shape)} e {tuple(mascara.shape)}."
        )
    if predicao.ndim != 4 or predicao.shape[1] != 1:
        raise ValueError(
            f"{tuple(predicao.shape)}."
        )
    if predicao.device != mascara.device:
        raise ValueError("predicao e mascara devem estar no mesmo device.")

    for nome, tensor in (("predicao", predicao), ("mascara", mascara)):
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{nome} não pode conter NaN ou infinito.")
        if not torch.logical_or(tensor == 0, tensor == 1).all().item():
            raise ValueError(f"{nome} deve conter apenas os valores 0 e 1.")


def _validar_classe(classe: int) -> None:
    if classe not in (0, 1):
        raise ValueError("classe deve ser 0 (fundo) ou 1 (classe de interesse).")


def _contagens_por_classe(
    predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validar_mascaras(predicao, mascara)
    _validar_classe(classe)

    predicao_float = predicao.to(dtype=torch.float32)
    mascara_float = mascara.to(dtype=torch.float32)
    if classe == 0:
        predicao_float = 1.0 - predicao_float
        mascara_float = 1.0 - mascara_float

    tp = (predicao_float * mascara_float).sum()
    fp = (predicao_float * (1.0 - mascara_float)).sum()
    fn = ((1.0 - predicao_float) * mascara_float).sum()
    return tp, fp, fn


def _razao(
    numerador: torch.Tensor,
    denominador: torch.Tensor,
    epsilon: float,
    valor_se_vazio: float,
) -> torch.Tensor:
    _validar_epsilon(epsilon)
    resultado = numerador / denominador.clamp_min(epsilon)
    return torch.where(
        denominador == 0,
        torch.full_like(resultado, valor_se_vazio),
        resultado,
    )


def logits_para_mascara(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "logits devem ter formato [N, 1, H, W]; recebido "
            f"{tuple(logits.shape)}."
        )
    if not torch.is_floating_point(logits):
        raise TypeError("logits devem ter tipo de ponto flutuante.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre 0 e 1.")
    if torch.isnan(logits).any().item():
        raise ValueError("logits não podem conter NaN.")

    return (torch.sigmoid(logits) >= threshold).to(dtype=torch.float32)


def dice_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe, epsilon: float = EPSILON) -> torch.Tensor:
    tp, fp, fn = _contagens_por_classe(predicao, mascara, classe)
    return _razao(2.0 * tp, 2.0 * tp + fp + fn, epsilon, valor_se_vazio=1.0)


def iou_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe, epsilon: float = EPSILON) -> torch.Tensor:
    tp, fp, fn = _contagens_por_classe(predicao, mascara, classe)
    return _razao(tp, tp + fp + fn, epsilon, valor_se_vazio=1.0)


def precision_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe, epsilon: float = EPSILON) -> torch.Tensor:
    tp, fp, _ = _contagens_por_classe(predicao, mascara, classe)
    return _razao(tp, tp + fp, epsilon, valor_se_vazio=0.0)


def recall_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe, epsilon: float = EPSILON) -> torch.Tensor:
    tp, _, fn = _contagens_por_classe(predicao, mascara, classe)
    return _razao(tp, tp + fn, epsilon, valor_se_vazio=0.0)


def _metricas_de_contagens(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, epsilon: float) -> dict[str, torch.Tensor]:
    return {
        "dice": _razao(2.0 * tp, 2.0 * tp + fp + fn, epsilon, valor_se_vazio=1.0),
        "iou": _razao(tp, tp + fp + fn, epsilon, valor_se_vazio=1.0),
        "precision": _razao(tp, tp + fp, epsilon, valor_se_vazio=0.0),
        "recall": _razao(tp, tp + fn, epsilon, valor_se_vazio=0.0),
    }


def calcular_metricas(
    logits: torch.Tensor,
    mascara: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = EPSILON,
) -> MetricasBinarias:
    _validar_epsilon(epsilon)
    predicao = logits_para_mascara(logits, threshold)
    _validar_mascaras(predicao, mascara)

    por_classe = {
        classe: _metricas_de_contagens(*_contagens_por_classe(predicao, mascara, classe), epsilon)
        for classe in (0, 1)
    }
    classe_0 = {nome: valor.item() for nome, valor in por_classe[0].items()}
    classe_1 = {nome: valor.item() for nome, valor in por_classe[1].items()}
    return {
        "classe_0": classe_0,
        "classe_1": classe_1,
        "mdice": (por_classe[0]["dice"] + por_classe[1]["dice"]).div(2.0).item(),
        "miou": (por_classe[0]["iou"] + por_classe[1]["iou"]).div(2.0).item(),
    }


if __name__ == "__main__":
    def logits_da_mascara(predicao: torch.Tensor) -> torch.Tensor:
        return torch.where(
            predicao == 1,
            torch.full_like(predicao, 10.0),
            torch.full_like(predicao, -10.0),
        )

    mascara_teste = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    perfeitos = calcular_metricas(logits_da_mascara(mascara_teste), mascara_teste)
    assert all(perfeitos[classe][metrica] == 1.0 for classe in ("classe_0", "classe_1") for metrica in ("dice", "iou", "precision", "recall"))

    parcial = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    resultado = calcular_metricas(logits_da_mascara(parcial), mascara_teste)
    assert abs(resultado["classe_1"]["dice"] - 0.5) < EPSILON
    assert abs(resultado["classe_1"]["iou"] - 1.0 / 3.0) < EPSILON

    zeros = torch.zeros_like(mascara_teste)
    vazio = calcular_metricas(logits_da_mascara(zeros), zeros)
    assert vazio["classe_1"] == {"dice": 1.0, "iou": 1.0, "precision": 0.0, "recall": 0.0}

    try:
        calcular_metricas(logits_da_mascara(mascara_teste), mascara_teste * 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Máscara não binária deveria ser rejeitada.")

    print("Testes de métricas concluídos com sucesso.")
