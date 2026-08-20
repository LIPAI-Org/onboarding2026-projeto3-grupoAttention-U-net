"""Métricas para segmentação semântica binária."""

from __future__ import annotations

from typing import Dict

import torch


EPSILON = 1e-7


def _validar_epsilon(epsilon: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon deve ser maior que zero.")


def _validar_mascaras(predicao: torch.Tensor, mascara: torch.Tensor) -> None:
    if predicao.shape != mascara.shape:
        raise ValueError(
            f"Recebidos: {tuple(predicao.shape)} e {tuple(mascara.shape)}."
        )
    if predicao.ndim != 4 or predicao.shape[1] != 1:
        raise ValueError(
            f"Recebido: {tuple(predicao.shape)}."
        )
    if predicao.device != mascara.device:
        raise ValueError("predicao e mascara devem estar no mesmo device.")


def _validar_classe(classe: int) -> None:
    if classe not in (0, 1):
        raise ValueError("classe deve ser 0 (fundo) ou 1 (classe de interesse).")


def _contagens_por_classe(
    predicao: torch.Tensor, mascara: torch.Tensor, classe: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validar_mascaras(predicao, mascara)
    _validar_classe(classe)

    predicao_classe = predicao if classe == 1 else 1.0 - predicao
    mascara_classe = mascara if classe == 1 else 1.0 - mascara

    verdadeiros_positivos = (predicao_classe * mascara_classe).sum()
    falsos_positivos = (predicao_classe * (1.0 - mascara_classe)).sum()
    falsos_negativos = ((1.0 - predicao_classe) * mascara_classe).sum()
    return verdadeiros_positivos, falsos_positivos, falsos_negativos


def _divisao_segura(
    numerador: torch.Tensor, denominador: torch.Tensor, epsilon: float
) -> torch.Tensor:
    _validar_epsilon(epsilon)
    resultado = numerador / denominador.clamp_min(epsilon)
    # Se não há positivos reais nem preditos, a classe foi acertada por ausência.
    return torch.where(denominador == 0, torch.ones_like(resultado), resultado)


def logits_para_mascara(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"Recebido: {tuple(logits.shape)}."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre 0 e 1.")

    probabilidades = torch.sigmoid(logits)
    return (probabilidades >= threshold).to(dtype=torch.float32)


def dice_por_classe(
    predicao: torch.Tensor,
    mascara: torch.Tensor,
    classe: int,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    tp, fp, fn = _contagens_por_classe(predicao, mascara, classe)
    return _divisao_segura(2.0 * tp, 2.0 * tp + fp + fn, epsilon)


def iou_por_classe(
    predicao: torch.Tensor,
    mascara: torch.Tensor,
    classe: int,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    tp, fp, fn = _contagens_por_classe(predicao, mascara, classe)
    return _divisao_segura(tp, tp + fp + fn, epsilon)


def precision_por_classe(
    predicao: torch.Tensor,
    mascara: torch.Tensor,
    classe: int,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    tp, fp, _ = _contagens_por_classe(predicao, mascara, classe)
    return _divisao_segura(tp, tp + fp, epsilon)


def recall_por_classe(
    predicao: torch.Tensor,
    mascara: torch.Tensor,
    classe: int,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    tp, _, fn = _contagens_por_classe(predicao, mascara, classe)
    return _divisao_segura(tp, tp + fn, epsilon)


def calcular_metricas(
    logits: torch.Tensor,
    mascara: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = EPSILON,
) -> Dict[str, Dict[str, float] | float]:
    predicao = logits_para_mascara(logits, threshold)
    _validar_mascaras(predicao, mascara)
    _validar_epsilon(epsilon)

    metricas_por_classe: Dict[str, Dict[str, torch.Tensor]] = {}
    for classe in (0, 1):
        chave = f"classe_{classe}"
        metricas_por_classe[chave] = {
            "dice": dice_por_classe(predicao, mascara, classe, epsilon),
            "iou": iou_por_classe(predicao, mascara, classe, epsilon),
            "precision": precision_por_classe(predicao, mascara, classe, epsilon),
            "recall": recall_por_classe(predicao, mascara, classe, epsilon),
        }

    mdice = (metricas_por_classe["classe_0"]["dice"] + metricas_por_classe["classe_1"]["dice"]) / 2.0
    miou = (metricas_por_classe["classe_0"]["iou"] + metricas_por_classe["classe_1"]["iou"]) / 2.0

    return {
        "classe_0": {nome: valor.item() for nome, valor in metricas_por_classe["classe_0"].items()},
        "classe_1": {nome: valor.item() for nome, valor in metricas_por_classe["classe_1"].items()},
        "mdice": mdice.item(),
        "miou": miou.item(),
    }


if __name__ == "__main__":
    # Testes simples dos cenários descritos no projeto. Não são executados ao importar.
    mascara_teste = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

    def _logits_da_mascara(predicao_teste: torch.Tensor) -> torch.Tensor:
        return torch.where(predicao_teste == 1, torch.tensor(10.0), torch.tensor(-10.0))

    # Caso 1: predição perfeita.
    perfeitos = calcular_metricas(_logits_da_mascara(mascara_teste), mascara_teste)
    for classe in ("classe_0", "classe_1"):
        for metrica in ("dice", "iou", "precision", "recall"):
            assert perfeitos[classe][metrica] == 1.0

    # Caso 2: nenhuma interseção entre predição e máscara.
    oposta = 1.0 - mascara_teste
    sem_intersecao = calcular_metricas(_logits_da_mascara(oposta), mascara_teste)
    for classe in ("classe_0", "classe_1"):
        for metrica in ("dice", "iou", "precision", "recall"):
            assert sem_intersecao[classe][metrica] == 0.0

    # Caso 3: TP = FP = FN = 1 para a classe positiva.
    parcial = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    resultado_parcial = calcular_metricas(_logits_da_mascara(parcial), mascara_teste)
    assert abs(resultado_parcial["classe_1"]["dice"] - 0.5) < EPSILON
    assert abs(resultado_parcial["classe_1"]["iou"] - (1.0 / 3.0)) < EPSILON
    assert abs(resultado_parcial["classe_1"]["precision"] - 0.5) < EPSILON
    assert abs(resultado_parcial["classe_1"]["recall"] - 0.5) < EPSILON

    print("Testes de métricas concluídos com sucesso.")