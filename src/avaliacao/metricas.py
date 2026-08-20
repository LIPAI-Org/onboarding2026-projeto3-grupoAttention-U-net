"""Métricas para segmentação semântica binária"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence, TypedDict

import torch


EPSILON = 1e-7
Classe = Literal[0, 1]
PoliticaZeroDivisao = Literal["padrao", "zero", "one"]


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


class EstatisticasMetrica(TypedDict):
    media: float
    desvio_padrao: float


def _validar_epsilon(epsilon: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon deve ser maior que zero.")


def _validar_politica(politica_zero_divisao: PoliticaZeroDivisao) -> None:
    if politica_zero_divisao not in ("padrao", "zero", "one"):
        raise ValueError("politica_zero_divisao deve ser 'padrao', 'zero' ou 'one'.")


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
        predicao_float, mascara_float = 1.0 - predicao_float, 1.0 - mascara_float
    tp = (predicao_float * mascara_float).sum()
    fp = (predicao_float * (1.0 - mascara_float)).sum()
    fn = ((1.0 - predicao_float) * mascara_float).sum()
    return tp, fp, fn


def _valor_vazio(
    metrica: Literal["dice", "iou", "precision", "recall"],
    politica_zero_divisao: PoliticaZeroDivisao,
) -> float:
    _validar_politica(politica_zero_divisao)
    if politica_zero_divisao == "zero":
        return 0.0
    if politica_zero_divisao == "one":
        return 1.0
    return 1.0 if metrica in ("dice", "iou") else 0.0


def _razao(
    numerador: torch.Tensor,
    denominador: torch.Tensor,
    epsilon: float,
    valor_se_vazio: float,
) -> torch.Tensor:
    _validar_epsilon(epsilon)
    resultado = numerador / denominador.clamp_min(epsilon)
    return torch.where(denominador == 0, torch.full_like(resultado, valor_se_vazio), resultado)


def logits_para_mascara(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"{tuple(logits.shape)}."
        )
    if not torch.is_floating_point(logits):
        raise TypeError("logits devem ter tipo de ponto flutuante.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold deve estar entre 0 e 1.")
    if torch.isnan(logits).any().item():
        raise ValueError("logits não podem conter NaN.")
    return (torch.sigmoid(logits) >= threshold).to(dtype=torch.float32)


def dice_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe,
                    epsilon: float = EPSILON, politica_zero_divisao: PoliticaZeroDivisao = "padrao") -> torch.Tensor:
    tp, fp, fn = _contagens_por_classe(predicao, mascara, classe)
    return _razao(2 * tp, 2 * tp + fp + fn, epsilon, _valor_vazio("dice", politica_zero_divisao))


def iou_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe,
                   epsilon: float = EPSILON, politica_zero_divisao: PoliticaZeroDivisao = "padrao") -> torch.Tensor:
    tp, fp, fn = _contagens_por_classe(predicao, mascara, classe)
    return _razao(tp, tp + fp + fn, epsilon, _valor_vazio("iou", politica_zero_divisao))


def precision_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe,
                         epsilon: float = EPSILON, politica_zero_divisao: PoliticaZeroDivisao = "padrao") -> torch.Tensor:
    tp, fp, _ = _contagens_por_classe(predicao, mascara, classe)
    return _razao(tp, tp + fp, epsilon, _valor_vazio("precision", politica_zero_divisao))


def recall_por_classe(predicao: torch.Tensor, mascara: torch.Tensor, classe: Classe,
                      epsilon: float = EPSILON, politica_zero_divisao: PoliticaZeroDivisao = "padrao") -> torch.Tensor:
    tp, _, fn = _contagens_por_classe(predicao, mascara, classe)
    return _razao(tp, tp + fn, epsilon, _valor_vazio("recall", politica_zero_divisao))


def _metricas_de_contagens(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, epsilon: float,
    politica_zero_divisao: PoliticaZeroDivisao,
) -> dict[str, torch.Tensor]:
    return {
        "dice": _razao(2 * tp, 2 * tp + fp + fn, epsilon, _valor_vazio("dice", politica_zero_divisao)),
        "iou": _razao(tp, tp + fp + fn, epsilon, _valor_vazio("iou", politica_zero_divisao)),
        "precision": _razao(tp, tp + fp, epsilon, _valor_vazio("precision", politica_zero_divisao)),
        "recall": _razao(tp, tp + fn, epsilon, _valor_vazio("recall", politica_zero_divisao)),
    }


def calcular_metricas_imagem(
    predicao: torch.Tensor, mascara: torch.Tensor, epsilon: float = EPSILON,
    politica_zero_divisao: PoliticaZeroDivisao = "padrao",
) -> MetricasBinarias:
    _validar_epsilon(epsilon)
    _validar_politica(politica_zero_divisao)
    _validar_mascaras(predicao, mascara)
    if predicao.shape[0] != 1:
        raise ValueError("calcular_metricas_imagem espera exatamente uma imagem [1, 1, H, W].")
    por_classe = {
        classe: _metricas_de_contagens(
            *_contagens_por_classe(predicao, mascara, classe), epsilon, politica_zero_divisao
        )
        for classe in (0, 1)
    }
    classe_0 = {nome: valor.item() for nome, valor in por_classe[0].items()}
    classe_1 = {nome: valor.item() for nome, valor in por_classe[1].items()}
    return {
        "classe_0": classe_0,
        "classe_1": classe_1,
        "mdice": ((por_classe[0]["dice"] + por_classe[1]["dice"]) / 2).item(),
        "miou": ((por_classe[0]["iou"] + por_classe[1]["iou"]) / 2).item(),
    }


def calcular_metricas_por_imagem(
    logits: torch.Tensor, mascara: torch.Tensor, threshold: float = 0.5,
    epsilon: float = EPSILON, politica_zero_divisao: PoliticaZeroDivisao = "padrao",
) -> list[MetricasBinarias]:
    _validar_epsilon(epsilon)
    _validar_politica(politica_zero_divisao)
    predicao = logits_para_mascara(logits, threshold)
    _validar_mascaras(predicao, mascara)
    return [
        calcular_metricas_imagem(predicao[indice:indice + 1], mascara[indice:indice + 1], epsilon, politica_zero_divisao)
        for indice in range(predicao.shape[0])
    ]


def agregar_metricas(metricas_por_imagem: Sequence[MetricasBinarias]) -> MetricasBinarias:
    if not metricas_por_imagem:
        raise ValueError("é necessária pelo menos uma imagem para agregar métricas.")

    def media(*caminho: str) -> float:
        try:
            valores = [
                resultado[caminho[0]][caminho[1]] if len(caminho) == 2 else resultado[caminho[0]]
                for resultado in metricas_por_imagem
            ]
        except (KeyError, TypeError) as erro:
            raise ValueError("estrutura de métricas inválida para agregação.") from erro
        return float(torch.tensor(valores, dtype=torch.float64).mean().item())

    return {
        "classe_0": {nome: media("classe_0", nome) for nome in MetricasClasse.__annotations__},
        "classe_1": {nome: media("classe_1", nome) for nome in MetricasClasse.__annotations__},
        "mdice": media("mdice"),
        "miou": media("miou"),
    }


def calcular_metricas(
    logits: torch.Tensor, mascara: torch.Tensor, threshold: float = 0.5,
    epsilon: float = EPSILON, politica_zero_divisao: PoliticaZeroDivisao = "padrao",
) -> MetricasBinarias:
    return agregar_metricas(calcular_metricas_por_imagem(logits, mascara, threshold, epsilon, politica_zero_divisao))


def agregar_repeticoes(
    resultados: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(resultados) < 2:
        raise ValueError("são necessárias pelo menos duas repetições para desvio padrão amostral.")

    def agregar_no(caminho: tuple[str, ...]) -> object:
        valores = []
        for resultado in resultados:
            atual: object = resultado
            for chave in caminho:
                if not isinstance(atual, Mapping) or chave not in atual:
                    raise ValueError("todas as repetições devem ter a mesma estrutura de métricas.")
                atual = atual[chave]
            valores.append(atual)
        if any(not isinstance(valor, (int, float)) for valor in valores):
            raise ValueError("as folhas das métricas devem ser numéricas.")
        tensor = torch.tensor(valores, dtype=torch.float64)
        return {"media": tensor.mean().item(), "desvio_padrao": tensor.std(unbiased=True).item()}

    def visitar(modelo: Mapping[str, object], caminho: tuple[str, ...] = ()) -> dict[str, object]:
        saida: dict[str, object] = {}
        for chave, valor in modelo.items():
            proximo = caminho + (chave,)
            saida[chave] = visitar(valor, proximo) if isinstance(valor, Mapping) else agregar_no(proximo)
        return saida

    return visitar(resultados[0])


if __name__ == "__main__":
    def logits_da_mascara(predicao: torch.Tensor) -> torch.Tensor:
        return torch.where(predicao == 1, torch.full_like(predicao, 10.0), torch.full_like(predicao, -10.0))

    mascara_teste = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    perfeitos = calcular_metricas(logits_da_mascara(mascara_teste), mascara_teste)
    assert all(perfeitos[classe][metrica] == 1.0 for classe in ("classe_0", "classe_1") for metrica in ("dice", "iou", "precision", "recall"))

    oposta = 1.0 - mascara_teste
    sem_intersecao = calcular_metricas(logits_da_mascara(oposta), mascara_teste)
    assert all(sem_intersecao[classe][metrica] == 0.0 for classe in ("classe_0", "classe_1") for metrica in ("dice", "iou", "precision", "recall"))

    parcial = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    resultado_parcial = calcular_metricas(logits_da_mascara(parcial), mascara_teste)
    assert abs(resultado_parcial["classe_1"]["dice"] - 0.5) < EPSILON
    assert abs(resultado_parcial["classe_1"]["iou"] - 1.0 / 3.0) < EPSILON

    mascaras_lote = torch.cat((mascara_teste, mascara_teste))
    predicoes_lote = torch.cat((mascara_teste, oposta))
    por_imagem = calcular_metricas_por_imagem(logits_da_mascara(predicoes_lote), mascaras_lote)
    assert len(por_imagem) == 2 and por_imagem[0]["mdice"] == 1.0 and por_imagem[1]["miou"] == 0.0

    agregadas = agregar_metricas([
        {"classe_0": {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}, "classe_1": {"dice": 0.2, "iou": 0.2, "precision": 0.2, "recall": 0.2}, "mdice": 0.1, "miou": 0.1},
        {"classe_0": {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}, "classe_1": {"dice": 0.6, "iou": 0.6, "precision": 0.6, "recall": 0.6}, "mdice": 0.8, "miou": 0.8},
    ])
    assert agregadas["classe_1"]["dice"] == 0.4 and agregadas["mdice"] == 0.45

    repeticoes = agregar_repeticoes([{"mdice": 0.2}, {"mdice": 0.4}, {"mdice": 0.6}])
    assert abs(repeticoes["mdice"]["media"] - 0.4) < EPSILON
    assert abs(repeticoes["mdice"]["desvio_padrao"] - 0.2) < EPSILON

    zeros = torch.zeros_like(mascara_teste)
    ausente_ambas = calcular_metricas(logits_da_mascara(zeros), zeros)
    assert ausente_ambas["classe_1"] == {"dice": 1.0, "iou": 1.0, "precision": 0.0, "recall": 0.0}
    ausente_predicao = calcular_metricas(logits_da_mascara(zeros), mascara_teste)
    assert ausente_predicao["classe_1"]["precision"] == 0.0
    ausente_mascara = calcular_metricas(logits_da_mascara(mascara_teste), zeros)
    assert ausente_mascara["classe_1"]["recall"] == 0.0

    try:
        calcular_metricas(logits_da_mascara(mascara_teste), mascara_teste * 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Máscara não binária deveria ser rejeitada.")

    print("Testes de métricas concluídos com sucesso.")