""" Factory das funções de loss """

from torch.nn import BCEWithLogitsLoss

from src.losses.dice_loss import DiceLoss

def pegar_f_loss(f_loss: str):
    """
    Duas funções de perda:
    BCE: Binary Cross Entropy com Logits
    DICE: Dice loss
    """
    if f_loss.upper() == "BCE":
        return BCEWithLogitsLoss()
    elif f_loss.upper() == "DICE":
        return DiceLoss()
    else:
        raise ValueError("Informe uma função de perda válida")
