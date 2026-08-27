""" Factory das funções de loss """

from torch.nn import BCEWithLogitsLoss, Module

from src.losses.dice_loss import DiceLoss

def pegar_f_loss(f_loss: str) -> Module:
    """
    Duas funções de perda:

    BCE: Binary Cross Entropy com Logits
    
    DICE: Dice loss

    Levanta ValueError caso não seja informada uma f_loss válida.
    """
    if f_loss.upper() == "BCE":
        return BCEWithLogitsLoss()
    elif f_loss.upper() == "DICE":
        return DiceLoss()
    else:
        raise ValueError("Informe uma função de perda válida")
