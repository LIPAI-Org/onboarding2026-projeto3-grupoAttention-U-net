""" Definição de algumas configurações básicas """

import torch.cuda as cuda

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from src.losses.dice_loss import DiceLoss

TAM_PATCH = 256
NORMALIZACAO_MEAN = (0.485, 0.456, 0.406)
NORMALIZACAO_STD = (0.229, 0.224, 0.225)
NUM_EPOCAS = 50
LIMIAR = 0.5
OTIMIZADOR = Adam
DEVICE = "cuda" if cuda.is_available() else "cpu"
TAXA_APRENDIZADO = 1e-4
TAM_BATCH = 8
F_LOSS_1 = BCEWithLogitsLoss()
F_LOSS_2 = DiceLoss()
