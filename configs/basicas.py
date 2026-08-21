""" Definição de algumas configurações básicas """

import torch.cuda as cuda

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

TAM_PATCH = 256
NORMALIZACAO_MEAN = (0.485, 0.456, 0.406)
NORMALIZACAO_STD = (0.229, 0.224, 0.225)
NUM_EPOCAS = 50
LIMIAR = 0.5
OTIMIZADOR = Adam
DEVICE = "cuda" if cuda.is_available() else "cpu"
TAXA_APRENDIZADO = 1e-3
TAM_BATCH = 8
REDUTOR_LR = StepLR
TAM_STEP = 10
GAMMA = 0.1
