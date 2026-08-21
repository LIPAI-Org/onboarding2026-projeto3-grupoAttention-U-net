""" Configs dos experimentos """

from itertools import product

MODELOS = "UNETFS", "UNETPTALL", "ATUNET"
DATASETS = "OEDB", "HE"
F_LOSSES = "BCE", "DICE"
AUMENTO = True, False
SEEDS = 42, 123, 2025

COMBINACOES = list(product(MODELOS, DATASETS, F_LOSSES, AUMENTO, SEEDS))
