""" Paths usados pelo código """

import os

# root
PATH_RAIZ = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# dados
PATH_DATA = os.path.join(
    PATH_RAIZ,
    "data"
)

PATH_RAW = os.path.join(
    PATH_DATA,
    "raw"
)

PATH_SPLITS = os.path.join(
    PATH_DATA,
    "splits"
)

PATH_MASCARAS = os.path.join(
    PATH_DATA,
    "mascaras"
)

PATH_RAW_HE = os.path.join(
    PATH_RAW,
    "HE"
)

PATH_RAW_OEDB = os.path.join(
    PATH_RAW,
    "OEDB"
)

PATH_SPLITS_HE = os.path.join(
    PATH_SPLITS,
    "HE"
)

PATH_SPLITS_OEDB = os.path.join(
    PATH_SPLITS,
    "OEDB"
)

PATH_SPLIT_TREINO_HE = os.path.join(
    PATH_SPLITS_HE,
    "treino"
)

PATH_SPLIT_TESTE_HE = os.path.join(
    PATH_SPLITS_HE,
    "teste"
)

PATH_SPLIT_VAL_HE = os.path.join(
    PATH_SPLITS_HE,
    "val"
)

PATH_SPLIT_TREINO_OEDB = os.path.join(
    PATH_SPLITS_OEDB,
    "treino"
)

PATH_SPLIT_TESTE_OEDB = os.path.join(
    PATH_SPLITS_OEDB,
    "teste"
)

PATH_SPLIT_VAL_OEDB = os.path.join(
    PATH_SPLITS_OEDB,
    "val"
)

PATH_MASC_HE = os.path.join(
    PATH_MASCARAS,
    "HE"
)

PATH_MASC_OEDB = os.path.join(
    PATH_MASCARAS,
    "OEDB"
)

# resultados
PATH_RESULTS = os.path.join(
    PATH_RAIZ,
    "results"
)

PATH_MODELOS = os.path.join(
    PATH_RESULTS,
    "modelos"
)

PATH_MELHOR_ATUNET = os.path.join(
    PATH_MODELOS,
    "melhor_atunet.pt"
)

PATH_MELHOR_UNET = os.path.join(
    PATH_MODELOS,
    "melhor_unet.pt"
)
