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

PATH_METRICAS = os.path.join(
    PATH_RESULTS,
    "metrics"
)

PATH_MODELOS = os.path.join(
    PATH_RESULTS,
    "modelos"
)

PATH_QUALITATIVE = os.path.join(
    PATH_RESULTS,
    "qualitative"
)

PATH_QUALITATIVE_MOSAICOS = os.path.join(
    PATH_QUALITATIVE,
    "mosaicos"
)

PATH_QUALITATIVE_AUGMENTATIONS = os.path.join(
    PATH_QUALITATIVE,
    "augmentations"
)

PATH_PLOTS = os.path.join(
    PATH_RESULTS,
    "plots"
)

PATH_TABELA_CONSOLIDADA = os.path.join(
    PATH_METRICAS,
    "resultados_consolidados.csv"
)

PATH_TABELA_COMPLETA = os.path.join(
    PATH_METRICAS,
    "resultados_completos.csv"
)

PATH_MELHOR_ATUNET = os.path.join(
    PATH_MODELOS,
    "melhor_atunet"
)

PATH_MELHOR_UNET = os.path.join(
    PATH_MODELOS,
    "melhor_unet"
)

# Checkpoints finais, um por arquitetura e dataset.
PATH_MELHOR_ATUNET_HE = os.path.join(PATH_MODELOS, "melhor_atunet_HE")
PATH_MELHOR_ATUNET_OEDB = os.path.join(PATH_MODELOS, "melhor_atunet_OEDB")
PATH_MELHOR_UNET_HE = os.path.join(PATH_MODELOS, "melhor_unet_HE")
PATH_MELHOR_UNET_OEDB = os.path.join(PATH_MODELOS, "melhor_unet_OEDB")

PATH_PLOTS_CURVAS_APRENDIZADO = os.path.join(
    PATH_PLOTS,
    "curvas_aprendizado"
)

PATH_PLOTS_CURVAS_APRENDIZADO_LOSS = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "loss"
)

PATH_PLOTS_CURVAS_APRENDIZADO_MDICE = os.path.join(
    PATH_PLOTS_CURVAS_APRENDIZADO,
    "mdice"
)

PATH_PLOTS_GRAFICOS_GLOBAIS = os.path.join(
    PATH_PLOTS,
    "graficos_globais"
)

PATH_GRAFICOS_GLOBAIS_MDICE = os.path.join(
    PATH_PLOTS_GRAFICOS_GLOBAIS,
    "mdice"
)

PATH_GRAFICOS_GLOBAIS_MIOU = os.path.join(
    PATH_PLOTS_GRAFICOS_GLOBAIS,
    "miou"
)

PATH_GRAFICOS_GLOBAIS_GFLOPS_PARAMETROS = os.path.join(
    PATH_PLOTS_GRAFICOS_GLOBAIS,
    "gflops_e_parametros"
)

PATH_MOSAICOS_ATTENTION = os.path.join(
    PATH_QUALITATIVE,
    "mosaicos_attention"
)

PATH_MOSAICOS_ATTENTION_HE = os.path.join(
    PATH_MOSAICOS_ATTENTION,
    "HE"
)

PATH_MOSAICOS_ATTENTION_OEDB = os.path.join(
    PATH_MOSAICOS_ATTENTION,
    "OEDB"
)
