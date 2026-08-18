"""

Script para criação dos splits

Distribuição:
Treino: 56%
Validação: 14%
Teste: 30%

"""

import os
import sys
import shutil
import random

from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.paths import (
    PATH_RAW_HE,
    PATH_RAW_OEDB,
    PATH_SPLIT_TESTE_HE,
    PATH_SPLIT_TREINO_HE,
    PATH_SPLIT_VAL_HE,
    PATH_SPLIT_TESTE_OEDB,
    PATH_SPLIT_TREINO_OEDB,
    PATH_SPLIT_VAL_OEDB
)

def dividir_dataset(origem, extensao, destino_treino, destino_teste, destino_val, semente=42):
    random.seed(semente)
    
    arquivos = [f for f in Path(origem).iterdir() if f.is_file() and f.suffix.lower() == extensao.lower()]
    arquivos.sort()
    random.shuffle(arquivos)
    
    total = len(arquivos)
    qtd_treino = int(total * 0.56)
    qtd_teste = int(total * 0.30)
    
    divisoes = {
        destino_treino: arquivos[:qtd_treino],
        destino_teste: arquivos[qtd_treino:qtd_treino + qtd_teste],
        destino_val: arquivos[qtd_treino + qtd_teste:]
    }
    
    for pasta_destino, lista_arquivos in divisoes.items():
        os.makedirs(pasta_destino, exist_ok=True)
        for arquivo in lista_arquivos:
            shutil.copy(arquivo, Path(pasta_destino) / arquivo.name)

dividir_dataset(PATH_RAW_HE, ".png", PATH_SPLIT_TREINO_HE, PATH_SPLIT_TESTE_HE, PATH_SPLIT_VAL_HE)
dividir_dataset(PATH_RAW_OEDB, ".tif", PATH_SPLIT_TREINO_OEDB, PATH_SPLIT_TESTE_OEDB, PATH_SPLIT_VAL_OEDB)
