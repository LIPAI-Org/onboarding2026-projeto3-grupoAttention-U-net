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

def script_dividir_dataset(
        origem: str,
        extensao: str,
        destino_treino: str,
        destino_teste: str,
        destino_val: str,
        semente: int =42
    ) -> None:
    """
    Script para criação dos splits dos datasets.

    Deve ser executado apenas uma vez, após arrumar os
    datasets em data/raw e antes de qualquer treinamento/teste.
    """
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

if __name__ == '__main__':
    script_dividir_dataset(PATH_RAW_HE, ".png", PATH_SPLIT_TREINO_HE, PATH_SPLIT_TESTE_HE, PATH_SPLIT_VAL_HE)
    script_dividir_dataset(PATH_RAW_OEDB, ".tif", PATH_SPLIT_TREINO_OEDB, PATH_SPLIT_TESTE_OEDB, PATH_SPLIT_VAL_OEDB)
