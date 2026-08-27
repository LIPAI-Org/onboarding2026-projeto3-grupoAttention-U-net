""" Script bem básico para consolidação de resultados """

import sys

from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.tabela_resultado import consolidar_resultados

def script_consolidar_resultados() -> None:
    """
    Script para consolidar os resultados das execuções.

    Os resultados consolidados ficarão em results/metrics/resultados_consolidados.csv
    por padrão, caso deseje mudar o destino deve-se alterar a variável PATH_TABELA_CONSOLIDADA
    em src/utils/paths.py para uma str que aponte para a tabela destino.
    """
    try:
        consolidar_resultados()
    except (TypeError, ValueError) as e:
        print(f'Ocorreu um erro na consolidação: {e}')
        print('Logo, não foi possível consolidar os resultados.')

if __name__ == '__main__':
    script_consolidar_resultados()
