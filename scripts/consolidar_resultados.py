""" Script bem básico para consolidação de resultados """

import sys

from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.tabela_resultado import consolidar_resultados

def script_consolidar_resultados():
    try:
        consolidar_resultados()
    except (TypeError, ValueError) as e:
        print(f'Ocorreu um erro na consolidação: {e}')
        print('Logo, não foi possível consolidar os resultados.')

if __name__ == '__main__':
    script_consolidar_resultados()
