"""
Script para treinamento completo de todos os modelos
(registrando os resultados na tabela completa e os consolidando)
"""

import sys
from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.experimentos import experimentos_factory
from src.utils.rodar_experimentos import rodar_todos_experimentos
from src.utils.tabela_resultado import consolidar_resultados

def script_rodar_todos_e_consolidar():
    experimentos = experimentos_factory()
    rodar_todos_experimentos(experimentos)
    consolidar_resultados()

if __name__ == '__main__':
    script_rodar_todos_e_consolidar()
