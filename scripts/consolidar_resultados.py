""" Script bem básico para consolidação de resultados """

import sys

from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.tabela_resultado import consolidar_resultados

consolidar_resultados()
