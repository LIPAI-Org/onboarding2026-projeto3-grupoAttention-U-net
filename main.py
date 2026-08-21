""" Temporariamente para testes de execução, posteriormente conterá a execução completa """

from src.utils.experimentos import experimentos_factory
from src.utils.rodar_experimentos import rodar_todos_experimentos

experimentos = experimentos_factory()
rodar_todos_experimentos(experimentos)
