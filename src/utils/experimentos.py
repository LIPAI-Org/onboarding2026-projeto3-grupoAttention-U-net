""" Responsável por gerar um vetor de experimentos, utilizado para treinar """

from configs.experimentos import COMBINACOES

class Experimento:
    def __init__(self, modelo, dataset, f_loss, aumento, seed):
        self._modelo = modelo
        self._dataset = dataset
        self._f_loss = f_loss
        self._aumento = aumento
        self._seed = seed

    # Getters
    def get_modelo(self):
        return self._modelo

    def get_dataset(self):
        return self._dataset

    def get_f_loss(self):
        return self._f_loss

    def get_aumento(self):
        return self._aumento

    def get_seed(self):
        return self._seed

def experimentos_factory():
    lista_experimentos = []
    
    for combinacao in COMBINACOES:
        experimento = Experimento(*combinacao)
        lista_experimentos.append(experimento)
        
    return lista_experimentos
