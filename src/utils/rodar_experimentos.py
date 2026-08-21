""" Funções para lidar com o treinamento/avaliação de experimentos """

from configs.basicas import DEVICE
from src.utils.experimentos import Experimento
from src.modelos.modelo_factory import pegar_modelo
from src.data.dataloader import criar_dataloaders
from src.treino.treinamento import treinar_modelo
from src.avaliacao.avaliador import avaliar_modelo

def rodar_um_experimento(experimento: Experimento):
    nome_modelo = experimento.get_modelo()
    dataset = experimento.get_dataset()
    aumento = experimento.get_aumento()
    f_loss = experimento.get_f_loss()
    seed = experimento.get_seed()
    modo_treinamento = 'FS' if nome_modelo.upper() in ("UNETFS", "ATUNET") else 'PTALL'
    nome = f'{nome_modelo}_{dataset}_{f_loss}_{aumento}_{seed}'

    modelo = pegar_modelo(nome_modelo)
    modelo.to(DEVICE)
    dl_treino, dl_val, dl_teste = criar_dataloaders(
        nome_dataset=dataset,
        num_workers=0,
        aplicar_aug=aumento
    )
    modelo, melhor_mdice_val = treinar_modelo(
        nome=nome,
        modelo=modelo,
        dataloader_treino=dl_treino,
        dataloader_val=dl_val,
        f_loss=f_loss,
        seed=seed
    )

    metricas_teste = avaliar_modelo(
        modelo=modelo,
        dataloader_teste=dl_teste
    )
