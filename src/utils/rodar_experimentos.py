""" Funções para lidar com o treinamento/avaliação de experimentos """

from configs.basicas import DEVICE
from src.utils.experimentos import Experimento
from src.modelos.modelo_factory import pegar_modelo
from src.losses.loss_factory import pegar_f_loss
from src.data.dataloader import criar_dataloaders
from src.treino.treinamento import treinar_modelo
from src.avaliacao.avaliador import avaliar_modelo
from src.utils.tabela_resultado import adicionar_resultado_completo

def rodar_um_experimento(experimento: Experimento):
    """Executa um experimento completo de treinamento e avaliação."""
    nome_modelo = experimento.get_modelo().upper()
    dataset = experimento.get_dataset()
    aumento = experimento.get_aumento()
    nome_f_loss = experimento.get_f_loss().upper()
    seed = experimento.get_seed()
    nome = f'{nome_modelo}_{dataset}_{nome_f_loss}_{aumento}_{seed}'

    modelo = pegar_modelo(nome_modelo)
    modelo.to(DEVICE)
    f_loss = pegar_f_loss(nome_f_loss)
    dl_treino, dl_val, dl_teste = criar_dataloaders(
        nome_dataset=dataset,
        num_workers=0,
        aplicar_aug=aumento
    )
    modelo, melhor_mdice_val, melhor_epoca, _ = treinar_modelo(
        nome=nome,
        modelo=modelo,
        dataset=dataset,
        dataloader_treino=dl_treino,
        dataloader_val=dl_val,
        f_loss=f_loss,
        seed=seed
    )

    modelo.to(DEVICE)

    metricas_teste = avaliar_modelo(
        modelo=modelo,
        dataloader_teste=dl_teste
    )

    modo_treinamento = 'FS' if nome_modelo in ("UNETFS", "ATUNET") else 'PTALL'

    adicionar_resultado_completo(
        modelo=modelo,
        dataset=dataset,
        modo_treinamento=modo_treinamento,
        loss=nome_f_loss,
        augmentation=str(aumento),
        seed=seed,
        dice_background_test=metricas_teste["classe_0"]["dice"],
        dice_foreground_test=metricas_teste["classe_1"]["dice"],
        mDice_test=metricas_teste["mdice"],
        iou_background_test=metricas_teste["classe_0"]["iou"],
        iou_foreground_test=metricas_teste["classe_1"]["iou"],
        mIoU_test=metricas_teste["miou"],
        precision_foreground_test=metricas_teste["classe_1"]["precision"],
        recall_foreground_test=metricas_teste["classe_1"]["recall"],
        best_epoch=melhor_epoca,
        val_mDice_best=melhor_mdice_val
    )

def rodar_todos_experimentos(experimentos):
    """Executa todos os experimentos fornecidos."""
    for experimento in experimentos:
        rodar_um_experimento(experimento)
