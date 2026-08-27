import os
import sys

from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.paths import (
    PATH_MELHOR_ATUNET_HE, PATH_MELHOR_ATUNET_OEDB,
    PATH_MOSAICOS_ATTENTION_HE, PATH_MOSAICOS_ATTENTION_OEDB
)
from src.utils.checkpoints import carregar_checkpoint
from src.data.dataloader import criar_dataloaders
from configs.basicas import TAM_PATCH, NORMALIZACAO_MEAN, NORMALIZACAO_STD
from src.modelos.attention_unet import AttentionUNet 

class CapturadorAtencao:
    """
    Classe para registrar um Forward Hook no PyTorch.

    Irá capturar a saída do bloco psi (que gera os coeficientes alpha)
    durante o passo de forward (inferência).
    """
    def __init__(self):
        self.mapa_atencao = None

    def __call__(self, module, entrada, saida):
        self.mapa_atencao = saida.detach()

def desnormalizar_imagem(
        tensor_img: torch.Tensor,
        mean: tuple[float, float, float],
        std: tuple[float, float, float]
    ) -> torch.Tensor:
    """
    Aplica a transformação inversa da normalização, trazendo o tensor
    de volta para o espectro visível em RGB [0, 1].
    """
    device = tensor_img.device
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)
    
    img_desnormalizada = tensor_img * std + mean
    return torch.clamp(img_desnormalizada, 0.0, 1.0)

def gerar_mosaico_dataset(
        nome_dataset: str,
        path_checkpoint: str,
        path_saida: str
    ) -> None:
    """
    Gera um mosáico com mapas de atenção para o dataset nome_dataset

    nome_dataset deve ser HE ou OEDB

    path_checkpoint deve apontar para o modelo sendo analisado
    (localizados em results/modelos)
    """
    print(f"\n[{nome_dataset}] Iniciando processo de extração de atenção...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelo = AttentionUNet().to(device)
    modelo, _ = carregar_checkpoint(modelo, path_checkpoint)
    modelo.eval()

    hook = CapturadorAtencao()
    handle_hook = modelo.atencao1.psi.register_forward_hook(hook)

    _, _, dl_teste = criar_dataloaders(nome_dataset, num_workers=0, aplicar_aug=False)

    imagens_amostra, mascaras_amostra = [], []
    for batch_img, batch_mask in dl_teste:
        for i in range(len(batch_img)):
            if len(imagens_amostra) < 2:
                imagens_amostra.append(batch_img[i].unsqueeze(0))
                mascaras_amostra.append(batch_mask[i].unsqueeze(0))
        if len(imagens_amostra) >= 2:
            break

    tamanho_alvo = (TAM_PATCH, TAM_PATCH) if isinstance(TAM_PATCH, int) else TAM_PATCH

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22, 9))
    titulos = ['Imagem Original', 'Attention Map Puro', 'Overlay de Atenção', 'Predição da Rede', 'Ground Truth']

    for col, titulo in enumerate(titulos):
        axes[0, col].set_title(titulo, fontsize=14, fontweight='bold', pad=15)

    for idx in range(2):
        img_tensor = imagens_amostra[idx].to(device)
        mask_real = mascaras_amostra[idx].to(device)

        with torch.no_grad():
            saida = modelo(img_tensor)
            predicao = (torch.sigmoid(saida) > 0.5).float()

        mapa_atencao = hook.mapa_atencao

        mapa_atencao_interp = F.interpolate(
            mapa_atencao, size=tamanho_alvo, mode='bilinear', align_corners=False
        )

        img_desnorm = desnormalizar_imagem(img_tensor, NORMALIZACAO_MEAN, NORMALIZACAO_STD)
        img_np = img_desnorm.squeeze().cpu().permute(1, 2, 0).numpy()
        
        attn_np = mapa_atencao_interp.squeeze().cpu().numpy()
        pred_np = predicao.squeeze().cpu().numpy()
        mask_np = mask_real.squeeze().cpu().numpy()

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_ylabel(f'Amostra {idx+1}', fontsize=14, fontweight='bold', labelpad=15)

        axes[idx, 1].imshow(attn_np, cmap='jet', vmin=0.0, vmax=1.0)

        axes[idx, 2].imshow(img_np)
        axes[idx, 2].imshow(attn_np, cmap='jet', alpha=0.4, vmin=0.0, vmax=1.0)

        axes[idx, 3].imshow(pred_np, cmap='gray')

        axes[idx, 4].imshow(mask_np, cmap='gray')

        for col in range(5):
            axes[idx, col].set_xticks([])
            axes[idx, col].set_yticks([])

    plt.tight_layout()

    os.makedirs(path_saida, exist_ok=True)
    caminho_arquivo = os.path.join(path_saida, f"mosaico_attention_{nome_dataset}.png")
    
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

    handle_hook.remove()
    print(f"[{nome_dataset}] Mosaico salvo com sucesso em: {caminho_arquivo}")

def script_gerar_mosaicos_attention() -> None:
    """
    Script para geração dos mosáicos de ambos os datasets
    """
    gerar_mosaico_dataset(
            nome_dataset="HE",
            path_checkpoint=PATH_MELHOR_ATUNET_HE,
            path_saida=PATH_MOSAICOS_ATTENTION_HE
        )
    
    gerar_mosaico_dataset(
        nome_dataset="OEDB",
        path_checkpoint=PATH_MELHOR_ATUNET_OEDB,
        path_saida=PATH_MOSAICOS_ATTENTION_OEDB
    )

if __name__ == '__main__':
    script_gerar_mosaicos_attention()
