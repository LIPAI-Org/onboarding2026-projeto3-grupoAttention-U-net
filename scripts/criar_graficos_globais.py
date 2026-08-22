""" Script para criação dos gráficos globais """
import sys
from pathlib import Path

raiz_projeto = Path(__file__).resolve().parent.parent
sys.path.append(str(raiz_projeto))

from src.utils.graficos_globais import gerar_graficos_globais
from src.modelos import unet, attention_unet

print("Gerando gráficos globais...")

arquivos = gerar_graficos_globais(modelos={
    "U-Net": unet.UNetClassica(),
    "Attention U-Net": attention_unet.AttentionUNet()
})

for metrica, valor in arquivos.items():
    if isinstance(valor, dict):
        for dataset, caminho in valor.items():
            print(f"{metrica} - {dataset}: {caminho}")
    else:
        print(f"{metrica}: {valor}")

print("Gráficos globais gerados com sucesso.")
