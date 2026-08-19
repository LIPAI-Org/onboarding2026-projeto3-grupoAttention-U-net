"""
Implementação da U-Net clássica
Pesos para transfer learning: Imagenet
Encoder usado: resnet34
"""

import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetClassica(nn.Module):
    def __init__(self, canais_entrada=3, classes_saida=1, usar_pesos_imagenet=True):
        super().__init__()
        
        pesos = "imagenet" if usar_pesos_imagenet else None
        
        self.modelo = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=pesos,
            in_channels=canais_entrada,
            classes=classes_saida,
        )

    def forward(self, x):
        return self.modelo(x)
