"""
Implementação da U-Net clássica
Pesos para transfer learning: Imagenet
Encoder usado: resnet34
"""

import torch.nn as nn
import segmentation_models_pytorch as smp

from torch import Tensor

class UNetClassica(nn.Module):
    """
    U-Net classica, vinda de segmentation_models_pytorch
    """
    def __init__(
            self,
            canais_entrada: int = 3,
            classes_saida: int = 1,
            usar_pesos_imagenet: bool =True
        ) -> None:
        super().__init__()
        
        pesos = "imagenet" if usar_pesos_imagenet else None
        
        self.modelo = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=pesos,
            in_channels=canais_entrada,
            classes=classes_saida,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.modelo(x)
