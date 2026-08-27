"""
Implementação da Attention U-Net, que foi a arquitetura escolhida pelo grupo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class BlocoConvolucional(nn.Module):
    """
    O bloco convolucional da Attention U-Net usada.
    """
    def __init__(self, canais_entrada: int, canais_saida: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class PortaoAtencao(nn.Module):
    """
    O Attention Gate da Attention U-Net usada.
    """
    def __init__(
            self,
            canais_x: int,
            canais_g: int,
            canais_inter: int
        ) -> None:
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(canais_g, canais_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(canais_inter)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(canais_x, canais_inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(canais_inter)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(canais_inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)
        
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode='bilinear', align_corners=False)
            
        soma = self.relu(g_proj + x_proj)
        coeficientes_atencao = self.psi(soma)
        
        if coeficientes_atencao.shape[2:] != x.shape[2:]:
            coeficientes_atencao = F.interpolate(coeficientes_atencao, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return x * coeficientes_atencao

class AttentionUNet(nn.Module):
    """
    A Attention U-Net usada.
    """
    def __init__(
            self,
            canais_entrada: int = 3,
            classes_saida: int = 1,
            filtros: List[int] = [64, 128, 256, 512, 1024]
        ) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder1 = BlocoConvolucional(canais_entrada, filtros[0])
        self.encoder2 = BlocoConvolucional(filtros[0], filtros[1])
        self.encoder3 = BlocoConvolucional(filtros[1], filtros[2])
        self.encoder4 = BlocoConvolucional(filtros[2], filtros[3])
        
        self.gargalo = BlocoConvolucional(filtros[3], filtros[4])
        
        self.upconv4 = nn.ConvTranspose2d(filtros[4], filtros[3], kernel_size=2, stride=2)
        self.atencao4 = PortaoAtencao(canais_x=filtros[3], canais_g=filtros[3], canais_inter=filtros[3]//2)
        self.decoder4 = BlocoConvolucional(filtros[4], filtros[3])
        
        self.upconv3 = nn.ConvTranspose2d(filtros[3], filtros[2], kernel_size=2, stride=2)
        self.atencao3 = PortaoAtencao(canais_x=filtros[2], canais_g=filtros[2], canais_inter=filtros[2]//2)
        self.decoder3 = BlocoConvolucional(filtros[3], filtros[2])
        
        self.upconv2 = nn.ConvTranspose2d(filtros[2], filtros[1], kernel_size=2, stride=2)
        self.atencao2 = PortaoAtencao(canais_x=filtros[1], canais_g=filtros[1], canais_inter=filtros[1]//2)
        self.decoder2 = BlocoConvolucional(filtros[2], filtros[1])
        
        self.upconv1 = nn.ConvTranspose2d(filtros[1], filtros[0], kernel_size=2, stride=2)
        self.atencao1 = PortaoAtencao(canais_x=filtros[0], canais_g=filtros[0], canais_inter=filtros[0]//2)
        self.decoder1 = BlocoConvolucional(filtros[1], filtros[0])
        
        self.saida = nn.Conv2d(filtros[0], classes_saida, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))
        
        b = self.gargalo(self.maxpool(e4))
        
        d4 = self.upconv4(b)
        x4 = self.atencao4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        x3 = self.atencao3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        x2 = self.atencao2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        x1 = self.atencao1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        saida = self.saida(d1)
        return saida
