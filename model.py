from torch import nn
import torch

import torch.nn as nn

class YOLOModule(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        
        self.features = nn.Sequential(
            # 第一组
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二组
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三组
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四组（4次重复）
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # 最后两个卷积
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五组
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            # 第六组
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(start_dim=1),  # [B, 1024, 7, 7] -> [B, 1024*7*7]
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),  # 7*7*(2*5+20)=1470
            nn.Unflatten(dim=1, unflattened_size=(S, S, B * 5 + C))  # [B, 1470] -> [B, 7, 7, 30]
        )

    def forward(self, x):
        x = self.features(x)      # [B, 3, 448, 448] -> [B, 1024, 7, 7]
        x = self.fc_layers(x)     # [B, 1024, 7, 7] -> [B, 7, 7, 30]
        return x

def create_model(args):
    model = YOLOModule()
    
    return model