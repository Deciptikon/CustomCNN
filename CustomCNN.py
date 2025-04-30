import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Блоки Conv -> BN -> ReLU -> Pool
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        
        # Первый слой (без пулинга)
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Основные блоки
        self.block1 = conv_block(128, 16)    # 128 → 16
        self.block2 = conv_block(16, 32)     # 16 → 32
        self.block3 = conv_block(32, 64)     # 32 → 64
        self.block4 = conv_block(64, 128)    # 64 → 128
        self.block5 = conv_block(128, 256)   # 128 → 256
        self.block6 = conv_block(256, 512)   # 256 → 512
        
        # Финальные слои
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        # Классификатор
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1024 * 1 * 1, num_classes)

    def forward(self, x):
        x = self.conv0(x)           # [3,128,128] → [128,128,128]
        x = self.block1(x)          # [128,128,128] → [16,64,64]
        x = self.block2(x)          # [16,64,64] → [32,32,32]
        x = self.block3(x)          # [32,32,32] → [64,16,16]
        x = self.block4(x)          # [64,16,16] → [128,8,8]
        x = self.block5(x)          # [128,8,8] → [256,4,4]
        x = self.block6(x)          # [256,4,4] → [512,2,2]
        x = self.final_conv(x)      # [512,2,2] → [1024,1,1]
        
        x = torch.flatten(x, 1)      # [1024,1,1] → [batch, 1024]
        x = self.dropout(x)
        x = self.fc(x)               # [batch, num_classes]
        return x