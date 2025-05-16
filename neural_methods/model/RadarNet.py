from typing import Tuple
import torch

class RadarNet(torch.nn.Module):
    def __init__(self, channels=10):  
        super(RadarNet, self).__init__()
        
        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 64, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(256),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(512),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock5 = torch.nn.Sequential(
            torch.nn.Conv1d(512, 256, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(256),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.downsample1 = torch.nn.MaxPool1d(kernel_size=2)
        self.downsample2 = torch.nn.MaxPool1d(kernel_size=2)

        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        self.ConvBlock6 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 128, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock7 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 64, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock8 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock9 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 64, 7, stride=1, padding=3, padding_mode='reflect'),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout1d(0.2),
        )

        self.ConvBlock10 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 1, 1, stride=1, padding=0)
        )
        
    def forward(self, x_orig: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = self.ConvBlock1(x_orig)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.downsample1(x)
        x = self.ConvBlock4(x)
        x = self.downsample2(x)
        z_IQ  = self.ConvBlock5(x)
        
        z = self.ConvBlock6(z_IQ)
        z = self.ConvBlock7(z)
        z = self.upsample1(z)
        z = self.ConvBlock8(z)
        z = self.upsample2(z)
        z = self.ConvBlock9(z)
        x_decoded = self.ConvBlock10(z)        
        return x_decoded.squeeze(1), z_IQ