""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""
import torch
import torch.nn as nn


class ContrastPhys(nn.Module):
    def __init__(self, S=2, in_ch=3):
        super().__init__()

        self.S = S # S is the spatial dimension of ST-rPPG block

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, 
                        padding=(0, 2, 2), padding_mode='reflect'),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, 
                        padding=(1, 1, 1), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, 
                        padding=(1, 0, 0), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, 
                        padding=(1, 0, 0), padding_mode='reflect'),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.Dropout3d(p=0.2)
        )
                    

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, S, S)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, 
                        padding=(0, 0, 0), padding_mode='reflect'),
        )

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / (stds + 1e-4) # (B, C, T, 128, 128)
 
        parity = []
        x = self.start(x) # (B, C, T, 128, 128)
        x = self.loop1(x) # (B, 64, T, 64, 64)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x) # (B, 64, T/2, 32, 32)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x) # (B, 64, T/4, 16, 16)
        x = self.loop4(x) # (B, 64, T/4, 8, 8)

        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T/2, 8, 8)
        x = self.decoder1(x) # (B, 64, T/2, 8, 8)
        x = torch.nn.functional.pad(x, (0,0,0,0,0,parity[-1]), mode='replicate')
        x = torch.nn.functional.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T, 8, 8)
        x = self.decoder2(x) # (B, 64, T, 8, 8)
        x = torch.nn.functional.pad(x, (0,0,0,0,0,parity[-2]), mode='replicate')
        x = self.end(x) # (B, 1, T, S, S), ST-rPPG block

        x_list = []
        for a in range(self.S):
            for b in range(self.S):
                x_list.append(x[:,:,:,a,b]) # (B, 1, T)

        x = sum(x_list)/(self.S*self.S) # (B, 1, T)
        X = torch.cat(x_list+[x], 1) # (B, N, T), flatten all spatial signals to the second dimension
        return X