#Shallow Information Hiding Module
#Deep Information Hiding Module
#Construction Container Image Module

import torch
import torch.nn as nn
from transformers import SwinModel

class SwinSteganography(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinSteganography, self).__init__()

        #loading the pretrained Swin transformer model 
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.input_proj = nn.Conv2d(6,3, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1)
        )

    def forward(self, cover, secret):
        x = torch.cat((cover, secret), dim=1)
        x = self.input_proj(x)

        features = self.swin(x).last_hidden_state

        B, N, C = features.shape
        H, W = 14, 14
        features = features.permute(0, 2, 1).view(B, C, H, W)

        container = self.decoder(features)

        return container


class SwinExtractionNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinExtractionNetwork, self).__init__()

        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1)
        )

        def forward(self, container):
        # Extract features
            features = self.swin(container).last_hidden_state  # Shape: (B, N, 768)
        
        # Reshape back to image format
            B, N, C = features.shape
            H, W = 14, 14  # Swin Tiny produces 14x14 feature maps
            features = features.permute(0, 2, 1).view(B, C, H, W)

        # Generate extracted secret image
            secret = self.decoder(features)  # Shape: (B, 3, H, W)

            return secret