#Shallow Information Hiding Module
#Deep Information Hiding Module
#Construction Container Image Module

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel
from torchvision.models import vgg16

class SwinSteganography(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinSteganography, self).__init__()

        # Load Pre-trained Swin Transformer
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        # Reduce 6-channel (Cover + Secret) to 3-channel
        self.input_proj = nn.Conv2d(6, 3, kernel_size=3, padding=1)

        # Feature Refinement Layer to enhance embeddings
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Upsample Layer (Fixes Dimension Mismatch)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

        # Residual Blocks for better decoding
        self.residual_blocks = nn.Sequential(
            self._residual_block(768),
            self._residual_block(768)
        )

        # Adaptive Attention Layer for Skip Connection
        self.attention = nn.Conv2d(3, 3, kernel_size=1)

        # Decoder with Residual Blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self._residual_block(512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self._residual_block(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self._residual_block(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Normalize output to [0,1]
        )

    def _residual_block(self, channels):
        """Residual block to improve reconstruction."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, cover, secret):
        # Concatenate cover and encrypted secret image
        x = torch.cat((cover, secret), dim=1)  # Shape: (B, 6, 224, 224)

        # Project input to 3 channels
        x = self.input_proj(x)  # Shape: (B, 3, 224, 224)

        # Save a copy for skip connection
        x_skip = x.clone()

        # Feature refinement before Swin processing
        x = self.feature_refiner(x)

        # Extract Swin Transformer features
        features = self.swin(x).last_hidden_state  # Shape: (B, N, 768)

        # Reshape feature map for decoding
        B, N, C = features.shape
        H, W = int(N ** 0.5), int(N ** 0.5)  # Swin downsamples input

        assert H * W == N, f"Unexpected shape: {features.shape}, H={H}, W={W}"

        features = features.permute(0, 2, 1).view(B, C, H, W)  # Shape: (B, 768, H, W)

        # Apply Residual Blocks to refine features
        features = self.residual_blocks(features)

        # **Fix Dimension Mismatch** by upsampling Swin features
        features = self.upsample(features)  # Now (B, 768, 224, 224)

        # Decode features back into stego image
        stego = self.decoder(features)  # Shape: (B, 3, 224, 224)

        # Apply Adaptive Attention to Skip Connection
        x_skip = self.attention(x_skip)
        stego = stego * x_skip + (1 - x_skip) * x

        return stego
