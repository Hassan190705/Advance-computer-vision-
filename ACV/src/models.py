"""
Models for Semantic Segmentation Experiment
Assignment B - Unit 3: Advanced Computer Vision (CS107203CS)

Implements:
1. Baseline: Modest Encoder-Decoder with standard convolutions (dilation=1).
2. Controlled Change Variant 1: Dilated Bottleneck (dilation=2, 4) with exactly matched parameter budget (0% parameter delta).
3. Controlled Change Variant 2: Additive Skip Fusion vs Concatenation Skip Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class ConvBlock(nn.Module):
    """
    Convolution block: [Conv2d -> BatchNorm -> ReLU]
    Can be configured as single or double convolution. Supports configurable dilation rate.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, double_conv: bool = False):
        super().__init__()
        padding = dilation
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if double_conv:
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip fusion.
    Supports either 'concat' (Baseline) or 'add' (Controlled change).
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, skip_fusion: str = "concat", double_conv: bool = False):
        super().__init__()
        self.skip_fusion = skip_fusion
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        if skip_fusion == "concat":
            fusion_in_channels = in_channels + skip_channels
            self.conv = ConvBlock(fusion_in_channels, out_channels, double_conv=double_conv)
        elif skip_fusion == "add":
            self.skip_proj = nn.Conv2d(skip_channels, in_channels, kernel_size=1, bias=False) if skip_channels != in_channels else nn.Identity()
            self.conv = ConvBlock(in_channels, out_channels, double_conv=double_conv)
        else:
            raise ValueError(f"Unknown skip_fusion: {skip_fusion}")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.skip_fusion == "concat":
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            fused = torch.cat([x, skip], dim=1)
        else:  # add
            skip_proj = self.skip_proj(skip)
            if x.shape[2:] != skip_proj.shape[2:]:
                x = F.interpolate(x, size=skip_proj.shape[2:], mode="bilinear", align_corners=False)
            fused = x + skip_proj
        return self.conv(fused)


class ModestEncoderDecoder(nn.Module):
    """
    Modest Encoder-Decoder Architecture for Semantic Segmentation.
    
    Parameters:
    - num_classes: Number of target semantic categories (11 for CamVid benchmark).
    - mechanism:
        'baseline' -> Standard Bottleneck (dilation=1), Concatenation Skips.
        'dilated_bottleneck' -> Dilated Bottleneck (dilation=2 and 4), Concatenation Skips.
        'additive_skips' -> Standard Bottleneck (dilation=1), Additive Skip Fusion.
    - base_channels: Starting feature map channels (default: 16).
    - double_conv: Whether each stage has 1 or 2 conv layers (default: False for fast CPU iteration).
    """
    def __init__(
        self,
        num_classes: int = 11,
        mechanism: str = "baseline",
        base_channels: int = 16,
        double_conv: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mechanism = mechanism
        self.base_channels = base_channels
        self.double_conv = double_conv

        c = [base_channels * (2 ** i) for i in range(5)]  # [16, 32, 64, 128, 256]

        # --- 1. ENCODER ---
        self.enc1 = ConvBlock(3, c[0], double_conv=double_conv)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(c[0], c[1], double_conv=double_conv)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(c[1], c[2], double_conv=double_conv)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = ConvBlock(c[2], c[3], double_conv=double_conv)
        self.pool4 = nn.MaxPool2d(2, 2)

        # --- 2. BOTTLENECK (Ablation Mechanism: Receptive Field Dilation) ---
        if mechanism == "dilated_bottleneck":
            # Two dilated conv blocks with dilation=2 and dilation=4
            # Exact same weight matrix shape (128, 128, 3, 3) -> 0% parameter count change!
            self.bottleneck = nn.Sequential(
                ConvBlock(c[3], c[3], dilation=2, double_conv=double_conv),
                ConvBlock(c[3], c[3], dilation=4, double_conv=double_conv)
            )
        else:
            # Baseline & Additive Skips: standard dilation=1
            self.bottleneck = nn.Sequential(
                ConvBlock(c[3], c[3], dilation=1, double_conv=double_conv),
                ConvBlock(c[3], c[3], dilation=1, double_conv=double_conv)
            )

        # --- 3. DECODER ---
        fusion_type = "add" if mechanism == "additive_skips" else "concat"

        self.dec4 = DecoderBlock(c[3], c[2], c[2], skip_fusion=fusion_type, double_conv=double_conv)
        self.dec3 = DecoderBlock(c[2], c[1], c[1], skip_fusion=fusion_type, double_conv=double_conv)
        self.dec2 = DecoderBlock(c[1], c[0], c[0], skip_fusion=fusion_type, double_conv=double_conv)
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(c[0], c[0], double_conv=double_conv)

        # --- 4. CLASSIFICATION HEAD ---
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder path
        e1 = self.enc1(x)             # [B, 32, H, W]
        p1 = self.pool1(e1)           # [B, 32, H/2, W/2]

        e2 = self.enc2(p1)            # [B, 64, H/2, W/2]
        p2 = self.pool2(e2)           # [B, 64, H/4, W/4]

        e3 = self.enc3(p2)            # [B, 128, H/4, W/4]
        p3 = self.pool3(e3)           # [B, 128, H/8, W/8]

        e4 = self.enc4(p3)            # [B, 256, H/8, W/8]
        p4 = self.pool4(e4)           # [B, 256, H/16, W/16]

        # Bottleneck
        b = self.bottleneck(p4)       # [B, 256, H/16, W/16]

        # Decoder path with skip connections
        d4 = self.dec4(b, e3)         # [B, 128, H/8, W/8]
        d3 = self.dec3(d4, e2)        # [B, 64, H/4, W/4]
        d2 = self.dec2(d3, e1)        # [B, 32, H/2, W/2]

        d1 = self.final_upsample(d2)  # [B, 32, H, W]
        d1 = self.dec1(d1)            # [B, 32, H, W]

        logits = self.head(d1)        # [B, 11, H, W]

        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits


def build_model(
    mechanism: str = "baseline",
    num_classes: int = 11,
    base_channels: int = 16,
    double_conv: bool = False
) -> ModestEncoderDecoder:
    """Helper factory to instantiate models."""
    return ModestEncoderDecoder(
        num_classes=num_classes,
        mechanism=mechanism,
        base_channels=base_channels,
        double_conv=double_conv
    )


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 96, 128)) -> Dict[str, Any]:
    """
    Computes total trainable parameters, buffer size, and forward output shape.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(*input_size)
    with torch.no_grad():
        out = model(dummy_input)

    return {
        "mechanism": getattr(model, "mechanism", "unknown"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_shape": list(input_size),
        "output_shape": list(out.shape),
    }


if __name__ == "__main__":
    baseline = build_model("baseline", base_channels=16, double_conv=False)
    dilated = build_model("dilated_bottleneck", base_channels=16, double_conv=False)
    additive = build_model("additive_skips", base_channels=16, double_conv=False)

    print("=== MODEL COMPARISON & PARAMETER BUDGET (Fast CPU Architecture) ===")
    b_summary = get_model_summary(baseline)
    d_summary = get_model_summary(dilated)
    a_summary = get_model_summary(additive)

    print(f"1. Baseline (Standard Bottleneck, Concat Skips): {b_summary['total_parameters']:,} parameters")
    print(f"2. Dilated Bottleneck Variant (d=2, 4):          {d_summary['total_parameters']:,} parameters (Delta: {d_summary['total_parameters'] - b_summary['total_parameters']} / 0.00%)")
    print(f"3. Additive Skips Variant:                     {a_summary['total_parameters']:,} parameters (Delta: {a_summary['total_parameters'] - b_summary['total_parameters']:,})")
