import torch
import torch.nn.functional as F
from .. import nn
from .. import model
import numpy as np
from icecream import ic
from monai.networks.nets.swin_unetr import SwinUNETR
from typing import Any


class LightMRI3D(torch.nn.Module):
    """
    Lightweight 3D CNN for MRI feature extraction.

    Designed for limited data scenarios (~1000 samples) where large models
    like DenseNet (11M params) or SwinUNETR (62M params) would overfit.

    Architecture motivation:
    - Uses depthwise separable convolutions (like MobileNet) to reduce parameters
    - Aggressive pooling to quickly reduce spatial dimensions
    - Strong regularization (dropout, weight decay built-in)
    - Global average pooling instead of fully connected layers
    - ~500K parameters (vs 11M for DenseNet, 62M for SwinUNETR)

    Input: (B, 1, 128, 128, 128)
    Output: (B, out_dim)
    """

    def __init__(self, in_channels=1, out_dim=128, dropout=0.3):
        super(LightMRI3D, self).__init__()

        # Stage 1: Initial feature extraction
        # (B, 1, 128, 128, 128) -> (B, 32, 64, 64, 64)
        self.conv1 = torch.nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = torch.nn.GroupNorm(8, 32)  # GroupNorm for small batches

        # Stage 2: Depthwise separable conv block
        # (B, 32, 64, 64, 64) -> (B, 64, 32, 32, 32)
        self.dw_conv2 = torch.nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.pw_conv2 = torch.nn.Conv3d(32, 64, kernel_size=1)
        self.bn2 = torch.nn.GroupNorm(8, 64)

        # Stage 3: Depthwise separable conv block
        # (B, 64, 32, 32, 32) -> (B, 128, 16, 16, 16)
        self.dw_conv3 = torch.nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.pw_conv3 = torch.nn.Conv3d(64, 128, kernel_size=1)
        self.bn3 = torch.nn.GroupNorm(8, 128)

        # Stage 4: Depthwise separable conv block
        # (B, 128, 16, 16, 16) -> (B, 256, 8, 8, 8)
        self.dw_conv4 = torch.nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)
        self.pw_conv4 = torch.nn.Conv3d(128, 256, kernel_size=1)
        self.bn4 = torch.nn.GroupNorm(8, 256)

        # Global average pooling + projection
        # (B, 256, 8, 8, 8) -> (B, 256) -> (B, out_dim)
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(256, out_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.GroupNorm, torch.nn.BatchNorm3d)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Stage 2
        x = self.dw_conv2(x)
        x = F.relu(self.bn2(self.pw_conv2(x)))

        # Stage 3
        x = self.dw_conv3(x)
        x = F.relu(self.bn3(self.pw_conv3(x)))

        # Stage 4
        x = self.dw_conv4(x)
        x = F.relu(self.bn4(self.pw_conv4(x)))

        # Global pooling + FC
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ImagingModelWrapper(torch.nn.Module):
    def __init__(
            self,
            arch: str = 'ViTAutoEnc',
            tgt_modalities: dict | None = {},
            img_size: int | None = 128,
            patch_size: int | None = 16,
            ckpt_path: str | None = None,
            train_backbone: bool = False,
            out_dim: int = 128,
            layers: int | None = 1,
            device: str = 'cpu',
            fusion_stage: str = 'middle',
            ):
        super(ImagingModelWrapper, self).__init__()

        self.arch = arch
        self.tgt_modalities = tgt_modalities
        self.img_size = img_size
        self.patch_size = patch_size
        self.train_backbone = train_backbone
        self.ckpt_path = ckpt_path
        self.device = device
        self.out_dim = out_dim
        self.layers = layers
        self.fusion_stage = fusion_stage
        
        
        if "swinunetr" in self.arch.lower():
            if "emb" not in self.arch.lower():
                import os
                # Try multiple paths for pretrained weights
                swin_ckpt_paths = [
                    '/home/vatsal/MRI_with_additional_features/pretrained_models/model_swinvit.pt',  # Local download
                    '/projectnb/ivc-ml/dlteif/pretrained_models/model_swinvit.pt',  # Cluster path
                ]
                swin_ckpt_path = None
                for path in swin_ckpt_paths:
                    if os.path.exists(path):
                        swin_ckpt_path = path
                        break

                # MONAI SwinUNETR API (v1.3+) no longer uses img_size parameter
                self.img_model = SwinUNETR(
                    in_channels=1,
                    out_channels=1,
                    feature_size=48,
                    spatial_dims=3,
                    use_checkpoint=False,  # Disable gradient checkpointing for stability
                )
                if swin_ckpt_path:
                    print(f"Loading pretrained SwinUNETR weights from {swin_ckpt_path}")
                    ckpt = torch.load(swin_ckpt_path, map_location='cpu')
                    ckpt["state_dict"] = {k.replace("swinViT.", "module."): v for k, v in ckpt["state_dict"].items()}
                    self.img_model.load_from(ckpt)
                    print("Pretrained weights loaded successfully!")
                else:
                    print(f"WARNING: No pretrained SwinUNETR checkpoint found. Training from scratch (not recommended).")
            self.dim = 768

        elif "vit" in self.arch.lower():    
            if "emb" not in self.arch.lower():
                # Initialize image model
                self.img_model = nn.__dict__[self.arch](
                    in_channels = 1,
                    img_size = self.img_size,
                    patch_size = self.patch_size,
                )

                if self.ckpt_path:
                    self.img_model.load(self.ckpt_path, map_location=self.device)
                self.dim = self.img_model.hidden_size
            else:
                self.dim = 768

        if "vit" in self.arch.lower() or "swinunetr" in self.arch.lower():    
            dim = self.dim
            if self.fusion_stage == 'middle':
                downsample = torch.nn.ModuleList()
                # print('Number of layers: ', self.layers)
                for i in range(self.layers):
                    if i == self.layers - 1:
                        dim_out = self.out_dim
                        ks = 2
                        stride = 2
                    else:
                        dim_out = dim // 2
                        ks = 2
                        stride = 2

                    downsample.append(
                        torch.nn.Conv1d(in_channels=dim, out_channels=dim_out, kernel_size=ks, stride=stride)
                    )

                    dim = dim_out

                    # Use GroupNorm instead of BatchNorm1d to handle small batch sizes
                    # GroupNorm with num_groups=1 is equivalent to LayerNorm
                    downsample.append(
                        torch.nn.GroupNorm(num_groups=1, num_channels=dim)
                    )
                    downsample.append(
                        torch.nn.ReLU()
                    )
                # downsample.append(torch.nn.Linear(8, self.out_dim))
                    
                    
                self.downsample = torch.nn.Sequential(*downsample)
            elif self.fusion_stage == 'late':
                self.downsample = torch.nn.Identity()
            else:
                pass
            
            # print('Downsample layers: ', self.downsample)
                
        elif "densenet" in self.arch.lower():
            if "emb" not in self.arch.lower():
                import os
                if self.ckpt_path is not None and os.path.exists(self.ckpt_path):
                    # Load from checkpoint
                    self.img_model = model.ImagingModel.from_ckpt(self.ckpt_path, device=self.device, img_backend=self.arch, load_from_ckpt=True).net_
                else:
                    # Initialize DenseNet from scratch
                    print(f"No checkpoint found at {self.ckpt_path}. Initializing DenseNet from scratch.")
                    self.img_model = nn.DenseNet(
                        tgt_modalities=self.tgt_modalities if self.tgt_modalities else {},
                        load_from_ckpt=False
                    )

                # Calculate actual output size based on image dimensions
                if isinstance(self.img_size, tuple):
                    test_img = torch.ones((1, 1) + self.img_size)
                else:
                    test_img = torch.ones((1, 1, self.img_size, self.img_size, self.img_size))
                with torch.no_grad():
                    out = self.img_model.features(test_img)
                    densenet_out_size = out.view(-1).size(0)
                print(f"DenseNet output size for input {self.img_size}: {densenet_out_size}")
            else:
                densenet_out_size = 3900  # Default for embeddings

            self.downsample = torch.nn.Linear(densenet_out_size, self.out_dim)

        elif "lightmri" in self.arch.lower():
            # Lightweight 3D CNN designed for limited data
            # ~500K params vs 11M for DenseNet
            self.img_model = LightMRI3D(
                in_channels=1,
                out_dim=self.out_dim,
                dropout=0.3
            )
            self.downsample = torch.nn.Identity()  # LightMRI already outputs out_dim
            print(f"LightMRI3D initialized with {self.img_model.count_parameters():,} parameters")

        # randomly initialize weights for downsample block
        for p in self.downsample.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            p.requires_grad = True
            
        if "emb" not in self.arch.lower():
            # freeze imaging model parameters
            if "densenet" in self.arch.lower():
                for n, p in self.img_model.features.named_parameters():
                    if not self.train_backbone:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                for n, p in self.img_model.tgt.named_parameters():
                    p.requires_grad = False
            elif "lightmri" in self.arch.lower():
                # LightMRI is always fully trainable (designed for limited data)
                for n, p in self.img_model.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in self.img_model.named_parameters():
                    # print(n, p.requires_grad)
                    if not self.train_backbone:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
    
    def forward(self, x):
        # print("--------ImagingModelWrapper forward--------")
        if "emb" not in self.arch.lower():
            if "swinunetr" in self.arch.lower():
                # Use encoder features only (not full U-Net)
                # swinViT returns list of features at different scales
                hidden_states = self.img_model.swinViT(x, normalize=True)
                # Take deepest features: (B, 768, 4, 4, 4)
                out = hidden_states[-1]
                # Flatten spatial dims: (B, 768, 4, 4, 4) -> (B, 768, 64)
                out = out.view(out.size(0), out.size(1), -1)
                # Apply Conv1d downsample
                out = self.downsample(out)
                out = torch.mean(out, dim=-1)
            elif "vit" in self.arch.lower():
                out = self.img_model(x, return_emb=True)
                ic(out.size())
                out = self.downsample(out)
                out = torch.mean(out, dim=-1)
            elif "densenet" in self.arch.lower():
                out = torch.nn.Sequential(*list(self.img_model.features.children()))(x)
                # print(out.size())
                out = torch.flatten(out, 1)
                out = self.downsample(out)
            elif "lightmri" in self.arch.lower():
                # LightMRI directly outputs (B, out_dim)
                out = self.img_model(x)
        else:
            # print(x.size())
            if "swinunetr" in self.arch.lower():
                x = torch.squeeze(x, dim=1)
                x = x.view(x.size(0),self.dim, -1)
            # print('x: ', x.size())    
            out = self.downsample(x)
            # print('out: ', out.size())
            if self.fusion_stage == 'middle':
                if "vit" in self.arch.lower() or "swinunetr" in self.arch.lower():
                    out = torch.mean(out, dim=-1)
                    # out = torch.mean(out, dim=1)
                else:
                    out = torch.squeeze(out, dim=1)
            elif self.fusion_stage == 'late':
                pass
            # print(out.shape)

        return out

