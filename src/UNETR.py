import json, torch, os
import torch.nn as nn
from transformers import ViTConfig, ViTModel
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        return x
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip_conn):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv(x)
        x = torch.cat((x, skip_conn), dim=1)
        return x


class UNETR(nn.Module):
    def __init__(self, 
                 mae_config_path: str,
                 mae_checkpoint_path: str,
                 num_classes: int = 4,
                 freeze_encoder_blocks: int = 0):
        super().__init__()
        # 1) Load your MAE config.json
        with open(mae_config_path, 'r') as f:
            mae_cfg = json.load(f)

        # 2) Extract all fields ViTConfig needs
        vit_kwargs = dict(
            image_size=mae_cfg["image_size"],
            patch_size=mae_cfg["patch_size"],
            num_channels=mae_cfg["num_channels"],
            hidden_size=mae_cfg["hidden_size"],
            num_hidden_layers=mae_cfg["num_hidden_layers"],
            num_attention_heads=mae_cfg["num_attention_heads"],
            intermediate_size=mae_cfg["intermediate_size"],
            qkv_bias=mae_cfg["qkv_bias"],
            hidden_act=mae_cfg["hidden_act"],
            hidden_dropout_prob=mae_cfg["hidden_dropout_prob"],
            layer_norm_eps=mae_cfg["layer_norm_eps"],
            add_pooling_layer=False  # Not creates that pooler
        )
        vit_config = ViTConfig(**vit_kwargs)

        # 3) Instantiate the encoder
        self.encoder = ViTModel(vit_config)
        # if hasattr(self.encoder, "pooler"):
        #     del self.encoder.pooler
        # Freeze CLS token embedding
        self.encoder.embeddings.cls_token.requires_grad_(False)

        # Freeze pooler parameters (if you kept it)
        for param in getattr(self.encoder, "pooler", []).parameters():
            param.requires_grad_(False)

        # 4) Load MAE weights (only encoder part)
        ckpt = torch.load(mae_checkpoint_path, map_location=device)
        # Assume keys are like 'vit.encoder.layer...', adjust if needed
        encoder_state_dict = {k.replace("vit.", ""): v 
                              for k, v in ckpt.items() 
                              if k.startswith("vit.")}
        # self.encoder.load_state_dict(encoder_state_dict, strict=False)
        missing, unexpected = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        # 5) Optionally freeze first N transformer blocks
        for name, param in self.encoder.named_parameters():
            # block names look like 'encoder.layer.0...' through 'encoder.layer.11...'
            layer_num = None
            parts = name.split('.')
            if len(parts) > 2 and parts[1] == "layer":
                layer_num = int(parts[2])
            if layer_num is not None and layer_num < freeze_encoder_blocks:
                param.requires_grad = False

        # 6) Decoder heads
        hidden_size = vit_config.hidden_size  # e.g. 768
        # We will take outputs at layers 3,6,9,12 for skip connections
        self.proj3  = nn.Sequential(
            DeconvBlock(hidden_size, 256),
            DeconvBlock(256, 128),
            DeconvBlock(128, 128),
        ) # Conv2d(hidden_size, 256, kernel_size=1)
        self.proj6  = nn.Sequential(
            DeconvBlock(hidden_size, 512),
            DeconvBlock(512, 256),
        ) # nn.Conv2d(hidden_size, 512, kernel_size=1)
        self.proj9  = nn.Sequential(
            DeconvBlock(hidden_size, 512),
        ) # nn.Conv2d(hidden_size, 768, kernel_size=1)
        self.proj12 = nn.ConvTranspose2d(hidden_size, 512, kernel_size=2, stride=2)

        # And for the input image
        self.proj0  = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
        )
        # 7) Skip connections
        self.sc_3 = UpBlock(256, 64)
        self.sc6 = UpBlock(512, 128)
        self.sc9 = UpBlock(1024, 256)

        # 8) Initialize head
        self.head = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        
    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        # 1) Get ViT outputs at all layers
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        hs = outputs.hidden_states  # tuple of length num_hidden_layers+1

        # 2) Extract patch maps at layers 3,6,9,12
        #    hidden_states[i] is (B, num_patches+1, hidden_size)
        z3  = self._reshape_layer(hs[3], H, B)
        z6  = self._reshape_layer(hs[6], H, B)
        z9  = self._reshape_layer(hs[9], H, B)
        z12 = self._reshape_layer(hs[12],H, B)

        # 3) Project to decoder channels
        p0  = self.proj0(x)
        p3  = self.proj3(z3)
        p6  = self.proj6(z6)
        p9  = self.proj9(z9)
        p12 = self.proj12(z12)

        # 4) Concatenate and decode
        sc_9_12 = torch.cat([p9, p12], dim=1)       # 1024 
        sc_9_6 = self.sc9(x=sc_9_12, skip_conn=p6)  # 1024 -> 256
        sc_6_3 = self.sc6(x=sc_9_6, skip_conn=p3)   # 512 -> 128
        sc_3_0 = self.sc_3(x=sc_6_3, skip_conn=p0)    # 256 -> 64

        out  = self.head(sc_3_0)
        return out
    
    def _reshape_layer(self, h, H, B):
            # drop cls token
            h = h[:, 1:, :]
            p = int((H // self.encoder.config.patch_size))
            h = h.permute(0,2,1).view(B, self.encoder.config.hidden_size, p, p)
            return h




# class UNETR(nn.Module):
#     def __init__(self, 
#                  mae_config_path: str,
#                  mae_checkpoint_path: str,
#                  num_classes: int = 4,
#                  freeze_encoder_blocks: int = 0):
#         super().__init__()
#         # 1) Load your MAE config.json
#         with open(mae_config_path, 'r') as f:
#             mae_cfg = json.load(f)

#         # 2) Extract all fields ViTConfig needs
#         vit_kwargs = dict(
#             image_size=mae_cfg["image_size"],
#             patch_size=mae_cfg["patch_size"],
#             num_channels=mae_cfg["num_channels"],
#             hidden_size=mae_cfg["hidden_size"],
#             num_hidden_layers=mae_cfg["num_hidden_layers"],
#             num_attention_heads=mae_cfg["num_attention_heads"],
#             intermediate_size=mae_cfg["intermediate_size"],
#             qkv_bias=mae_cfg["qkv_bias"],
#             hidden_act=mae_cfg["hidden_act"],
#             hidden_dropout_prob=mae_cfg["hidden_dropout_prob"],
#             layer_norm_eps=mae_cfg["layer_norm_eps"],
#         )
#         vit_config = ViTConfig(**vit_kwargs)

#         # 3) Instantiate the encoder
#         self.encoder = ViTModel(vit_config)

#         # 4) Load MAE weights (only encoder part)
#         ckpt = torch.load(mae_checkpoint_path, map_location="cpu")
#         # Assume keys are like 'vit.encoder.layer...', adjust if needed
#         encoder_state_dict = {k.replace("vit.encoder.", ""): v 
#                               for k, v in ckpt.items() 
#                               if k.startswith("vit.encoder.")}
#         self.encoder.load_state_dict(encoder_state_dict, strict=False)

#         # 5) Optionally freeze first N transformer blocks
#         for name, param in self.encoder.named_parameters():
#             # block names look like 'encoder.layer.0...' through 'encoder.layer.11...'
#             layer_num = None
#             parts = name.split('.')
#             if len(parts) > 2 and parts[1] == "layer":
#                 layer_num = int(parts[2])
#             if layer_num is not None and layer_num < freeze_encoder_blocks:
#                 param.requires_grad = False

#         # 6) Decoder heads
#         hidden_size = vit_config.hidden_size  # e.g. 768
#         # We will take outputs at layers 3,6,9,12 for skip connections
#         self.proj3  = nn.Conv2d(hidden_size, 256, kernel_size=1)
#         self.proj6  = nn.Conv2d(hidden_size, 512, kernel_size=1)
#         self.proj9  = nn.Conv2d(hidden_size, 768, kernel_size=1)
#         self.proj12 = nn.Conv2d(hidden_size, 1024, kernel_size=1)

#         # Simple UNet‐style decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256+512+768+1024, 512, 2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, 2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, 3, padding=1),  nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, 2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, num_classes, 1)
#         )
#         # 7) Initialize head
#         self.head = nn.Sequential(
#             ConvBlock(128, 64),
#             ConvBlock(64, 32),
#             nn.Conv2d(32, num_classes, kernel_size=1),
#         )
        
#     def forward(self, x):
#         """
#         x: (B, C, H, W)
#         returns: (B, num_classes, H, W)
#         """
#         B, C, H, W = x.shape
#         # 1) Get ViT outputs at all layers
#         outputs = self.encoder(pixel_values=x, output_hidden_states=True)
#         hs = outputs.hidden_states  # tuple of length num_hidden_layers+1

#         # 2) Extract patch maps at layers 3,6,9,12
#         #    hidden_states[i] is (B, num_patches+1, hidden_size)
#         z3  = self._reshape_layer(hs[3], H, B)
#         z6  = self._reshape_layer(hs[6], H, B)
#         z9  = self._reshape_layer(hs[9], H, B)
#         z12 = self._reshape_layer(hs[12],H, B)

#         # 3) Project to decoder channels
#         p3  = self.proj3(z3)
#         p6  = self.proj6(z6)
#         p9  = self.proj9(z9)
#         p12 = self.proj12(z12)

#         # 4) Concatenate and decode
#         feat = torch.cat([p3, p6, p9, p12], dim=1)
#         out  = self.decoder(feat)
#         return out
    
#     def _reshape_layer(self, h, H, B):
#             # drop cls token
#             h = h[:, 1:, :]
#             p = int((H // self.encoder.config.patch_size))
#             h = h.permute(0,2,1).view(B, self.encoder.config.hidden_size, p, p)
#             return h