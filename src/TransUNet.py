import json, torch, os
import torch.nn as nn
from transformers import ViTConfig, ViTModel
import torch.nn.functional as F
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, cnn_out, vit_out):
        assert cnn_out.shape[2] == vit_out.shape[2], "CNN and ViT outputs must have the same channel dimension"
        x = torch.cat([cnn_out, vit_out], dim=1)
        x = self.conv(x)
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels=768, out_channels=512, num_classes=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.n_layers = (256, 128, 64)
        self.decode_layer_1 = DecoderLayer(512+512, 256)
        self.decode_layer_2 = DecoderLayer(256+256, 128)
        self.decode_layer_3 = DecoderLayer(128+64, 64)

        self.vit_conv_layer = ConvLayer(in_channels, out_channels)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, vit_out, skip_outs):
    
        # Reshape (n_patch, D) to (D, H/16, W/16)
        # vit_out = vit_out.permute(0, 2, 1).reshape(vit_out.shape[0], vit_out.shape[2], 14, 14)
        # Put forward from conv layer
        vit_out = self.vit_conv_layer(vit_out)  # (512, H/16, W/16)
        # Upsample to match skip connections
        x = self.up(vit_out)
        # Concatenate all skip connections
        x = self.decode_layer_1(x, skip_outs[2])
        x = self.decode_layer_2(x, skip_outs[1])
        x = self.decode_layer_3(x, skip_outs[0])
        # Final upsample to match input size
        x = self.head(x)  # (num_classes, H, W)
        return x


class TransUNet(nn.Module):
    def __init__(self, 
                 mae_config_path: str,
                 mae_checkpoint_path: str,
                 num_classes: int = 4,
                 freeze_encoder_blocks: int = 0):
        super().__init__()
        # I. Load transfoer like second encoder from MAE pretrained weights
        # 1) Load your MAE config.json
        with open(mae_config_path, 'r') as f:
            mae_cfg = json.load(f)

        # 2) Extract all fields ViTConfig needs
        # vit_kwargs = dict(
        #     image_size=mae_cfg["image_size"],
        #     patch_size=mae_cfg["patch_size"],
        #     num_channels=mae_cfg["num_channels"],
        #     hidden_size=mae_cfg["hidden_size"],
        #     num_hidden_layers=mae_cfg["num_hidden_layers"],
        #     num_attention_heads=mae_cfg["num_attention_heads"],
        #     intermediate_size=mae_cfg["intermediate_size"],
        #     qkv_bias=mae_cfg["qkv_bias"],
        #     hidden_act=mae_cfg["hidden_act"],
        #     hidden_dropout_prob=mae_cfg["hidden_dropout_prob"],
        #     layer_norm_eps=mae_cfg["layer_norm_eps"],
        #     add_pooling_layer=False  # Not creates that pooler
        # )
        # vit_config = ViTConfig(**vit_kwargs)
        vit_config = ViTConfig(**mae_cfg)

        # 3) Instantiate the encoder
        self.transformer = ViTModel(vit_config)
        # Freeze CLS token embedding
        # self.transformer.embeddings.cls_token.requires_grad_(False)
        # Freeze pooler parameters (if you kept it)
        for param in getattr(self.transformer, "pooler", []).parameters():
            param.requires_grad_(False)

        # 4) Load MAE weights (only encoder part)
        ckpt = torch.load(mae_checkpoint_path, map_location=device)
        # Assume keys are like 'vit.encoder.layer...', adjust if needed
        vit_state_dict = {k.replace("vit.", ""): v 
                              for k, v in ckpt.items() 
                              if k.startswith("vit.")}
  
        missing, unexpected = self.transformer.load_state_dict(vit_state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        # II. CNN backbone (ResNet50)
        self.backbone = resnet50(pretrained=True)
        self.backbone_layers = nn.ModuleList([
            nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu), self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ])
        self.projection = nn.Conv2d(1024, mae_cfg["hidden_size"], kernel_size=1)  # Project to hidden size channels
        # III. Decoder part
        # Decoder with Cascaded Upsampler (CUP)
        self.decoder = DecoderBlock(mae_cfg['hidden_size'], 512, num_classes=num_classes)  # 768 -> 512

        # 'Custom' cls token and pos embeddings 
        self.cls_token = self.transformer.embeddings.cls_token
        self.pos_embed = self.transformer.embeddings.position_embeddings
        self.dropout = self.transformer.embeddings.dropout

    def forward(self, x):
        """
        Forward pass for the TransUNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size,
                            C is the number of channels, and H, W are the height and width
                            of the input images.

        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W), where num_classes
                        is the number of segmentation classes.

        The function performs the following steps:
        1. Extracts feature maps using a CNN backbone.
        2. Resizes input if dimensions are not 224x224.
        3. Applies a Vision Transformer (ViT) to generate hidden states.
        4. Uses the last hidden state (excluding the CLS token) for decoding.
        5. Decodes using skip connections from the CNN feature maps and returns
        the final segmentation output.
        """
        # a) CNN feature maps
        f0 = self.backbone_layers[0](x)    # (B, 64, H/2, W/2)
        f1 = self.backbone_layers[1](f0)   # (B, 64, H/4, W/4)
        f2 = self.backbone_layers[2](f1)   # (B, 256, H/4, W/4)
        f3 = self.backbone_layers[3](f2)   # (B, 512, H/8, W/8)
        f4 = self.backbone_layers[4](f3)   # (B, 1024, H/16, W/16)
        # Project to match ViT hidden size
        f4_proj = self.projection(f4)  # (B, hidden_size, H/16, W/16)
        # 2. Flatten spatial patches to tokens
        Hf = f4_proj.shape[2]
        N = Hf * Hf
        tokens = f4_proj.flatten(2).transpose(1, 2)  # (B, N, hidden)
        # 3. Add cls token and positional embeddings
        B = tokens.shape[0]  # batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)        # (B, 1, hidden)
        tokens = torch.cat([cls_tokens, tokens], dim=1)      # (B, N+1, hidden)
        tokens = tokens + self.pos_embed                     # add position embeddings
        tokens = self.dropout(tokens)

        # vit_out = self.transformer(pixel_values=tokens, output_hidden_states=False)
        vit_out = self.transformer.encoder(tokens)
        print("ViT output shape:", vit_out.last_hidden_state.shape)  # (B, N, hidden_size)
        vit_out = vit_out.last_hidden_state[:, 1:, :]  # (B, N, hidden_size), exclude CLS token

        # 4. Reshape back to spatial map
        vit_out = vit_out.permute(0, 2, 1).reshape(vit_out.shape[0], vit_out.shape[2], Hf, Hf)  # (B, hidden_size, H/16, W/16)

        out = self.decoder(vit_out, [f0, f2, f3])
        return out
    


# Paths to your MAE config and checkpoint
# mae_cfg_path = "models/mae_pretrained/config.json"
# mae_ckpt_path = "models/mae_pretrained.pth"  # or .pth

# # 1) Instantiate TransUNet with MAE weights
# model = TransUNet(
#     mae_config_path=mae_cfg_path,
#     mae_checkpoint_path=mae_ckpt_path,
#     num_classes=3,
#     freeze_encoder_blocks=0
# )
# x = torch.randn(2, 3, 224, 224)
# seg_logits = model(x)  # shape (2, 4, 224, 224)

# print("Output shape:", seg_logits.shape)  # Should be (B, num_classes, H, W)