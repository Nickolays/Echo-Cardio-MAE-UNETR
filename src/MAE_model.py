import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from transformers import SegformerModel, SegformerConfig


class MaskedAutoencoderSegformer(pl.LightningModule):
    def __init__(self, encoder_name='nvidia/segformer-b0-finetuned-ade-512-512',
                 image_size=224, patch_size=16, mask_ratio=0.75, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio
        self.lr = lr

        # Load pretrained SegFormer encoder (ViT-based segmentation model)
        config = SegformerConfig.from_pretrained(encoder_name)
        self.encoder = SegformerModel.from_pretrained(encoder_name)
        # self.encoder = self.encoder.encoder

        # Decoder: map encoded tokens back to image patches
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1]),
            nn.GELU(),
            nn.Linear(config.hidden_sizes[-1], patch_size * patch_size),  # Reconstruct grayscale patch
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch_size
        patches = imgs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).flatten(2)  # (B, N, C*p*p)
        return patches

    def unpatchify(self, patches):
        B, N, D = patches.shape
        p = self.patch_size
        h = w = int(self.image_size // p)
        patches = patches.view(B, h * w, 1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(B, 1, h, p, w, p)
        return patches.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, h * p, w * p)

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, imgs):
        patches = self.patchify(imgs)
        x_masked, mask, ids_restore = self.random_masking(patches)

        # Encode
        encoder_outputs = self.encoder(pixel_values=imgs)
        encoded_tokens = encoder_outputs.last_hidden_state

        # Decode
        pred = self.decoder(encoded_tokens)  # (B, N_masked, patch_dim)

        # Reconstruct full patch sequence with predicted masked patches
        B, N, D = patches.shape
        full_pred = torch.zeros_like(patches)
        mask_indices = (mask == 1)
        full_pred[mask_indices] = pred.view(-1)
        full_pred[~mask_indices] = patches[~mask_indices]

        imgs_reconstructed = self.unpatchify(full_pred)
        return imgs_reconstructed, imgs

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        recons, originals = self(imgs)
        loss = F.mse_loss(recons, originals)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx == 0:
            imgs, _ = batch
            recons, _ = self(imgs[:4])
            grid = make_grid(recons, nrow=2)
            self.logger.experiment.add_image("reconstructions", grid, self.global_step)