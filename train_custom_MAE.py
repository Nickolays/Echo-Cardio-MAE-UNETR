import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from transformers import ViTModel, ViTConfig

from src.MAE_model import MaskedAutoencoderViT





# Example dataloader for grayscale images
def get_dataloaders(data_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os

    dataloader = get_dataloaders("/path/to/echo/data")
    model = MaskedAutoencoderViT()

    checkpoint = ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min")
    trainer = Trainer(max_epochs=1, precision=16, callbacks=[checkpoint], accelerator="auto")
    trainer.fit(model, dataloader)
