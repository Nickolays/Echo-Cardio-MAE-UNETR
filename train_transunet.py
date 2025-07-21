import torch, cv2, os, yaml
import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl 

from src.TransUNet import TransUNet
from src.unetr_model import UNETRModule
from src.datasets import LeftCamusDataset
from src.utils import get_image_filepaths


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ------------------------- Fixing the Label Pipeline -------------------------
# When using torchvision.transforms.Resize, it interpolates masks as float
# and can produce non-binary values like 0.5. Best practice for masks:
# - Resize using interpolation=InterpolationMode.NEAREST
# - Convert mask to long type with class indices (0, 1)


def mask_to_index(mask_tensor):
    # Input: Tensor (1, H, W) with values in [0, 127, 255]
    # Output: Tensor (H, W) with class indices [0=bg, 1=LA, 2=LV]
    mask = mask_tensor.squeeze(0)
    out = torch.zeros_like(mask, dtype=torch.long)
    out[mask == 127] = 1  # LA
    out[mask == 255] = 2  # LV
    return out

def build_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.Normalize(0.5, 0.5)
    ])

def build_mask_transform():
    """
    Returns a transform that takes a PIL.Image mask (values {0,127,255})
    and outputs a LongTensor of shape (H, W) with values {0,1,2}.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        # 1) Resize with NEAREST so we keep exact labels
        transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
        # 2)
        transforms.Lambda(lambda x: (x*2).long()) # type(torch.uint8)),  # map 0.5 -> 1 (LA), 1 -> 2 (LV)
    ])

train_path = "/home/suetin/Projects/UltrasoundCardiacReconstruction/HeartReconstruction/data/train/echoLA/training"
test_path = "/home/suetin/Projects/UltrasoundCardiacReconstruction/HeartReconstruction/data/train/echoLA/testing"

img_train_paths = get_image_filepaths(os.path.join(train_path, "images"), format='.jpg')
img_test_paths = get_image_filepaths(os.path.join(test_path, "images"), format='.jpg')

msk_train_paths = get_image_filepaths(os.path.join(train_path, "masks"), format='.png')
msk_test_paths = get_image_filepaths(os.path.join(test_path, "masks"), format='.png')

# 
train_dataset = LeftCamusDataset(img_train_paths, 
                                 msk_train_paths, 
                                 img_transforms=build_transforms(), 
                                 msk_transforms=build_mask_transform())
test_dataset = LeftCamusDataset(img_test_paths, 
                                msk_test_paths, 
                                img_transforms=build_transforms(), 
                                msk_transforms=build_mask_transform())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

batch = next(iter(test_loader))
print(torch.unique(batch['labels'], return_counts=True))

# Paths to your MAE config and checkpoint
mae_cfg_path = "models/mae_pretrained/config.json"
mae_ckpt_path = "models/mae_pretrained.pth"  # or .pth

# 1) Instantiate UNETR with MAE weights
unetr = TransUNet(
    mae_config_path=mae_cfg_path,
    mae_checkpoint_path=mae_ckpt_path,
    num_classes=3,
    freeze_encoder_blocks=0
)

model = UNETRModule(model=unetr, lr=cfg['lr'], loss_type=cfg['loss_type'])

trainer = pl.Trainer(
    max_epochs=cfg['epochs'],
    precision=16,        # mixed precision for speed
    accelerator="auto",
    gradient_clip_val=1.0,
    accumulate_grad_batches=5,
)
trainer.fit(model, train_loader, test_loader)

# Save the model
ckpt_path = "models/transunet_model.ckpt"
trainer.save_checkpoint(ckpt_path)  # UNETR.load_from_checkpoint(ckpt_path)

pth_path = "models/transunet_model.pth"
torch.save(model.model.state_dict(), pth_path)  # state_dict = torch.load(pth_path, map_location="cpu"); unetr.load_state_dict(state_dict)