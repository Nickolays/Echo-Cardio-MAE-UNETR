import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class EchoPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, json_list, transform):
        self.records = json_list          # list of dicts loaded from JSON
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        if isinstance(rec, dict):
            img_path = rec["image_path"]
        else:
            img_path = rec
        # img = Image.open(rec["image_path"]).convert("L")    # if we use json
        
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)      # → Tensor shape (1, H, W)
        return {"pixel_values": img_t}    # Lightning will collate to batch
    

class EchoDataModule(LightningDataModule):
    def __init__(self, manifest, batch_size=16, transform=None):
        super().__init__()
        self.manifest = manifest
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.ds = EchoPretrainDataset(self.manifest, self.transform)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    

class LeftCamusDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, la_mask_paths, 
                 img_transforms=None, msk_transforms=None):
        self.images = image_paths
        self.la_masks = la_mask_paths
        self.img_transforms = img_transforms
        self.msk_transforms = msk_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1) Load RGB image
        img = Image.open(self.images[idx]).convert("RGB")
        if self.img_transforms:
            img = self.img_transforms(img)  # Tensor (3,224,224)

        # 2) Load and combine LV & LA into one PIL mask
        la = Image.open(self.la_masks[idx]).convert("L")
        lv_path = self.la_masks[idx].replace("echoLA", "echoLV")
        lv = Image.open(lv_path).convert("L")
        # create a 2D NumPy label (overwrite LA where both present)
        la_arr = np.array(la, dtype=np.uint8)
        lv_arr = np.array(lv, dtype=np.uint8)
        combined = np.zeros_like(la_arr)
        combined[lv_arr > 0] = 255
        combined[la_arr > 0] = 128
        mask_pil = Image.fromarray(combined)

        # 3) Transform mask to class‐index tensor
        if self.msk_transforms:
            mask = self.msk_transforms(mask_pil)  # Tensor (224,224), Long
        else:
            mask = torch.from_numpy(combined).long()

        return {"pixel_values": img, "labels": mask}
