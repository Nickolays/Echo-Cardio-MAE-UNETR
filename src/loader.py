import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os, cv2
from natsort import natsorted



class ExampleCRAFTDataset(Dataset):
    def __init__(self, main_path, feature_extractor, is_train=True):
       
        self.feature_extractor = feature_extractor

        sub_path = "training" if is_train else "testing"

        assert os.path.exists(main_path), "Root directory does not exist"
        assert os.path.exists(os.path.join(main_path, sub_path)), f"{sub_path} directory does not exist in root directory"
        assert os.path.exists(os.path.join(main_path, sub_path, "images")), f"images directory does not exist in {sub_path} directory"

        self.img_dir = os.path.join(main_path, sub_path, "images")
        # print(self.img_dir)
        self.ann_dir = os.path.join(main_path, sub_path, "masks")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        images = natsorted(image_file_names)
        self.images = [os.path.join(self.img_dir, img_name) for img_name in images]

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
            
        annotations = natsorted(annotation_file_names)
        self.annotations = [os.path.join(self.ann_dir, img_name) for img_name in annotations]

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"
        assert len(self.images) > 0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """ We already have 512 size  
        
         The model expected in output 3D tensor, if we have 4 dimmension input image. In this function 
        this dosn't work
        """
        image = Image.open(self.images[idx]).convert("RGB")
        mask = cv2.imread(self.annotations[idx], 0)  # Image.open(self.annotations[idx])
        # mask = np.expand_dims(mask, axis=0)    # (1, 512, 512)

        encoding = self.feature_extractor(
            image,
            size=256,
            do_resize=True,
            do_normalize=True,
            return_tensors="pt"
        )
        pixel_values = encoding['pixel_values'].squeeze(0)  # (3, 512, 512)


        mask_la, mask_lv = np.where(mask == 127, 1., 0.), np.where(mask == 255, 1., 0.)
        mask_la = np.array(mask_la, dtype=np.float32)
        mask_lv = np.array(mask_lv, dtype=np.float32)

        mask_2ch = np.stack([mask_lv, mask_la], axis=0)

        labels = torch.from_numpy(np.clip(mask_2ch, 0, 1))  # shape (2, 512, 512)

        labels = labels.argmax(0)
        # return {
        #     "pixel_values": pixel_values,  # (3, 512, 512)
        #     "labels": labels               # (2, 512, 512)
        # }
        return {
                "pixel_values": pixel_values,  # (3, 512, 512)
                "labels": labels               # (1, 512, 512)
            }