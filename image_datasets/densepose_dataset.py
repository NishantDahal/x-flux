# densepose_dataset.py

import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def c_crop(image):
    """
    Center-crop the image to a square, using the smaller dimension.
    """
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class DensePoseImageDataset(Dataset):
    """
    Loads a triplet of:
      - Original image (RGB)
      - DensePose map (RGB or 3-channel)
      - Caption text from JSON
    Expects files like:
      images_dir/1.png
      maps_dir/1_segm.png
      contents_dir/1.json  (with {"caption": "..."} inside)
    """
    def __init__(self, 
                 images_dir,
                 maps_dir,
                 contents_dir,
                 img_size=512):
        """
        Args:
            images_dir   : path to folder with original images
            maps_dir     : path to folder with DensePose segm maps
            contents_dir : path to folder with JSON files (each has {"caption": "..."})
            img_size     : final image size (width=height=img_size)
        """
        self.images_dir = images_dir
        self.maps_dir = maps_dir
        self.contents_dir = contents_dir
        self.img_size = img_size

        # Gather all image filenames (e.g., 1.png, 2.png, etc.)
        self.images = [
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith('.png') or f.lower().endswith('.jpg')
        ]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            # 1) Original image
            image_filename = self.images[idx]
            image_path = os.path.join(self.images_dir, image_filename)
            img = Image.open(image_path).convert('RGB')  # ensure RGB
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))

            # 2) DensePose map
            base_name = os.path.splitext(image_filename)[0]  # e.g., "1" from "1.png"
            dp_map_path = os.path.join(self.maps_dir, f"{base_name}_segm.png")
            dp_img = Image.open(dp_map_path).convert('RGB')  # also ensure 3 channels
            dp_img = c_crop(dp_img)
            dp_img = dp_img.resize((self.img_size, self.img_size))

            # 3) Caption from JSON
            json_path = os.path.join(self.contents_dir, f"{base_name}.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            prompt = data.get('caption', '')

            # Convert to tensors in [-1, 1] range
            img = torch.from_numpy((np.array(img, dtype=np.float32) / 127.5) - 1.0)
            img = img.permute(2, 0, 1)  # HWC -> CHW

            dp_img = torch.from_numpy((np.array(dp_img, dtype=np.float32) / 127.5) - 1.0)
            dp_img = dp_img.permute(2, 0, 1)

            return img, dp_img, prompt

        except Exception as e:
            # If there's an error (missing file, bad image, etc.), fallback to a random item
            print(f"[DensePoseImageDataset] Error loading idx {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(
    train_batch_size,
    num_workers,
    images_dir,
    maps_dir,
    contents_dir,
    img_size=512,
):
    """
    Returns a PyTorch DataLoader for the DensePoseImageDataset.
    
    Args:
        train_batch_size : number of samples per batch
        num_workers      : number of subprocesses for data loading
        images_dir       : path to the folder with original images
        maps_dir         : path to the folder with DensePose segm maps
        contents_dir     : path to the folder with JSON captions
        img_size         : final size for both original image & segm map
    """
    dataset = DensePoseImageDataset(
        images_dir=images_dir,
        maps_dir=maps_dir,
        contents_dir=contents_dir,
        img_size=img_size
    )
    return DataLoader(dataset, 
                      batch_size=train_batch_size, 
                      num_workers=num_workers, 
                      shuffle=True)
