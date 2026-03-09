import os
import random
from typing import Optional
import numpy as np
import torch
import albumentations as A
import torch.nn as nn
from torch.utils.data import TensorDataset
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from collections import OrderedDict
from data import build_precomputed_embedding_banks

# IMAGE_DIR = "/home/raaghav/hebbff/Images/OBJECTSALL"
# base_out_npy = "precomputed_embeddings/base_bank/brady_base.npy"
# aug_out_npy = "precomputed_embeddings/aug_bank/brady_aug.npy"
IMAGE_DIR = "/home/raaghav/hebbff/Images/Stimuli"
base_out_npy = "precomputed_embeddings/base_bank/rutis_base_layer3_no_pool.npy"
aug_out_npy = "precomputed_embeddings/aug_bank/rutis_aug_layer3_no_pool.npy"
layer = "layer3_no_pool"

# Clear existing files
if os.path.exists(base_out_npy):
    os.remove(base_out_npy)
if os.path.exists(aug_out_npy):
    os.remove(aug_out_npy)

os.makedirs(os.path.dirname(base_out_npy), exist_ok=True)
os.makedirs(os.path.dirname(aug_out_npy), exist_ok=True)

images_paths = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_paths.append(os.path.join(root, file))

print(f"Total images found: {len(images_paths)}")

# Build the embedding banks
num_augmentations_per_image = 1000
# Using batch size 1 because images are different size
build_precomputed_embedding_banks(images_paths, base_out_npy, aug_out_npy, num_augmentations_per_image, verbose=True, max_img_cache=3000, batch_size=1, device="cuda", layer=layer)

# Print checks
base_embeddings = np.load(base_out_npy, mmap_mode="r")
aug_embeddings = np.load(aug_out_npy, mmap_mode="r")
print(f"\nBase embeddings shape: {base_embeddings.shape}")
print(f"Augmented embeddings shape: {aug_embeddings.shape}")
print(f"Base embeddings dtype: {base_embeddings.dtype}")
print(f"Augmented embeddings dtype: {aug_embeddings.dtype}")
