import random
import numpy as np
import torch
import sys
import albumentations as A
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
# sys.path.append('/content/drive/MyDrive/CNS186_Final_Project/Files_from_Git/')
import data
import networks
import dt_utils
import net_utils
from torch.utils.data import TensorDataset
import random
import numpy as np
import torch
import albumentations as A
import torch.nn as nn
from torch.utils.data import TensorDataset
from PIL import Image
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

class GenRecogClassifyData_AD():

    def __init__(self, image_paths=None, transform=None, device='cpu',show=False):
        """
        Args:
            image_paths: list of image file paths, we no longer use a pickle file
            transform: albumentations transform to apply to images
            device: torch device
        """
        self.image_paths = image_paths
        self.device = device
        self.datasize = len(image_paths) if image_paths else 0
        self.show=show
        
        # Load pretrained ResNet18 and remove classifier
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove classifier
        self.resnet.eval()
        self.resnet.to(device)
        
        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.RandomRotate90(),
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                A.RandomBrightnessContrast(p=0.5),
            ])
        else:
            self.transform = transform
            
        # ImageNet normalization for ResNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            
    def load_transform_and_embed_batch(self, indices, apply_transform=True):
        """Load batch of images, apply transformations, and get ResNet embeddings"""
        show=self.show
        embeddings = []
        for idx in indices:
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            img_np = np.array(img)
            
            if apply_transform and self.transform is not None:
                img = self.transform(image=img)['image']
                img_np = np.array(img)

            #visualization to debug
            if show==True:
              plt.imshow(img_np)
              plt.title(f"Index: {idx} | Transform: {apply_transform}")
              plt.axis('off')
              plt.show()
                
            # Convert to tensor and normalize for ResNet
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = self.normalize(img_tensor).to(self.device)
            
            # Get ResNet embedding (512-dim)
            with torch.no_grad():
                embedding = self.resnet(img_tensor.unsqueeze(0)).squeeze(-1).squeeze(-1)
            embeddings.append(embedding.squeeze())
        
        return torch.stack(embeddings)
    
    def __call__(self, T, R, P1=0.5, P2=0.5, batchSize=-1, multiRep=False, device=None):
        device = self.device
        
        # Handle R as scalar or list
        Rlist = [R] if np.isscalar(R) else R
        print("datasize:", self.datasize, "T:", T, "datasize//T:", self.datasize // T)
        # Determine batchSize
        squeezeFlag = False
        if batchSize is None:
            batchSize = 1
            squeezeFlag = True
        elif batchSize < 0:
          batchSize = int(self.datasize // T)  # ensure Python int
          batchSize = max(1, batchSize)        # optional safeguard

        total_samples = T * batchSize
        if total_samples<self.datasize:
            print(f'total samples is {total_samples} and the size of dataset is {self.datasize}!')
        random_indices = torch.randperm(self.datasize)[:total_samples].reshape(T, batchSize)
        
        # Initialize embeddings and labels
        x = torch.zeros(T, batchSize, 512, device=device)
        y = torch.zeros(T, batchSize, dtype=torch.bool, device=device)
        
        # Load initial batch (t=0)
        initial_indices = random_indices[0].tolist()
        x[0] = self.load_transform_and_embed_batch(initial_indices, apply_transform=False)
        
        # Main loop
        for t in range(1, T):
            R = Rlist[np.random.randint(0, len(Rlist))]
            
            # Determine repeat mask
            if t-R>=0:
              repeatMask = torch.rand(batchSize, device=device) > P1
              if not multiRep:
                  repeatMask = repeatMask * (~y[t-R])
            else:
              repeatMask = torch.zeros(batchSize, device=device, dtype=torch.bool)
            
            # Determine transform mask
            transformMask = torch.rand(batchSize, device=device) > P2
            
            # CASE 1: repeat + transform
            mask1 = repeatMask & transformMask
            if mask1.any():
                indices = random_indices[t-R, mask1].tolist()
                x[t, mask1] = self.load_transform_and_embed_batch(indices, apply_transform=True)
                y[t, mask1] = 1
            
            # CASE 2: repeat only (no transform)
            mask2 = repeatMask & (~transformMask)
            if mask2.any():
                indices = random_indices[t-R, mask2].tolist()
                x[t, mask2] = self.load_transform_and_embed_batch(indices, apply_transform=False)
                y[t, mask2] = 1
            
            # CASE 3: new image (neither repeat nor transform)
            mask3 = ~(repeatMask) & (~transformMask)
            if mask3.any():
                indices = random_indices[t, mask3].tolist()
                x[t, mask3] = self.load_transform_and_embed_batch(indices, apply_transform=False)
                y[t, mask3] = 0  # truly new image
            # CASE 4: new image (neither repeat BUT transform)
            mask3 = ~(repeatMask) & (transformMask)
            if mask3.any():
                indices = random_indices[t, mask3].tolist()
                x[t, mask3] = self.load_transform_and_embed_batch(indices, apply_transform=True)
                y[t, mask3] = 0  # truly new image
              
        
        # Format output
        y_out = y.unsqueeze(2).float()
        c = torch.zeros(T, batchSize, 1, device=device)
        y_out = torch.cat((y_out, c), dim=-1)
        
        data = TensorDataset(x, y_out)
        
        if squeezeFlag:
            data = TensorDataset(*data[:, 0, :])
        
        return data

class GenRecogClassifyData_AD_Batched():
    def __init__(self, image_paths=None, transform=None, device='cpu', show=False, max_img_cache=1024):
        self.image_paths = image_paths
        self.device = device
        self.datasize = len(image_paths) if image_paths else 0
        self.show = show
        self.max_img_cache = max_img_cache

        # Load pretrained ResNet18 and remove classifier
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval().to(device)
        self.resnet.requires_grad_(False)

        if transform is None:
            self.transform = A.Compose([
                A.RandomRotate90(),
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                A.RandomBrightnessContrast(p=0.5),
            ])
        else:
            self.transform = transform

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # caches
        self._img_cache = OrderedDict()      # idx -> np.uint8 HWC
        self._embed_cache = {}               # idx -> torch.Tensor(512) for apply_transform=False only

    def _load_image_np(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if idx in self._img_cache:
            img = self._img_cache.pop(idx)
            self._img_cache[idx] = img
            return img.copy()

        img = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        self._img_cache[idx] = img
        if len(self._img_cache) > self.max_img_cache:
            self._img_cache.popitem(last=False)
        return img.copy()

    def _embed_batch_from_indices(self, indices, apply_transform=True):
        if len(indices) == 0:
            return torch.empty((0, 512), device=self.device)

        img_tensors = []
        for idx in indices:
            img = self._load_image_np(int(idx))
            if apply_transform and self.transform is not None:
                img = self.transform(image=img)['image']

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img_tensor = self.normalize(img_tensor)
            img_tensors.append(img_tensor)

        batch = torch.stack(img_tensors, dim=0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            emb = self.resnet(batch).squeeze(-1).squeeze(-1)  # [B,512]
        return emb

    def load_transform_and_embed_batch(self, indices, apply_transform=True):
        """Load batch, optional transform, return [B,512] embeddings."""
        indices = [int(i) for i in indices]
        if apply_transform:
            return self._embed_batch_from_indices(indices, apply_transform=True)

        # cache only no-transform embeddings
        out = [None] * len(indices)
        miss_pos, miss_idx = [], []
        for pos, idx in enumerate(indices):
            cached = self._embed_cache.get(idx, None)
            if cached is None:
                miss_pos.append(pos)
                miss_idx.append(idx)
            else:
                out[pos] = cached

        if miss_idx:
            miss_emb = self._embed_batch_from_indices(miss_idx, apply_transform=False)
            for pos, idx, emb in zip(miss_pos, miss_idx, miss_emb):
                self._embed_cache[idx] = emb.detach()
                out[pos] = self._embed_cache[idx]

        return torch.stack(out, dim=0)

    def __call__(self, T, R, P1=0.5, P2=0.5, batchSize=-1, multiRep=False, device=None):
        device = self.device
        Rlist = [R] if np.isscalar(R) else R

        squeezeFlag = False
        if batchSize is None:
            batchSize = 1
            squeezeFlag = True
        elif batchSize < 0:
            batchSize = max(1, int(self.datasize // T))

        total_samples = T * batchSize
        random_indices = torch.randperm(self.datasize)[:total_samples].reshape(T, batchSize)  # CPU

        x = torch.zeros(T, batchSize, 512, device=device)
        y = torch.zeros(T, batchSize, dtype=torch.bool, device=device)

        x[0] = self.load_transform_and_embed_batch(random_indices[0].tolist(), apply_transform=False)

        for t in range(1, T):
            Rt = Rlist[np.random.randint(0, len(Rlist))]

            if t - Rt >= 0:
                repeatMask = (torch.rand(batchSize, device=device) > P1)
                if not multiRep:
                    repeatMask = repeatMask & (~y[t - Rt])
            else:
                repeatMask = torch.zeros(batchSize, dtype=torch.bool, device=device)

            transformMask = (torch.rand(batchSize, device=device) > P2)

            # one source index per position: repeat -> t-R, else -> t
            source_idx = torch.where(
                repeatMask.cpu(),
                random_indices[t - Rt],
                random_indices[t]
            )  # CPU [batchSize]

            pos_tf = torch.where(transformMask)[0]      # device
            pos_ntf = torch.where(~transformMask)[0]    # device

            if pos_tf.numel() > 0:
                idx_tf = source_idx[pos_tf.cpu()].tolist()
                x[t, pos_tf] = self.load_transform_and_embed_batch(idx_tf, apply_transform=True)

            if pos_ntf.numel() > 0:
                idx_ntf = source_idx[pos_ntf.cpu()].tolist()
                x[t, pos_ntf] = self.load_transform_and_embed_batch(idx_ntf, apply_transform=False)

            y[t] = repeatMask

        y_out = y.unsqueeze(2).float()
        c = torch.zeros(T, batchSize, 1, device=device)
        y_out = torch.cat((y_out, c), dim=-1)

        data = TensorDataset(x, y_out)
        if squeezeFlag:
            data = TensorDataset(*data[:, 0, :])
        return data

from networks import HebbFeatureLayer
# ROOT_DIR = "/content/drive/MyDrive/CNS186_Final_Project/Stimuli/newolddelay/"
ROOT_DIR = "/home/raaghav/hebbff/Images/Stimuli/newolddelay/"
def is_image(fname):
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))

# -------- Get random category --------
categories = [d for d in os.listdir(ROOT_DIR)
              if os.path.isdir(os.path.join(ROOT_DIR, d))]

category = random.choice(categories)
category_path = os.path.join(ROOT_DIR, category)

# -------- Get images for the chosen category --------
# Correctly list image files from the currently selected category_path
images = [os.path.join(category_path, f) for f in os.listdir(category_path) if is_image(f)]
print(f"Number of images in selected category: {len(images)}")


#save_weights_file = "results/HebbFeature_freezehebbff.pkl"
#logs = "results/tensor_board_logs"

# Parameters
Nx = 512  # Raw image feature dimension (from ResNet output)
d = 50    # Compressed feature dimension
Nh = 16   # Hidden Hebbian layer size
Ny = 1    # Output dimension (recognition task)

# Train parameters
Tmul = 10
Tmin = 10
P = 0.5 # Probability of repeat
noMultiRep = True
increment = 'plus1'
R0 = 1
Rf = float('inf') # No upper limit on R during training
batchSize = 1 # seems like the code does not support minibatch right now

generator = GenRecogClassifyData_AD_Batched(image_paths=images,show=True)
print(f"Generator image paths initialized with: {generator.image_paths}")
    
def generate_recog_data_batch(T,d,R,P,multiRep,batchSize,**kwargs):
    effective_batch = 1 if batchSize is None else batchSize
    x,y = x, y = generator(
    T=15,
    R=R,
    P1=P,
    P2=0.5,
    batchSize=batchSize,
    multiRep=multiRep
    ).tensors
    return TensorDataset(x, y[..., 0:1])


gen_data = generate_recog_data_batch(T=max(Tmin, 2*Tmul),
                                                   d=d,
                                                   R=2,
                                                   P=P,
                                                   softLabels=False,
                                                   interleave=True,
                                                   multiRep=(not noMultiRep),
                                                   batchSize=batchSize,
                                                   xDataVals='+-',
                                                   device='cpu')

print(gen_data)
print(gen_data.tensors[1])
