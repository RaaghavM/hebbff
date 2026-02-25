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
               
def generate_recog_data(T=2000, d=50, R=1, P=0.5, interleave=True, multiRep=False, xDataVals='+-', softLabels=False):
    """Generates "image recognition dataset" sequence of (x,y) tuples. 
    x[t] is a d-dimensional random binary vector, 
    y[t] is 1 if x[t] has appeared in the sequence x[0] ... x[t-1], and 0 otherwise
    
    if interleave==False, (e.g. R=3) ab.ab.. is not allowed, must have a..ab..b.c..c (dots are new samples)
    if multiRep==False a pattern will only be (intentionally) repeated once in the trial
    
    T: length of trial
    d: length of x
    R: repeat interval
    P: probability of repeat
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    data = []
    repeatFlag = False
    r=0 #countdown to repeat
    for t in range(T): 
        #decide if repeating
        R = Rlist[np.random.randint(0, len(Rlist))]
        if interleave:
            repeatFlag = np.random.rand()<P
        else:
            if r>0:
                repeatFlag = False
                r-=1
            else:
                repeatFlag = np.random.rand()<P 
                if repeatFlag:
                    r = R
                
        #generate datapoint
        if t>=R and repeatFlag and (multiRep or data[t-R][1].round()==0):
            x = data[t-R][0]
            y = 1
        else:
            if xDataVals == '+-': #TODO should really do this outside the loop...
                x = 2*np.round(np.random.rand(d))-1
            elif xDataVals.lower() == 'normal':
                x = np.sqrt(d)*np.random.randn(d)    
            elif xDataVals.lower().startswith('uniform'):
                upper, lower = parse_xDataVals_string(xDataVals)
                x = np.random.rand(d)*(upper-lower)+lower
            elif xDataVals == '01':
                x = np.round(np.random.rand(d))
            else:
                raise ValueError('Invalid value for "xDataVals" arg')           
            y = 0
            
        if softLabels:
            y*=(1-2*softLabels); y+=softLabels               
        data.append((x,np.array([y]))) 
        
    return data_to_tensor(data)
 
def generate_recog_data_batch(T=2000, batchSize=1, d=25, R=1, P=0.5, interleave=True, multiRep=False, softLabels=False, xDataVals='+-', device='cpu'):
    """Faster version of recognition data generation. Generates in batches and uses torch directly    
    Note: this is only faster when approx batchSize>4
    """  
    if np.isscalar(R):
        Rlist = [R]
    else:
        Rlist = R
    
    if xDataVals == '+-':
        x = 2*torch.rand(T,batchSize,d, device=device).round()-1 #faster than (torch.rand(T,B,d)-0.5).sign()
    elif xDataVals.lower() == 'normal':
        x = torch.randn(T,batchSize,d, device=device)    
    elif xDataVals.lower().startswith('uniform'):
        upper, lower = parse_xDataVals_string(xDataVals)
        x = torch.rand(T,batchSize,d, device=device)*(upper-lower)+lower
    elif xDataVals == '01':
        x = torch.rand(T,batchSize,d, device=device).round()
    else:
        raise ValueError('Invalid value for "xDataVals" arg')  
    
    y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
    
    for t in range(max(Rlist), T):
        R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
        
        if interleave:
            repeatMask = torch.rand(batchSize)>P
        else:
            raise NotImplementedError
        
        if not multiRep:
            repeatMask = repeatMask*(~y[t-R]) #this changes the effective P=n/m to P'=n/(n+m)
          
        x[t,repeatMask] = x[t-R,repeatMask]            
        y[t,repeatMask] = 1
        
    y = y.unsqueeze(2).float()
    if softLabels:
        y = y*0.98 + 0.01

    return TensorDataset(x, y)

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
          batchSize = max(1, batchSize)        

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

def build_precomputed_embedding_banks(
    image_paths,
    base_out_npy: str,
    aug_out_npy: Optional[str] = None,
    num_aug: int = 0,
    batch_size: int = 128,
    device: str = "cpu",
    transform=None,
    dtype=np.float16,
    verbose=False,
    max_img_cache: int = 1024,
):
    """
    One-time offline precompute.

    Writes:
      - base_out_npy: [N, 512] no-transform embeddings
      - aug_out_npy:  [N, num_aug, 512] transformed embeddings
    """
    assert len(image_paths) > 0, "image_paths must be non-empty"

    # Frozen ResNet18 backbone (512-d output)
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1]).eval().to(device)
    resnet.requires_grad_(False)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Default transform if none provided
    if transform is None:
        transform = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            A.RandomBrightnessContrast(p=0.5),
        ])

    N = len(image_paths)
    base_bank = np.lib.format.open_memmap(
        base_out_npy, mode="w+", dtype=dtype, shape=(N, 512)
    )

    aug_bank = None
    if num_aug > 0:
        if aug_out_npy is None:
            raise ValueError("aug_out_npy is required when num_aug > 0")
        aug_bank = np.lib.format.open_memmap(
            aug_out_npy, mode="w+", dtype=dtype, shape=(N, num_aug, 512)
        )

    # LRU cache: idx -> np.uint8 HWC image
    img_cache = OrderedDict()

    def _load_image_cached(idx: int) -> np.ndarray:
        idx = int(idx)
        if idx in img_cache:
            img = img_cache.pop(idx)
            img_cache[idx] = img
            return img.copy()

        img = np.array(Image.open(image_paths[idx]).convert("RGB"))
        img_cache[idx] = img
        if len(img_cache) > max_img_cache:
            img_cache.popitem(last=False)
        return img.copy()

    def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        return normalize(t)

    with torch.inference_mode():
        # Base bank
        if verbose:
            print("Computing base embeddings...")
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            imgs = []
            for i in range(s, e):
                img = _load_image_cached(i)
                imgs.append(_to_tensor(img))
            batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            emb = resnet(batch).squeeze(-1).squeeze(-1).cpu().numpy().astype(dtype, copy=False)
            base_bank[s:e] = emb

        if verbose:
            print("Computing augmented embeddings...")

        # Aug bank
        if aug_bank is not None and transform is not None:
            for k in range(num_aug):
                if verbose and k % 10 == 0:
                    print(f"  Augmentation {k+1}/{num_aug}")
                for s in range(0, N, batch_size):
                    e = min(s + batch_size, N)
                    imgs = []
                    for i in range(s, e):
                        img = _load_image_cached(i)
                        img = transform(image=img)["image"]
                        imgs.append(_to_tensor(img))
                    batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
                    emb = resnet(batch).squeeze(-1).squeeze(-1).cpu().numpy().astype(dtype, copy=False)
                    aug_bank[s:e, k, :] = emb

    # flush
    base_bank.flush()
    if aug_bank is not None:
        aug_bank.flush()

class GenRecogClassifyData_AD_Precompute():
    """
    Runtime generator using precomputed embedding banks.
    """
    def __init__(
        self,
        image_paths,
        base_bank_npy: str,
        aug_bank_npy: Optional[str] = None,
        device: str = "cpu",
    ):
        self.image_paths = image_paths
        self.datasize = len(image_paths) if image_paths else 0
        self.device = device

        if not os.path.exists(base_bank_npy):
            raise FileNotFoundError(base_bank_npy)

        self.base_bank = np.load(base_bank_npy, mmap_mode="r")  # [N,512]
        self.aug_bank = np.load(aug_bank_npy, mmap_mode="r") if aug_bank_npy else None

        if self.base_bank.shape[0] != self.datasize or self.base_bank.shape[1] != 512:
            raise ValueError("base bank shape must be [len(image_paths), 512]")
        if self.aug_bank is not None:
            if self.aug_bank.shape[0] != self.datasize or self.aug_bank.shape[2] != 512:
                raise ValueError("aug bank shape must be [len(image_paths), K, 512]")

    def load_embed_batch(self, indices, apply_transform=True):
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return torch.empty((0, 512), device=self.device)

        if apply_transform and self.aug_bank is not None:
            K = self.aug_bank.shape[1]
            k = np.random.randint(0, K, size=idx.shape[0])
            emb = self.aug_bank[idx, k, :]  # [B,512]
        else:
            emb = self.base_bank[idx, :]    # [B,512]

        emb = np.asarray(emb, dtype=np.float32)
        return torch.from_numpy(emb).to(self.device, non_blocking=True)

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

        x[0] = self.load_embed_batch(random_indices[0].tolist(), apply_transform=False)

        for t in range(1, T):
            Rt = Rlist[np.random.randint(0, len(Rlist))]

            if t - Rt >= 0:
                repeatMask = (torch.rand(batchSize, device=device) > P1)
                if not multiRep:
                    repeatMask = repeatMask & (~y[t - Rt])
            else:
                repeatMask = torch.zeros(batchSize, dtype=torch.bool, device=device)

            transformMask = (torch.rand(batchSize, device=device) > P2)

            source_idx = torch.where(
                repeatMask.cpu(),
                random_indices[t - Rt],
                random_indices[t],
            )  # CPU [B]

            pos_tf = torch.where(transformMask)[0]
            pos_ntf = torch.where(~transformMask)[0]

            if pos_tf.numel() > 0:
                idx_tf = source_idx[pos_tf.cpu()].tolist()
                x[t, pos_tf] = self.load_embed_batch(idx_tf, apply_transform=True)

            if pos_ntf.numel() > 0:
                idx_ntf = source_idx[pos_ntf.cpu()].tolist()
                x[t, pos_ntf] = self.load_embed_batch(idx_ntf, apply_transform=False)

            y[t] = repeatMask

        y_out = y.unsqueeze(2).float()
        c = torch.zeros(T, batchSize, 1, device=device)
        y_out = torch.cat((y_out, c), dim=-1)

        if squeezeFlag:
            return TensorDataset(x[:, 0, :], y_out[:, 0, :])
        return TensorDataset(x, y_out)

class GenRecogClassifyData():
    def __init__(self, d=None, teacher=None, datasize=int(1e4), sampleSpace=None, save=False, device='cpu'):
        if sampleSpace is None:
            x = torch.rand(datasize,d, device=device).round()*2-1
            if teacher is None:
                c = torch.randint(2,(datasize,1), device=device, dtype=torch.float)
            else:
                c = torch.empty(datasize,1, device=device, dtype=torch.float)
                for i,xi in enumerate(x):
                    c[i] = teacher(xi)
                c = (c-c.mean()+0.5).round()
            self.sampleSpace = TensorDataset(x,c)
            if save:
                if type(save) == str:
                    fname = save
                else:
                    fname = 'sampleSpace.pkl'
                torch.save(self.sampleSpace, fname)
        elif type(sampleSpace) == str:
            self.sampleSpace = torch.load(sampleSpace) 
        elif type(sampleSpace) == TensorDataset:
            self.sampleSpace = sampleSpace
            
        self.datasize, self.d = self.sampleSpace.tensors[0].shape            
        
        
    def __call__(self, T, R, P=0.5, batchSize=-1, multiRep=False, device='cpu'):
        if np.isscalar(R):
            Rlist = [R]
        else:
            Rlist = R
        
        squeezeFlag=False
        if batchSize is None:
            batchSize=1
            squeezeFlag=True
        elif batchSize < 0:
            batchSize = self.datasize/T
            
        randomSubsetIdx = torch.randperm(len(self.sampleSpace))[:T*batchSize]
        x,c = self.sampleSpace[randomSubsetIdx]
        x = x.reshape(T,batchSize,self.d)
        c = c.reshape(T,batchSize,1)
        y = torch.zeros(T,batchSize, dtype=torch.bool, device=device)
        for t in range(max(Rlist), T):    
            R = Rlist[np.random.randint(0, len(Rlist))] #choose repeat interval   
                   
            repeatMask = torch.rand(batchSize)>P   
            if not multiRep:
                repeatMask = repeatMask*(~y[t-R]) #this changes the effective P
              
            x[t,repeatMask] = x[t-R,repeatMask] 
            c[t,repeatMask] = c[t-R,repeatMask]            
            y[t,repeatMask] = 1
         
        y = y.unsqueeze(2).float()
        y = torch.cat((y,c), dim=-1)        
        data = TensorDataset(x,y)
        
        if squeezeFlag:
            data = TensorDataset(*data[:,0,:])
    
        return data
    

#%%############
### Helpers ###   
###############
def parse_xDataVals_string(xDataVals):
    assert xDataVals.lower().startswith('uniform')
    delimIdx = xDataVals.find('_')
    if delimIdx > 0:
        assert delimIdx==7
        lims = xDataVals[delimIdx+1:]
        lower = float(lims[:lims.find('_')])
        upper = float(lims[lims.find('_')+1:])
    else:
        lower = -1
        upper = 1
    return upper, lower


def prob_repeat_to_frac_novel(P, multiRep=False):
    if multiRep:
        return P
    n,m = P.as_integer_ratio()
    return 1 - float(n)/(m+n)
    

def check_recognition_data(data, R):
    """Make sure there are no spurious repeats"""
    if len(data) == 0:
        return False
    for i in range(len(data)):
        for j in range(0,i-1):
            if all(data[i][0] == data[j][0]):   
                if i-j != R:
                    print( 'bad R', i, j )
                    return False
                if not data[i][1]:
                    print( 'unmarked', i, j )
                    return False
    return True

             
def recog_chance(data):
    """Calculates expected performance if network simply guesses based on output statistics
    i.e. the number of zeroes in the data"""
    return 1-np.sum([xy[1] for xy in data], dtype=float)/len(data) 


def batch(generate_data, batchsize=1, batchDim=1, **dataKwargs):
    dataList = []
    for b in range(batchsize):
        dataList.append( generate_data(**dataKwargs) )
    x = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim)
    y = torch.cat([data.tensors[0].unsqueeze(batchDim) for data in dataList], dim=batchDim) 
    return TensorDataset(x,y)


def data_to_tensor(data, y_dtype=torch.float, device='cpu'):
    '''Convert from list of (x,y) tuples to TensorDataset'''
    x,y = zip(*data)
    return TensorDataset(torch.as_tensor(x, dtype=torch.float, device=device), 
                   torch.as_tensor(y, dtype=y_dtype, device=device))


