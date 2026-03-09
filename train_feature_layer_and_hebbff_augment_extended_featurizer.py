import math
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from networks import HebbExtendedFeatureLayer, HebbFeatureLayer
from data import GenRecogClassifyData_AD_Precompute
from plotting import plot_generalization, get_recog_positive_rates
from torch.utils.data import TensorDataset
import os
from pathlib import Path

# save_weights_file = "results/HebbFeature_no_freeze_augment_brady.pkl"
# logs = "results/tensor_board_logs_no_freeze_augment_brady"
# IMAGE_DIR = "/home/raaghav/hebbff/Images/OBJECTSALL"
# save_fig_file = "results/gen_plot_feature_and_hebbff_augment_brady.png"
# base_bank = "precomputed_embeddings/base_bank/brady_base.npy"
# aug_bank = "precomputed_embeddings/aug_bank/brady_aug.npy"
save_weights_file = "results/HebbFeature_no_freeze_augment_extended_rutis.pkl"
logs = "results/tensor_board_logs_no_freeze_augment_extended_rutis"
IMAGE_DIR = "/home/raaghav/hebbff/Images/Stimuli"
save_fig_file = "results/gen_plot_feature_and_hebbff_augment_extended_rutis.png"
base_bank = "precomputed_embeddings/base_bank/rutis_base_layer3_no_pool.npy"
aug_bank = "precomputed_embeddings/aug_bank/rutis_aug_layer3_no_pool.npy"

# Parameters
Nx = 512   # Raw image feature dimension (from ResNet output)
d = 128    # Compressed feature dimension
Nh = 32   # Hidden Hebbian layer size
Ny = 1    # Output dimension (recognition task)

device = "cuda" if torch.cuda.is_available() else "cpu"

input_emb_size = 256 * 14 * 14 # From resnet layer 3

# Train parameters
# pretrained = "/home/raaghav/hebbff/results/tensor_board_logs_no_freeze_augment_rutis/results/HebbFeature_no_freeze_augment_rutis_(4).pkl"
Tmul = 20
Tmin = 200
P = 0.5 # Probability of repeat
P2 = 0 # Always do augmentation
noMultiRep = True
increment = 'plus1'
R0 = 2
Rf = float('inf') # No upper limit on R during training
# itersToQuit = 2*10**6
itersToQuit = 20000
accuracyStopThres = 4.9 # Stop training if average accuracy over last 5 epochs exceeds this threshold
cache_batch_size = 64
train_minibatch_size = 16

# Initialize network
full_model = resnet18(weights=ResNet18_Weights.DEFAULT)
full_model.eval()

class ResNet18AfterLayer3(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):
        if x.dim() == 1:
            x = x.reshape(1, 256, 14, 14)
        elif x.dim() == 2:
            x = x.reshape(x.shape[0], 256, 14, 14)
        else:
            raise ValueError(f"Expected flattened layer3 embedding with rank 1 or 2, got rank {x.dim()}")
        x = self.layer4(x)                 # [B, 512, 7, 7]
        x = self.avgpool(x)               # [B, 512, 1, 1]
        x = torch.flatten(x, 1)           # [B, 512]
        return x

embedding_model = ResNet18AfterLayer3(full_model)
net = HebbExtendedFeatureLayer(init=[d, Nh, Ny], Nx=Nx, embedding_model=embedding_model)
net = net.to(device)
# net.load(pretrained)

# Everything will be trained
print("Trainable parameters:")
print(f"Using device: {device}")
for name, param in net.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# Train
# Recursively get all images from IMAGE_DIR and subdirectories
images_paths = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_paths.append(os.path.join(root, file))

print(f"Total images found: {len(images_paths)}")

# No-reuse constraint in precomputed generator: need T <= number of images.
# Since T=max(Tmin, R*Tmul), cap Rf accordingly to avoid mid-training failures.
max_r_no_reuse = max(1, len(images_paths) // Tmul)
if Rf > max_r_no_reuse:
    print(f"Capping Rf from {Rf} to {max_r_no_reuse} to avoid novel-image reuse.")
    Rf = max_r_no_reuse

generator = GenRecogClassifyData_AD_Precompute(image_paths=images_paths, device=device, base_bank_npy=base_bank, aug_bank_npy=aug_bank, embedding_size=input_emb_size)
    
def generate_recog_data_batch(T,d,R,P,P2,multiRep,batchSize,**kwargs):
    x,y = generator(T, R, P, P2, batchSize, multiRep).tensors
    return TensorDataset(x, y[..., 0:1])

gen_data = lambda R: generate_recog_data_batch(T=max(Tmin, R*Tmul), 
                                                   d=d, 
                                                   R=R, 
                                                   P=P, 
                                                   P2=P2,
                                                   softLabels=False,
                                                   interleave=True, 
                                                   multiRep=(not noMultiRep), 
                                                   batchSize=cache_batch_size,
                                                   xDataVals='+-')
if increment.startswith('plus'):
    n = int(increment[4:])
    increment = lambda R: R+n
elif increment.startswith('times'):
    n = float(increment[5:])
    increment = lambda R: int(math.ceil(R*n)) 
    
net.fit('curriculum', gen_data, iters=float('inf'), itersToQuit=itersToQuit, batchSize=train_minibatch_size, learningRate=1e-3,
        filename=save_weights_file, overwrite=False, folder=logs, R0=R0, Rf=Rf, increment=increment, accuracyStopThres=accuracyStopThres)

#plot generalization
gen_data = lambda R: generate_recog_data_batch(T=max(Tmin, R*Tmul), 
                                                   d=d, 
                                                   R=R, 
                                                   P=P, 
                                                   P2=P2,
                                                   softLabels=False,
                                                   interleave=True, 
                                                   multiRep=(not noMultiRep), 
                                                   batchSize=None,
                                                   xDataVals='+-')
testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data)
fig, ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate)
fig.savefig(save_fig_file, dpi=200, bbox_inches="tight")