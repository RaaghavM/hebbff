import math
import torch
import torch.nn as nn
from networks import HebbFeatureLayer
from data import GenRecogClassifyData_AD_Precompute
from plotting import plot_generalization, get_recog_positive_rates
from torch.utils.data import TensorDataset
import os
from pathlib import Path

save_weights_file = "results/HebbFeature_no_freeze_augment_brady.pkl"
logs = "results/tensor_board_logs_no_freeze_augment_brady"
IMAGE_DIR = "/home/raaghav/hebbff/Images/OBJECTSALL"
save_fig_file = "results/gen_plot_feature_and_hebbff_augment_brady.png"
base_bank = "precomputed_embeddings/base_bank/brady_base.npy"
aug_bank = "precomputed_embeddings/aug_bank/brady_aug.npy"
# save_weights_file = "results/HebbFeature_no_freeze_augment_rutis.pkl"
# logs = "results/tensor_board_logs_no_freeze_augment_rutis"
# IMAGE_DIR = "/home/raaghav/hebbff/Images/Stimuli"
# save_fig_file = "results/gen_plot_feature_and_hebbff_augment_rutis.png"

# Parameters
Nx = 512  # Raw image feature dimension (from ResNet output)
d = 128    # Compressed feature dimension
Nh = 32   # Hidden Hebbian layer size
Ny = 1    # Output dimension (recognition task)

# Train parameters
Tmul = 20
Tmin = 200
P = 0.5 # Probability of repeat
P2 = 0 # Always do augmentation
noMultiRep = True
increment = 'plus1'
R0 = 2
Rf = float('inf') # No upper limit on R during training
# itersToQuit = 2*10**6
itersToQuit = 100000
accuracyStopThres = 4.9 # Stop training if average accuracy over last 5 epochs exceeds this threshold

# Initialize network
net = HebbFeatureLayer(init=[d, Nh, Ny], Nx=Nx)

# Everything will be trained
print("Trainable parameters:")
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

generator = GenRecogClassifyData_AD_Precompute(image_paths=images_paths, device="cpu", base_bank_npy=base_bank, aug_bank_npy=aug_bank)
    
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
                                                   batchSize=1,
                                                   xDataVals='+-')
if increment.startswith('plus'):
    n = int(increment[4:])
    increment = lambda R: R+n
elif increment.startswith('times'):
    n = float(increment[5:])
    increment = lambda R: int(math.ceil(R*n)) 
    
net.fit('curriculum', gen_data, iters=float('inf'), itersToQuit=itersToQuit, batchSize=None, learningRate=1e-3,
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