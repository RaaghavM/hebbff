import math
import torch
import torch.nn as nn
from networks import HebbFeatureLayer
from data import GenRecogClassifyData
from plotting import plot_generalization, get_recog_positive_rates
from torch.utils.data import TensorDataset

save_weights_file = "results/HebbFeature_no_freeze.pkl"
logs = "results/tensor_board_logs_no_freeze"

# Parameters
Nx = 512  # Raw image feature dimension (from ResNet output)
d = 50    # Compressed feature dimension
Nh = 16   # Hidden Hebbian layer size
Ny = 1    # Output dimension (recognition task)

# Train parameters
Tmul = 20
Tmin = 200
P = 0.5 # Probability of repeat
noMultiRep = True
increment = 'plus1'
R0 = 2
Rf = float('inf') # No upper limit on R during training

# Initialize network
net = HebbFeatureLayer(init=[d, Nh, Ny], Nx=Nx)

# Everything will be trained
print("Trainable parameters:")
for name, param in net.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# Train
img_dataset = "publish/conv/BradyOliva2008_UniqueObjects_ResNet18.pkl"
images = torch.load(img_dataset)
dummyClasses = torch.zeros(images.shape[0],1)
sampleSpace = TensorDataset(images, dummyClasses)
generator = GenRecogClassifyData(sampleSpace=sampleSpace)
    
def generate_recog_data_batch(T,d,R,P,multiRep,batchSize,**kwargs):
    effective_batch = 1 if batchSize is None else batchSize
    x,y = generator(T, R, P, effective_batch, multiRep).tensors
    return TensorDataset(x, y[..., 0:1])

if type(net) == HebbFeatureLayer:
    assert images.shape[1] == Nx # sanity check

gen_data = lambda R: generate_recog_data_batch(T=max(Tmin, R*Tmul), 
                                                   d=d, 
                                                   R=R, 
                                                   P=P, 
                                                   softLabels=False,
                                                   interleave=True, 
                                                   multiRep=(not noMultiRep), 
                                                   batchSize=None,
                                                   xDataVals='+-',
                                                   device='cpu')
if increment.startswith('plus'):
    n = int(increment[4:])
    increment = lambda R: R+n
elif increment.startswith('times'):
    n = float(increment[5:])
    increment = lambda R: int(math.ceil(R*n)) 
    
net.fit('curriculum', gen_data, iters=float('inf'), itersToQuit=(2*10**6), batchSize=None, learningRate=1e-3,
        filename=save_weights_file, overwrite=False, folder=logs, R0=R0, Rf=Rf, increment=increment)

#plot generalization
testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data)
ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate)
ax.figure.save_figure("results/gen_plot_feature_and_hebbff.png")