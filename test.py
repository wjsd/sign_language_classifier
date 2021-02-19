"""

Sign language classifier

test.py

Testing script for sign language classifier.

TODO: make sure printed letter is correct

"""
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from datasets import ASLAlphabet
from networks import SLClassifier
from argparse import ArgumentParser
from GradCAM import GradCAM
import cv2

import matplotlib.pyplot as plt
import os

parser = ArgumentParser()
parser.add_argument("--filename",type=str,default="./trainlogs/run0/state_dicts/net_state_dict_epoch0.pth",help="(str) filepath to network state dictionary")
parser.add_argument("--batchsize",type=int,default=2,help="Batch size")
parser.add_argument("--ntestbatches",type=int,default=10,help="Number of test batches to evaluate accuracy on")
# TODO: cams by batch
parser.add_argument("-cbb","--cams_by_batch",action="store_true",help="If flagged, make image grid by batch rather than by letter")
parser.add_argument("--device",type=str,default="cuda",help="Device to compute on")
opts = parser.parse_args()

#
# Test network
#
dset = ASLAlphabet(type="val")
dl = DataLoader(dataset=dset,batch_size=opts.batchsize,shuffle=True)
print("[ dataloader created ]")

net = SLClassifier() # TODO: load architecture from logging directory
net.load_state_dict(torch.load(opts.filename))
net.to(opts.device)
net.eval()
print("[ state_dict loaded into network ]")

# test accuracy
dliter = iter(dl) # dataloader iterator
accs = torch.empty(opts.ntestbatches)
for i in range(opts.ntestbatches):
    letters,samples = next(dliter)
    scores = net(samples.to(opts.device))
    _,predictions = scores.to("cpu").max(dim=1)
    acc = (predictions==letters).sum() / float(scores.size(0))
    accs[i] = acc

print("accs: ", accs)
print("mean accuracy: ", float(sum(accs))/len(accs))

# test grad-cam
GC = GradCAM(net,device=opts.device)
samples.requires_grad = True

guidemods = []
for mod in net.features:
    if mod.__class__.__name__ == "ReLU":
        guidemods.append(mod)

cams = GC(samples,submodule=net.features[-2],guided=guidemods)
print("cams.shape: ",cams.shape)

cams = torch.nn.Upsample(scale_factor=2,mode="bilinear")(torch.from_numpy(cams[0]).detach())
print("cams.shape: ",cams.shape)

imlist = [cam for cam in cams] # convert first dim into list
dirs = os.listdir("../asl_alphabet/val")
dirs.sort()

print("letter: ",dirs[letters[0]])
print("prediction: ",dirs[predictions[0]])

grid = make_grid(imlist,nrow=6,normalize=True).permute(1,2,0)

fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1)
ax1.imshow(samples[0].detach().permute(1,2,0))
ax2.imshow(grid)

plt.show()
