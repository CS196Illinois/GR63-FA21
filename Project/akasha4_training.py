import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import os
import torch.nn.functional as F
import h5py
import glob
import torchvision
import wandb 
import numpy as np 
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from models.akasha4_model import ViViT as Net
from dataloaders.akasha4_dl import CustomImageDataset as Dataset

wandb.init(project='group63-violent-action')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(42)


FRAME_WINDOW = 20
train_set = Dataset(FRAME_WINDOW, train=True)
test_set = Dataset(FRAME_WINDOW, train=False)


print(f"Finished Initializing Datasets of length:\nTrain: {len(train_set) }\n  Test: {len(test_set)}\n ")

batch_size = 32
workers = 12

train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=workers,
    persistent_workers=False,
    prefetch_factor=1
)

test_dataloader = DataLoader(
     test_set,
     batch_size=16,
     shuffle=False,
     pin_memory=True,
     num_workers=8,
     persistent_workers=False,
     prefetch_factor=1 
 )

torch.backends.cudnn.benchmark = True

model = Net(224, 16, 1, 20, dim=128, patch_d = 4, depth_s=6, depth_t=12, heads=8, dim_head=64, dropout=.2, emb_dropout=.2)
model = nn.DataParallel(model)
model = model.cuda()


opt = optim.SGD(model.parameters(), lr = .03, momentum=.9)
bce_loss = torch.nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, 2000, eta_min=.0001)


batch_factor = 2
for epoch in range(100):
    opt.zero_grad()
    i = 0
    loss_acc = 0 
    for (inputs, targets) in tqdm(train_dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda().float()
        
        # B, C, T, H, W
        with torch.cuda.amp.autocast():
            
            outs = model(inputs)
            
            loss = bce_loss(outs, targets)

        loss /= batch_factor
        loss.backward() 

        loss_acc += loss.item()
            # print(outs, targets) 
        i+=1
        if i%batch_factor == batch_factor-1:
            opt.step()
            opt.zero_grad()
            wandb.log({"train_loss": loss_acc, 'lr': scheduler.get_last_lr()[0]})
            loss_acc = 0 
            scheduler.step()
            
    torch.save(model.state_dict(), f"chkpts/akasha4_e{epoch}.pt")
    model.eval()
    torch.cuda.empty_cache()
    if epoch % 6 != 5:
        model.train()
        continue
    with torch.no_grad():
        test_loss = 0
        test_hits = 0
        test_n = 0
        for (inputs, targets) in tqdm(test_dataloader, desc="Testing"):
            inputs = inputs.cuda()
            targets = targets.cuda().float()
            outs = model(inputs)
            outs = torch.sigmoid(outs)
            
            outs_rounded = torch.round(outs)
 
            loss = bce_loss(outs, targets)
            test_loss += loss/len(test_dataloader)
            test_hits += torch.sum(outs_rounded == targets)
            test_n += len(targets)

        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_hits/test_n
        })
        
    torch.cuda.empty_cache()
    model.train()
