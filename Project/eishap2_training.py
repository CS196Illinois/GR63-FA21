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

from models.eishap2_model import Net as Net
from dataloaders.eishap2_dl import CustomImageDataset as Dataset

wandb.init(project='group63-violent-action')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(42)


train_set = Dataset(train=True)
test_set = Dataset(train=False)


print(f"Finished Initializing Datasets of length:\nTrain: {len(train_set) }\n  Test: {len(test_set)}\n ")

batch_size = 6
workers = 4

train_dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=workers,
    persistent_workers=False,
    prefetch_factor=2
)

test_dataloader = DataLoader(
     test_set,
     batch_size=16,
     shuffle=True,
     pin_memory=True,
     num_workers=1,
     persistent_workers=False,
     prefetch_factor=1 
 )

torch.backends.cudnn.benchmark = True


model = Net()
model = nn.DataParallel(model)
model = model.cuda()

opt = optim.SGD(model.parameters(), lr = .003, momentum=.9)
bce_loss = torch.nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, 3000, eta_min=.0001)


batch_factor = 5
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
            
    torch.save(model.state_dict(), f"chkpts/eishap2_e{epoch}.pt")
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_loss = 0
        test_hits = 0
        test_n = 0
        for (inputs, targets) in tqdm(test_dataloader, desc="Testing"):
            inputs = inputs.cuda()
            targets = targets.cuda().float()
            
            # split into 3 
            
            outs_1 = model(inputs[:,:,::3])
            outs_2 = model(inputs[:,:,1::3])
            outs_3 = model(inputs[:,:,2::3])
            
            mean_pred = torch.mean(torch.stack([outs_1, outs_2, outs_3]), dim=0)
            mean_pred_r = torch.round(mean_pred)
            
            outs_1_r = torch.round(outs_1)
            outs_2_r = torch.round(outs_2)
            outs_3_r = torch.round(outs_3)
            
            outs_rounded = torch.round(outs)
 
            loss = bce_loss(mean_pred, targets)
    
            test_loss += loss/len(test_dataloader)
        
            test_hits += torch.sum(outs_rounded == targets)
            test_hits_alt += torch.sum(mean_pred_r == targets)
            
            test_n += len(targets)

        wandb.log({
            'test_loss': test_loss,
            'test_acc': test_hits/test_n,
            'test_acc_alt': test_hits_alt/test_n
        })
        
    torch.cuda.empty_cache()
    model.train()
