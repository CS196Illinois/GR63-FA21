import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize

import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CustomResNeXtPreAct(nn.Module):
    def __init__(self, img_chan=3, num_paths=32):
        super(CustomResNeXtPreAct, self).__init__()
        self.conv1 = nn.Conv3d(img_chan, 128, 7, stride=(2,2,2))
        
        self.conv_block_1 = nn.Sequential(
                ResNextBlock(128, 64, num_paths, expansion=1),
                ResNextBlock(128, 64, num_paths, expansion=1),
                ResNextBlock(128, 64, num_paths, expansion=2)
        )
        self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        # 28
        self.conv_block_2 = nn.Sequential(
            ResNextBlock(256, 128, num_paths, expansion=1),
            ResNextBlock(256, 128, num_paths, expansion=1),
            ResNextBlock(256, 128, num_paths, expansion=1),
            ResNextBlock(256, 128, num_paths, expansion=2)
        )
        self.mp2 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))
        # 14
        self.conv_block_3 =  nn.Sequential(
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1)
        )
        self.mp3 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))
        # 7
        self.conv_block_4 = nn.Sequential(
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1),
            ResNextBlock(512, 256, num_paths, expansion=1)
        )
        self.mp4 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3), padding=(1,0,0))
        # 3
        self.conv_block_5 = nn.Sequential(
           ResNextBlock(512, 256, num_paths, expansion=1),
           ResNextBlock(512, 256, num_paths, expansion=1),
           ResNextBlock(512, 256, num_paths, expansion=1)
        )
       # 2
      #   self.mp5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.fc1 = nn.Sequential(
            nn.Linear(512*2*2*2 , 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(128, 1))   
        
    def forward(self,x):
        x = self.conv1(x)

        
        x = self.conv_block_1(x)
        x = self.mp1(x)
        x = self.conv_block_2(x)
        x = self.mp2(x)
        x = self.conv_block_3(x)
        x = self.mp3(x)
        x = self.conv_block_4(x)
        x = self.mp4(x)
        x = self.conv_block_5(x)
      #   x = self.mp5(x)
            
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
            
        
        return x
        

        
class ResNextBlock(nn.Module):
    def __init__(self, i_c, o_c, num_groups, expansion=1):
        super(ResNextBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(i_c)
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv3d(i_c, o_c, 1)
       
        self.bn2 = nn.BatchNorm3d(o_c)
        self.conv2 = nn.Conv3d(o_c, o_c, 3, padding=1, groups=num_groups)
        
        self.bn3 =  nn.BatchNorm3d(o_c)
        self.conv3 = nn.Conv3d(o_c, i_c, 1)
        
        self.conv4 = nn.Conv3d(i_c, i_c*expansion, 1)
        
        
    def forward(self,x):
        # i_c, t, h, w
        res = x 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x) 
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x) 
        
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x) 
        
        x += res 
        
        x = self.conv4(x)
        
        return x 


