import torch.nn as nn
import torch.nn.functional as F
import torch

class petchoo2_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = nn.MaxPool3d(2)
        self.pool_last = nn.MaxPool3d(2,padding=(1,0,0))
        self.pool_wo_t =  nn.MaxPool3d((1,2,2))
        
        
        self.conv1 = self.three_conv(3, 32) # nn.Conv3d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 =  self.three_conv(32, 64) #nn.Conv3d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 =  self.three_conv(64, 128) #nn.Conv3d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 =  self.three_conv(128, 128) #nn.Conv3d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.conv5 =  self.three_conv(128, 256) #nn.Conv3d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.conv6 =  self.three_conv(256, 256) #nn.Conv3d(128, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm3d(256)
        self.conv7 = self.three_conv(256, 256) # nn.Conv3d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(256)
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6*256, 128)  # 2x2 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1, bias=False)

    def forward(self, x):

        #24x256x256
        # print(x.shape)
        x = self.conv1(x)
        x = self.pool_wo_t(self.bn1(F.relu(x))) #24x128x128
        x = self.pool_wo_t(self.bn2(F.relu(self.conv2(x)))) #24x64x64
        x = self.pool_wo_t(self.bn3(F.relu(self.conv3(x)))) #24x32x32
        x = self.pool_wo_t(self.bn4(F.relu(self.conv4(x)))) #12x16x16
        x = self.pool_wo_t(self.bn5(F.relu(self.conv5(x)))) #6x8x8
        x = self.pool(self.bn6(F.relu(self.conv6(x)))) #6x4x4
        x = self.pool(self.bn7(F.relu(self.conv7(x)))) #6x2x2
        
        # print("SHP:", x.shape)
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        x = torch.sigmoid(x)
        return x
    def three_conv(self, input_channel, output_channel):
        return nn.Sequential(
            nn.Conv3d(input_channel, output_channel, 3, padding=1),
            nn.Conv3d(output_channel, output_channel, 3, padding=1),
           # nn.Conv3d(output_channel, output_channel, 3, padding=1),
            nn.Conv3d(output_channel, output_channel, 3, padding=1, bias=False)
        )
        
# test on a existign cctv camera
