import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 
        self.conv_block1 = self.conv_block((5,3,3), 3, 32, nn.GELU) # 
        self.conv_block2 = self.conv_block((5,3,3), 32, 32, nn.GELU)
        self.conv_block3 = self.conv_block((5,3,3), 32, 32, nn.GELU)
        self.conv_block4 = self.conv_block((5,3,3), 32, 64, nn.GELU)
        self.conv_block5 = self.conv_block((5,3,3), 64, 64, nn.GELU)
        self.conv_block6 = self.conv_block((5,3,3), 64, 96, nn.GELU)
        
        self.dropout = nn.Dropout(p=.2)
        self.final_maxpool = nn.MaxPool3d((2,3,3))
        self.fc1 = nn.Linear(96*25, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16,1)



    def forward(self, x):
        # x = 3, 50, 224, 224
        x = self.conv_block1(x) # 50, 112, 112
        x = self.conv_block2(x) # 50, 56, 56
        x = self.conv_block3(x) # 50, 28, 28
        x = self.conv_block4(x) # 50, 14, 14
        x = self.conv_block5(x) # 50, 6, 6
        x = self.conv_block6(x) # 50, 3, 3 * 96
        x = self.final_maxpool(x)
      #  print(x.shape)
        x = torch.flatten(x, start_dim=1)
       # print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        return x
    
    def conv_block(self, kernel_size, input_channels, output_channels, activation):
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size, padding=(2,1,1)),
            nn.Conv3d(output_channels, output_channels, kernel_size, padding=(2,1,1)),
            activation(),
            nn.BatchNorm3d(output_channels),
            nn.Conv3d(output_channels, output_channels, kernel_size, padding=(2,1,1)),
            nn.Conv3d(output_channels, output_channels, kernel_size, padding=(2,1,1)),
            activation(),
            nn.BatchNorm3d(output_channels),
            nn.Conv3d(output_channels, output_channels, kernel_size, padding=(2,1,1)),
            nn.Conv3d(output_channels, output_channels, kernel_size, padding=(2,1,1)),
            activation(),
            nn.BatchNorm3d(output_channels),
            nn.Dropout3d(p=0.2),
            nn.AvgPool3d((1,2,2))
        )
    
if __name__ == "__main__":
    model = Net().cuda()
    print(model)
    print("Total Params:", sum(p.numel() for p in model.parameters()))
    
    x = torch.rand(1,3,50,224,224).cuda()
    output = model(x)
    print(x.shape)
    
    
