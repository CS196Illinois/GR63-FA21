import torch
from torch import nn

class FastSlowModel(nn.Module):
    def __init__(self):
        super(FastSlowModel, self).__init__()
        self.slow = SlowPathway()
        self.fast = FastPathway()
        self.avgpool_slow = nn.AvgPool3d((4,7,7))
        self.avgpool_fast = nn.AvgPool3d((32,7,7))
        
        self.fc = nn.Linear(2048+256, 1)
        
    def forward(self, x):
        fast_out, lateral_connections = self.fast(x)
        fast_out = self.avgpool_fast(fast_out)
        fast_out = torch.flatten(fast_out, start_dim=1)
        
        
        slow_out = self.slow(x[:, :, ::8], lateral_connections) # b, c, t 
        slow_out = self.avgpool_slow(slow_out)
        slow_out = torch.flatten(slow_out, start_dim=1)
        
 
        
        x = torch.cat([fast_out, slow_out], dim=1)
        x = self.fc(x)
        return x
    
class FastPathway(nn.Module):
    def __init__(self):
        super(FastPathway, self).__init__()
        self.conv1 = nn.Conv3d(3, 8, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.pool1 = nn.MaxPool3d((1,3,3), stride=(1, 2, 2), padding=(0,1,1))
        self.pool2 = nn.MaxPool3d((1,2,2), stride=(1,2,2))
        self.conv_block_1 = nn.Sequential(
            BottleneckBlock(8, 8, resize=True, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(32, 8, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(32, 8, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)])
        )
        self.conv_block_2 = nn.Sequential(
            BottleneckBlock(32, 16, resize=True, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(64, 16, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(64, 16, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(64, 16, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)])
        )
        self.conv_block_3 = nn.Sequential(
            BottleneckBlock(64, 32, resize=True, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(128, 32, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(128, 32, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(128, 32, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(128, 32, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(128, 32, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)])
        )
        self.conv_block_4 = nn.Sequential(
            BottleneckBlock(128, 64, resize=True, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(256, 64, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(256, 64, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)])
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        lat1 = x
        x = self.conv_block_1(x)
        x = self.pool2(x)
        lat2 = x
        x = self.conv_block_2(x)
        x = self.pool2(x)
        lat3 = x
        x = self.conv_block_3(x)
        x = self.pool2(x)
        lat4 = x
        x = self.conv_block_4(x)
       # x = self.pool2(x)
        return x, [lat1, lat2, lat3, lat4]
    
class SlowPathway(nn.Module):
    def __init__(self):
        super(SlowPathway, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.pool1 = nn.MaxPool3d((1,3,3), stride=(1, 2, 2), padding=(0,1,1))
        self.pool2 = nn.MaxPool3d((1,2,2), stride=(1,2,2))
        
        self.lateral_conv1 = nn.Conv3d(8, 64, (8, 1, 1), stride=(8, 1, 1), padding=(0, 0, 0))
        self.lateral_conv2 = nn.Conv3d(32, 256, (8, 1, 1), stride=(8, 1, 1), padding=(0, 0, 0))
        self.lateral_conv3 = nn.Conv3d(64, 512, (8, 1, 1), stride=(8, 1, 1), padding=(0, 0, 0))
        self.lateral_conv4 = nn.Conv3d(128, 1024, (8, 1, 1), stride=(8, 1, 1), padding=(0, 0, 0))
        # torch.Size([1, 256, 4, 28, 28]) torch.Size([1, 32, 32, 28, 28])
        self.conv_block_1 = nn.Sequential(
            BottleneckBlock(64, 64, resize=True),
            BottleneckBlock(256, 64),
            BottleneckBlock(256, 64)
        )
        self.conv_block_2 = nn.Sequential(
            BottleneckBlock(256, 128, resize=True),
            BottleneckBlock(512, 128),
            BottleneckBlock(512, 128),
            BottleneckBlock(512, 128)
        )
        self.conv_block_3 = nn.Sequential(
            BottleneckBlock(512, 256, resize=True, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(1024, 256, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(1024, 256, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(1024, 256, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(1024, 256, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(1024, 256, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)])
        )
        self.conv_block_4 = nn.Sequential(
            BottleneckBlock(1024, 512, resize=True, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(2048, 512, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)]),
            BottleneckBlock(2048, 512, kernel_sizes=[(3,1,1), (1,3,3), (1,1,1)], padding_sizes=[(1,0,0), (0,1,1), (0,0,0)])
        )

    def forward(self, x, lateral_connections):
        #  [lat1, lat2, lat3, lat4]
        '''
        Shape: torch.Size([1, 8, 32, 56, 56])
        Shape: torch.Size([1, 32, 32, 28, 28])
        Shape: torch.Size([1, 64, 32, 14, 14])
        Shape: torch.Size([1, 128, 32, 7, 7])
        '''
        lat1, lat2, lat3, lat4 = lateral_connections
        x = self.conv1(x)
        x = self.pool1(x)
        
        x += self.lateral_conv1(lat1)
        x = self.conv_block_1(x)
        x = self.pool2(x)
      
        x += self.lateral_conv2(lat2)
        x = self.conv_block_2(x)
        x = self.pool2(x)

        x += self.lateral_conv3(lat3)
        x = self.conv_block_3(x)
        x = self.pool2(x)
 
        x += self.lateral_conv4(lat4)
        x = self.conv_block_4(x)
       # x = self.pool2(x)
        return x
    
    
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_sizes=[(1,1,1), (1,3,3), (1,1,1)], padding_sizes=[(0,0,0), (0,1,1), (0,0,0)], resize=False):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_sizes[0], padding=padding_sizes[0])
        self.bn1 = nn.BatchNorm3d(out_chan)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_sizes[1], padding=padding_sizes[1])
        self.bn2 = nn.BatchNorm3d(out_chan)
        
        self.conv3 = nn.Conv3d(out_chan, 4*out_chan, kernel_sizes[2], padding=padding_sizes[2])
        self.bn3 = nn.BatchNorm3d(4*out_chan)
        
        self.resize = resize
        if self.resize:
            self.resize_conv = nn.Conv3d(in_chan, 4*out_chan, (1,1,1))
        
    def forward(self, x):
        res = x 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.resize:
            res = self.resize_conv(res)
        
        x += res
        x = self.relu(x)
        
        return x
    
