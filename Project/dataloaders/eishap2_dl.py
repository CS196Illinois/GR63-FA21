import glob
import torch
from torchvision import transforms
import h5py 
from torch.utils.data import Dataset 


class CustomImageDataset(Dataset):
    def __init__(self, train=True):
        self.root_path = r'/home/shared/action/Data/RWF-2000-h5/*'
        self.paths = []
        #self.window_size = window_size
        folders = glob.glob(self.root_path)

        for folder in folders: # train and val
            print(folder)
            if train == ("train" in folder):
                print("Loading from:", folder)
                sub_folders = glob.glob(folder + "/*")
                for sub_folder in sub_folders:
                    files = glob.glob(sub_folder + "/*")
                    for file in files: 
                        self.paths.append(file)
        self.data = []
        for path in self.paths:
            self.data.append(path)
        self.paths = self.paths
        
        self.transforms = torch.nn.Sequential(
             transforms.RandomHorizontalFlip(),
             transforms.RandomResizedCrop((224, 224), scale=[.4, 1.], ratio=(.75, 1.33)),
           #  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    )
        self.val_transforms = torch.nn.Sequential(
                transforms.RandomResizedCrop((224,224), scale=[1., 1.], ratio=(1.,1.))
                )
        self.train=train

    def __len__(self):
        return len(self.paths) +  2 *len(self.paths) * int(self.train)

    def __getitem__(self, idx):
        
        path_idx = idx // 3 
        offset = idx%3
        

        path = self.paths[path_idx]
        
        hf = h5py.File(path, 'r')
        if self.train:
            data_slice = hf['vid_data'][:, offset::3]
        else:
            data_slice = hf['vid_data'][:, :]
       #  print(data_slice.shape)
        frame_data = torch.tensor(data_slice)[[2,1,0], :, :].float() # BGR to RGB, HWC to CHW
        label = not 'NonFight' in path # label == True == 1. 1 -> There is a fight. 
        frame_data /= 255
        
        if self.train:
            frame_data = self.transforms(frame_data)
        else:
            frame_data = self.val_transforms(frame_data)
        return frame_data, torch.tensor(label).float().unsqueeze(0) #  label?? # image, label
