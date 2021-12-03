import glob
import torch
from torchvision import transforms
import h5py 
from torch.utils.data import Dataset 

class CustomImageDataset(Dataset):
    def __init__(self, window_size, train=True):
        self.n_frames_per_video = 150-window_size+1
        self.root_path = r'/home/shared/action/Data/RWF-2000-h5/*'
        self.paths = []
        self.window_size = window_size
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
             transforms.RandomResizedCrop((224, 224), scale=[.5, 1.], ratio=(.75, 1.33)),
        )
        self.val_transforms = transforms.RandomResizedCrop((224,224), scale=[1., 1.], ratio=(1., 1.))
        self.train = train

    def __len__(self):
        return len(self.paths)*self.n_frames_per_video

    def __getitem__(self, idx):
        
        path_idx = idx // self.n_frames_per_video
        frame_idx = idx % self.n_frames_per_video
        
        
        path = self.paths[path_idx]
        
        hf = h5py.File(path, 'r')
        data_slice = hf['vid_data'][:, frame_idx:frame_idx+self.window_size:2]
       #  print(data_slice.shape)
        frame_data = torch.tensor(data_slice)[[2,1,0], :, :].float() # BGR to RGB, HWC to CHW
        label = not 'NonFight' in path # label == True == 1. 1 -> There is a fight. 
        frame_data /= 255
        if self.train:
            frame_data = self.transforms(frame_data)
        else:
            frame_data = self.val_transforms(frame_data)

        return frame_data, torch.tensor(label).float().unsqueeze(0) #  label?? # image, label