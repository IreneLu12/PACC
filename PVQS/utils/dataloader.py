from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import numpy as np



class frame_loader(Dataset):
    def __init__(self,info_file,current_dir='./'):
        f = open(info_file,"r")
        self.images_data = []
        self.base_dir = current_dir
        for lines in f.readlines():
            self.images_data.append(lines.strip('\n'))
        self.transform = T.Compose([
            T.ToTensor(),    
            T.Normalize( mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self,index):
        images_dat = self.images_data[index]
        images_dir = images_dat.split('?')[0]
        images_label = int(images_dat.split('?')[1])
        img = Image.open(self.base_dir + images_dir)
        img = img.convert('RGB')
        img = self.transform(img)
        return img.float(),np.int64(images_label)