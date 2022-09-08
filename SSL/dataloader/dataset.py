import os
import torch.utils.data as datautils
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


THIS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(os.path.join(THIS_PATH, '..', '..'))
IMAGE_PATH = os.path.join(ROOT_PATH, 'data/miniImagenet/images')
SPLIT_PATH = os.path.join(ROOT_PATH, 'data/miniImagenet/split')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageDataset(datautils.Dataset):
    def __init__(self, root_path, transform=None, cfg=None):
        super(ImageDataset, self).__init__()
        
        self.samples = []
        root_path = os.path.abspath(root_path)
        self.label_list = []
        lb = -1
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=cfg['normalize_mean'],
                                                                 std=cfg['normalize_std']),
                                                ])
        for root_sub_path in os.listdir(root_path):
            lb = lb + 1
            for element in os.listdir(os.path.join(root_path, root_sub_path)):
                self.samples.append(os.path.join(root_path, root_sub_path, element))
                self.label_list.append(lb)
            
        self.loader = pil_loader
        self.transform = transform
        
        
        classes, class_to_idx = self._find_classes(root_path)
        self.targets = self._make_targets(class_to_idx=class_to_idx)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index], self.targets[index]

        img = self.loader(path)

        if self.transform:
            img = self.transform(img)
        image_0 = self.to_tensor(img)
        image_90 = self.to_tensor(TF.rotate(img, 90))
        image_180 = self.to_tensor(TF.rotate(img, 180))
        image_270 = self.to_tensor(TF.rotate(img, 270))

        all_images = torch.stack([image_0, image_90, image_180, image_270], 0)

        return all_images, target
            
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
    def _make_targets(self, class_to_idx=None):
        if class_to_idx:
            self.num_classes = len(class_to_idx)
            return np.array([class_to_idx[os.path.split(os.path.split(sample)[0])[1]] for sample in self.samples])
            
class MiniImageNet(datautils.Dataset):

    def __init__(self, setname, cfg):
        csv_path = os.path.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.cfg = cfg
        
        if cfg['model_type'] == 'ConvNet':
            # for ConvNet512 and Convnet64
            image_size = 84
            self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                                    np.array([0.229, 0.224, 0.225]))
                                                ])
            
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size)
            ])            
        else:
            # for Resnet12
            image_size = 84
                     
            self.to_tensor = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
                                                ])
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size)
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
                
        image_0 = self.to_tensor(image)
        image_90 = self.to_tensor(TF.rotate(image, 90))
        image_180 = self.to_tensor(TF.rotate(image, 180))
        image_270 = self.to_tensor(TF.rotate(image, 270))

        all_images = torch.stack([image_0, image_90, image_180, image_270], 0) # <4, 3, size, size>
        
        return all_images, label