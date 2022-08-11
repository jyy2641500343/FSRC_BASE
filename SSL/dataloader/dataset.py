import os
import torch.utils.data as datautils
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageDataset(datautils.Dataset):
    def __init__(self, root_path, transform=None, cfg=None):
        super(ImageDataset, self).__init__()
        
        self.samples = []
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
            
