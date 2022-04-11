import os
from glob import glob

import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class ImageDataset(data.Dataset):
    def __init__(self, root, transform):
        super(ImageDataset, self).__init__()
        self.root = root
        print("Creating Dataset from", self.root)
        self.paths = glob(os.path.join(self.root, "**/*.*"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = None
        while img is None:
            # while loop to get around bad image read problems
            try:
                img = Image.open(str(path)).convert('RGB')
            except:
                # get a different one
                index = index + 10
                path = self.paths[(index) % len(self.paths)]

        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'ImageDataset'
