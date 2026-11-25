import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, class_balance=False, num_classes=15505):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        self.num_classes=num_classes
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data=[[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y=self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_classes)]
        # Labels sorted ascending by count (rarest = 0); LTIC's masks rely on it (plan §3.1, R1).
        # kind='stable' so tied classes get the same index on every machine.
        sorted_classes=np.argsort(self.cls_num_list, kind='stable')
        self.class_map=[0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           sample_class=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[sample_class])
           path = self.img_path[index]
           label = self.class_map[sample_class]
        else:
           path = self.img_path[index]
           label = self.class_map[self.labels[index]]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class LT_Dataset_Test(Dataset):
    def __init__(self, root, txt, transform=None, class_map=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_map=class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.class_map[self.labels[index]]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label 

class herbariumDataset(object):
    num_classes = 15505
    train_file = "train_hbm.txt"
    val_file = "val_hbm.txt"
    mean = [0.466, 0.471, 0.380]
    std = [0.195, 0.194, 0.192]

    def __init__(self, batch_size=32, root=None, class_balance=False, num_works=12,
                 train_txt=None, val_txt=None, num_classes=None, mean=None, std=None):
        if root is None:
            raise ValueError('pass --data_path (the image root that split-file paths are relative to)')
        # mean/std default to the dataset statistics; pass the encoder's to match its pretraining
        normalize = transforms.Normalize(mean=mean or self.mean, std=std or self.std)
        self.num_classes = num_classes or self.num_classes

        transform_train=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
           ])

        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        self.train_txt = train_txt or os.path.join(root, self.train_file)
        self.val_txt = val_txt or os.path.join(root, self.val_file)
        trainset = LT_Dataset(root, self.train_txt, transform=transform_train, class_balance=class_balance, num_classes=self.num_classes)
        testset = LT_Dataset_Test(root, self.val_txt, transform=transform_test, class_map=trainset.class_map)

        self.trainset = trainset
        self.testset = testset
        # Train images per class in label order: labels are sorted by count, rarest class first
        self.cls_num_list_sorted = sorted(trainset.cls_num_list)
        self.batch_size = batch_size
        self.num_works = num_works

        self.train = torch.utils.data.DataLoader(
            trainset,
            batch_size = batch_size, shuffle = True,
            num_workers = num_works, pin_memory = True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset,
            batch_size = batch_size, shuffle = False,
            num_workers = num_works, pin_memory = True)


