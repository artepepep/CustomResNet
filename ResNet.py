import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm
import cv2
import os


class Dataset2Class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir_list1 = sorted(os.listdir(path_dir1))
        self.dir_list2 = sorted(os.listdir(path_dir2))
    
    def __len__(self):
        return len(self.dir_list1) + len(self.dir_list2)
    
    def __getitem__(self, idx):
        if idx < len(self.dir_list1):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir_list1[idx])
        else:
            class_id = 1
            idx -= len(self.dir_list1)
            img_path = os.path.join(self.path_dir2, self.dir_list2[idx])
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255.0
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {'img': t_img, 'label': t_class_id}


train_dogs_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/training_set/dogs'
train_cats_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/training_set/cats'
test_dogs_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/test_set/dogs'
test_cats_path = '/Users/mac/Desktop/catdosgAI/archive-3/dataset/test_set/cats'
train_ds_catsdogs = Dataset2Class(train_dogs_path, train_cats_path)
test_ds_catsdogs = Dataset2Class(test_dogs_path, test_cats_path)


batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs,
    batch_size=batch_size,
    shuffle=1,
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs,
    batch_size=batch_size,
    shuffle=1,
    drop_last=False
)


class ResBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        
        self.conv0 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.norm0 = nn.BatchNorm2d(nc)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(nc)
    
    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(x)
        out = self.act(x)
        out = self.conv1(x)
        out = self.norm1(x)

        return self.act(x + out)


class ResTruck(nn.Module):
    def __init__(self, nc, num_blocks):
        super().__init__()

        truck = []
        for i in range(num_blocks):
            truck += [ResBlock(nc)]
        self.truck = nn.Sequential(*truck)
    
    def forward(self, x):
        return block(x)


class MyResNet(nn.Module):
    def __init__(self, in_nc, nc, out_nc):
        super().__init__()

        self.conv0 = nn.Conv2d(in_nc, nc, kernel_size=7, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = ResTruck(nc, 3)
        self.conv1 = nn.Conv2d(nc, nc*2, 3, padding=1, stride=2)
        self.layer2 = ResTruck(nc*2, 3)
        self.conv2 = nn.Conv2d(nc*2, nc*4, 3, padding=1, stride=2)
        self.layer3 = ResTruck(nc*4, 6)
        self.conv3 = nn.Conv2d(nc*4, nc*4, 3, padding=1, stride=2)
        self.layer4 = ResTruck(nc*4, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4*nc, out_nc)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.conv1(out)

        out = self.layer2(out)
        out = self.conv2(out)

        out = self.layer3(out)
        out = self.conv3(out)

        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out


my_resnet_model = MyResNet(3, 64, 1)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_resnet_model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_params(my_resnet_model))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_resnet_model = my_resnet_model.to(device)
loss_fn = loss_fn.to(device)

use_amp = True
scaler = torch.cuda.amp.GradScaler()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


epochs = 10
loss_epochs_list = []
acc_epochs_list = []
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['label']
        label = F.one_hot(label, 2).float()
        img = img.to(device)
        optimizer.zero_grad()

        with torch.autocast(enabled=use_amp):
            pred = my_resnet_model(img)
            loss = loss_fn(pred, label)

        scaler.scale(loss).backward()
        loss_item = loss.item()
        loss_val += loss_item
        
        scaler.step(optimizer)
        scaler.update()

        acc_current = accuracy(pred.cpu().float(), label.cpu().float())
        acc_val += acc_current

        pbar.set_description(f'loss: {loss_item: 5f}\taccuracy: {acc_current: 3f}')
    scheduler.step()
    loss_epochs_list += (loss_val/len(train_loader))
    acc_epochs_list += (loss_val/len(train_loader))
    print(loss_epochs_list[-1])
    print(acc_epochs_list[-1])


plt.title('Loss: ResNet')
plt.plot(loss_epochs_list)
plt.plot(acc_epochs_list)
plt.show()