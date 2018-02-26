import pandas as pd
from torch import np
import os
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import time

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


tag_to_ix = {"dog": 0, "cat": 1}
IMG_PATH = 'train/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'train.csv'


class dogsandcats(Dataset):
    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path, sep="   ")
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags'].str.split()

    def __getitem__(self, index):
        img = cv2.imread(self.X_train[index])
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        label = self.y_train[index]
        return img, label

    def __len__(self):
        return len(self.X_train.index)


transformations = transforms.Compose(
    [transforms.Scale(227), transforms.ToTensor()])
dest_train = dogsandcats(TRAIN_DATA, IMG_PATH, IMG_EXT, transformations)
train_loader = DataLoader(dest_train,
                          batch_size=256,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True
                          )


class ConvNet(torch.nn.Module):
    def __init__(self, output_dim):

        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential()
        # self.fc.add_module("dropout_1", nn.Dropout(p=0.5))
        # self.fc.add_module("bn_1", nn.BatchNorm1d(256))
        self.fc.add_module("fc1", nn.Linear(256 * 6 * 6, 4096))
        self.fc.add_module("relu_6", nn.ReLU())
        # self.fc.add_module("dropout_2", nn.Dropout(p=0.5))
        # self.fc.add_module("bn_2", nn.BatchNorm1d(4096))
        self.fc.add_module("fc2", nn.Linear(4096, 4096))
        self.fc.add_module("relu_7", nn.ReLU())
        self.fc.add_module("fc3", nn.Linear(4096, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc.forward(x)
        # pred = F.sigmoid(x)
        return x



n_classes = 2
model = ConvNet(output_dim=n_classes)
model.cuda()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

print get_n_params(model)

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.8)


losslist = []

txt_file = open("bn_loss.txt", "w")
loss_func = nn.CrossEntropyLoss()


def train(epoch):
    model.train()
    totalloss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda(async=True)

        # data=data.long()
        target = prepare_sequence(target[0], tag_to_ix)
        target = target.cuda(async=True)
        data = Variable(data)
        # target = Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # print(output, target)
        # print type(output),type(target)
        # print target
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        totalloss += loss.data[0]
        batchnum = batch_idx
    average_loss = totalloss / batchnum
    losslist.append(average_loss)
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, average_loss))
    return average_loss


totaltime = 0
for epoch in range(100):
    timestart = time.time()
    loss = train(epoch)
    timeend = time.time()
    timelen = timeend - timestart
    totaltime = totaltime + timelen
    txt_file.write(str(epoch) + "   " + str(loss) + "   " + str(totaltime) + "\n")
    print timelen
    if(loss < 0.02):
        break
txt_file.close()
