import numpy as np
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
from os import path, mkdir
from load_data import *
from time import time
from roi_pooling import roi_pooling_ims

numClasses = 4
numPoints = 4
imgSize = (360, 360)
batchSize = 8 
resume_file = str('module1.pth')


provNum, alphaNum, adNum = 38, 25, 35
alphabet_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class wR2(nn.Module):
    def __init__(self, num_classes=950):
        super(wR2, self).__init__()
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        layer2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        layer5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        layer6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        layer7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        layer8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        layer9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        layer10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
            layer6,
            layer7,
            layer8,
            layer9,
            layer10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            nn.Linear(100, 100),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x


class module1(nn.Module):
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        self.collector1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, provNum),
        )
        self.collector2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphaNum),
        )
        self.collector3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.collector4 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.collector5 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.collector6 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )
        self.collector7 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path,map_location='cpu'))

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)
        _x1 = self.wR2.module.features[1](x0)
        x2 = self.wR2.module.features[2](_x1)
        _x3 = self.wR2.module.features[3](x2)
        x4 = self.wR2.module.features[4](_x3)
        _x5 = self.wR2.module.features[5](x4)

        x6 = self.wR2.module.features[6](_x5)
        x7 = self.wR2.module.features[7](x6)
        x8 = self.wR2.module.features[8](x7)
        x9 = self.wR2.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.wR2.module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]), requires_grad=False)
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]), requires_grad=False)
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]), requires_grad=False)

        assert boxLoc.data.size()[1] == 4
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.collector1(_rois)
        y1 = self.collector2(_rois)
        y2 = self.collector3(_rois)
        y3 = self.collector4(_rois)
        y4 = self.collector5(_rois)
        y5 = self.collector6(_rois)
        y6 = self.collector7(_rois)
        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]


def isEqual(labelGT, labelP):
    print (labelGT)
    print (labelP)
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    return sum(compare)


model_conv = fh02(numPoints, numClasses)
model_conv = torch.nn.DataParallel(model_conv)
model_conv.load_state_dict(torch.load(resume_file,map_location='cpu'))
model_conv.eval()


dst = demoTestDataLoader('/home/bill/Desktop/CCPD-master/rpnet/demo'.split(','), imgSize)
trainloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=1)

start = time()
for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    img = cv2.imread(ims[0])
    left_up = [(cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0]]
    right_down = [(cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0]]
    cv2.rectangle(img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255), 2)
    # We only print western characters in this program, as it is more convenient
    lpn = alphabet_chars[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]
    cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1])-20), cv2.FONT_ITALIC, 2, (0, 0, 255),thickness=3)
    cv2.imwrite(ims[0], img)

