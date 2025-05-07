import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50

class DilatedResNet50(nn.Module):
    def __init__(self):
        super(DilatedResNet50, self).__init__()
        resnet = resnet50(pretrained=True)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        self.layer3 = self._make_dilated_layer(resnet.layer3, dilation=2)
        self.layer4 = self._make_dilated_layer(resnet.layer4, dilation=4)

    def _make_dilated_layer(self, layer, dilation):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding = (dilation, dilation)
        return layer

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PSPN(nn.Module):
    def __init__(self):
        super(PSPN, self).__init__()

        self.size_1 = 1
        self.size_2 = 2
        self.size_3 = 3
        self.size_6 = 6
        self.conv_1x1 = nn.Conv2d(in_channels=2048, out_channels=int(2048/4), kernel_size=1, stride=1, padding=0, bias=False)

        self.conv = nn.Sequential(nn.Conv2d(in_channels=4096, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(1024),nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride = 1, padding=1, bias=False))

    def pool(self,x,size):
        size_tuple = (size,size)
        x = nn.AdaptiveAvgPool2d(size_tuple)(x)
        return x

    def concat(self,feature_map, x1,x2,x3,x4):
        return torch.cat((feature_map,x1,x2,x3,x4),1)

    def forward(self, x):

        h,w = x.size()[2:]

        x_1 = self.pool(x, self.size_1)
        x_2 = self.pool(x, self.size_2)
        x_3 = self.pool(x, self.size_3)
        x_6 = self.pool(x, self.size_6)

        x_1 = self.conv_1x1(x_1)
        x_2 = self.conv_1x1(x_2)
        x_3 = self.conv_1x1(x_3)
        x_6 = self.conv_1x1(x_6)

        x_1 = nn.Upsample(size = (h,w) , mode='bilinear', align_corners=True)(x_1)
        x_2 = nn.Upsample(size = (h,w), mode='bilinear', align_corners=True)(x_2)
        x_3 = nn.Upsample(size= (h,w), mode='bilinear', align_corners=True)(x_3)
        x_6 = nn.Upsample(size = (h,w), mode='bilinear', align_corners=True)(x_6)

        x = self.concat(x, x_1, x_2, x_3, x_6) # [4096, 32, 32]

        x = self.conv(x)
        x = nn.Upsample(size = (256,256), mode='bilinear', align_corners=True)(x)

        return x


