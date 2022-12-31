import torch
import torch.nn as nn

class DNN1(nn.Module):
    def __init__(self, input_size: int, num_labels: int, p=0):
        super().__init__()
        self.classifier = nn.Sequential(
                                        nn.Linear(input_size, 512),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.Linear(512, 512),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.Linear(512, num_labels),
                                        )
    def forward(self, input):
        outputs = self.classifier(input)
        return outputs

class DNN2(nn.Module):
    def __init__(self, input_size: int, num_labels: int, p=0):
        super().__init__()
        self.classifier = nn.Sequential(
                                        nn.Linear(input_size, 768),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.Linear(768, 768),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.Linear(768, num_labels),
                                        )
    def forward(self, input):
        outputs = self.classifier(input)
        return outputs

class DNN3(nn.Module):
    def __init__(self, input_size: int, num_labels: int, p=0):
        super().__init__()
        self.classifier = nn.Sequential(
                                        nn.BatchNorm1d(input_size),
                                        nn.Linear(input_size, 768),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.BatchNorm1d(768),
                                        nn.Linear(768, 768),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.Linear(768, num_labels),
                                        )
    def forward(self, input):
        outputs = self.classifier(input)
        return outputs

class DNN4(nn.Module):
    def __init__(self, input_size: int, num_labels: int, p=0):
        super().__init__()
        self.classifier = nn.Sequential(
                                        nn.BatchNorm1d(input_size),
                                        nn.Linear(input_size, 768),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.BatchNorm1d(768),
                                        nn.Linear(768, 768),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.BatchNorm1d(768),
                                        nn.Linear(768, 128),
                                        nn.ReLU(True),
                                        nn.Dropout(p=p),

                                        nn.Linear(128, num_labels),
                                        )
    def forward(self, input):
        outputs = self.classifier(input)
        return outputs


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, is_bn=True):
        super(ConvBlock1, self).__init__()

        self.is_bn = is_bn
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=5, stride=2, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, input):
        if self.is_bn:
            output = self.bn(input)
            output = self.conv(output)
            output = self.maxpool(output)
        else:
            output = self.conv(input)
            output = self.maxpool(output)
        return output

class CNN1(nn.Module):
    def __init__(self, in_channels=1, num_labels=10, is_bn=True):
        super(CNN1, self).__init__()

        self.cb1 = ConvBlock1(in_channels, 64, is_bn=is_bn)
        self.cb2 = ConvBlock1(64, 128, is_bn=is_bn)
        self.flat = nn.Flatten()
        self.classifier = nn.Sequential(
                                        nn.BatchNorm1d(512),
                                        nn.Linear(512, 128),
                                        nn.ReLU(True),
                                        nn.Linear(128, num_labels),
        )
        
    def forward(self, input):
        output = self.cb1(input)
        output = self.cb2(output)
        output = self.flat(output)
        output = self.classifier(output)
        return output


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, is_bn=True):
        super(ConvBlock2, self).__init__()

        self.is_bn = is_bn
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=5, stride=2, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        
    def forward(self, input):
        if self.is_bn:
            output = self.bn(input)
            output = self.conv(output)
            output = self.maxpool(output)
        else:
            output = self.conv(input)
            output = self.maxpool(output)
        return output

class CNN2(nn.Module):
    def __init__(self, in_channels=1, num_labels=10, is_bn=True):
        super(CNN2, self).__init__()

        self.cb1 = ConvBlock2(in_channels, 16, is_bn=is_bn)
        self.cb2 = ConvBlock2(16, 8, is_bn=is_bn)
        self.flat = nn.Flatten()
        self.classifier = nn.Sequential(
                                        nn.BatchNorm1d(560),
                                        nn.Linear(560, 128),
                                        nn.ReLU(True),
                                        nn.Linear(128, num_labels)
        )
        
    def forward(self, input):
        output = self.cb1(input)
        output = self.cb2(output)
        output = self.flat(output)
        output = self.classifier(output)
        return output

class CNN3(nn.Module):
    def __init__(self, in_channels=1, num_labels=10, is_bn=True):
        super(CNN3, self).__init__()

        self.cb1 = ConvBlock2(in_channels, 16, is_bn=is_bn)
        self.cb2 = ConvBlock2(16, 16, is_bn=is_bn)
        self.flat = nn.Flatten()
        self.classifier = nn.Sequential(
                                        nn.BatchNorm1d(1120),
                                        nn.Linear(1120, 128),
                                        nn.ReLU(True),
                                        nn.Linear(128, num_labels)
        )
        
    def forward(self, input):
        output = self.cb1(input)
        output = self.cb2(output)
        output = self.flat(output)
        output = self.classifier(output)
        return output