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