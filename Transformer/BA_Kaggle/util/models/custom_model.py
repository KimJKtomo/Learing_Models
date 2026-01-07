import timm
import torch
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.fe = timm.create_model(
            'swin_large_patch4_window12_384_in22k', pretrained=True)
        self.fe.reset_classifier(0)

    # @torch.cuda.amp.autocast()
    def forward(self, inputs):
        x = self.fe(inputs)
        return x


class MainClassifier(nn.Module):
    def __init__(self, img_features=1536, main_classes=2, drop_p=0.1):
        super(MainClassifier, self).__init__()
        self.classifier = nn.Linear(img_features, main_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)

    # @torch.cuda.amp.autocast()
    def forward(self, inputs):
        x = self.classifier(self.act(self.drop(inputs)))
        return x


class SubClassifier(nn.Module):
    def __init__(self, img_features=1536, sub_classes=4, drop_p=0.1):
        super(SubClassifier, self).__init__()
        self.classifier = nn.Linear(img_features, sub_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_p)

    # @torch.cuda.amp.autocast()
    def forward(self, inputs, mask):
        x = self.classifier(self.act(self.drop(inputs)))
        return x.masked_fill_(mask, -10000.)
