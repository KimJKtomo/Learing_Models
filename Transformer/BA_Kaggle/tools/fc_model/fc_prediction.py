import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.timer import load_timer, inference_timer


@inference_timer
def ba_predict(part_result, fc_reg_model_set, fc_cls_model_set, gender, device):

    feature_extract = []
    part_name_list = ['dp3', 'mp3', 'pp3',
                      'radius', 'ulna', 'carpal', 'mc1', 'gp']

    for part_name in part_name_list:
        part_feature = part_result[part_name][0].tolist()
        feature_extract.extend(part_feature)

    feature_extract = np.array(feature_extract, dtype=np.float32)
    feature_extract = np.expand_dims(feature_extract, axis=0)
    feature_extract = feature_extract.reshape(-1)
    feature_extract = torch.tensor(feature_extract).unsqueeze(
        0).to(device, non_blocking=True)

    fc_reg_model = fc_reg_model_set[gender]
    fc_cls_model = fc_cls_model_set[gender]

    fc_reg_model.eval()
    fc_cls_model.eval()

    with torch.no_grad():
        reg_predict = fc_reg_model(feature_extract)
        reg_predict = reg_predict[0].detach().cpu().numpy()

        cls_predict = fc_cls_model(feature_extract)
        cls_predict = cls_predict[0].detach().cpu().numpy()

    return reg_predict[0], cls_predict


@load_timer
def load_fc_reg_model(device):
    model_set = {'f': [], 'm': []}
    for k in model_set.keys():
        model = Regressor(261, k)
        if k == 'f':
            model.load_state_dict(torch.load(
                'ckpts/fc_ckpt/regression/female/ba_reg.pt', map_location=device))
        elif k == 'm':
            model.load_state_dict(torch.load(
                'ckpts/fc_ckpt/regression/male/ba_reg.pt', map_location=device))
        model = model.to(device)
        model_set[k] = model
    return model_set


@load_timer
def load_fc_cls_model(device):
    model_set = {'f': [], 'm': []}
    for k in model_set.keys():
        model = Classifier(261, k)
        if k == 'f':
            model.load_state_dict(torch.load(
                'ckpts/fc_ckpt/classification/female/ba_cls.pt', map_location=device))
        elif k == 'm':
            model.load_state_dict(torch.load(
                'ckpts/fc_ckpt/classification/male/ba_cls.pt', map_location=device))
        model = model.to(device)
        model_set[k] = model
    return model_set


class Regressor(nn.Module):
    def __init__(self, in_features, gender):
        super(Regressor, self).__init__()

        layers = []
        if gender == 'f':
            feature_list = [384]
            dropout_list = [0.04008342975594972]
        elif gender == 'm':
            feature_list = [284]
            dropout_list = [0.4998068435054061]

        for feature, dropout in zip(feature_list, dropout_list):
            layers.append(nn.Linear(in_features, feature))
            layers.append(nn.BatchNorm1d(feature))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))

            in_features = feature

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, inputs):
        x = self.layers(inputs)
        y = self.fc(x)
        return y


class Classifier(nn.Module):
    def __init__(self, in_features, gender):
        super(Classifier, self).__init__()

        layers = []
        if gender == 'f':
            feature_list = [267]
            dropout_list = [0.20869487120574273]
        elif gender == 'm':
            feature_list = [130]
            dropout_list = [0.00911153738403415]

        for feature, dropout in zip(feature_list, dropout_list):
            layers.append(nn.Linear(in_features, feature))
            layers.append(nn.BatchNorm1d(feature))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            in_features = feature

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features, 34)

    def forward(self, inputs):
        x = self.layers(inputs)
        y = self.fc(x)
        return F.softmax(y, dim=1)
