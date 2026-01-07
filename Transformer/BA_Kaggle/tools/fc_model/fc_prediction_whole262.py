
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.timer import load_timer, inference_timer


# =========================
# FC inference (v2, whole+roi)
# - 기존 260-dim ROI/TW3 feature + whole_pred_month + whole_conf = 262
# - 새로 학습한 fc_ckpt를 로드해야 함 (기존 260 ckpt와 호환 X)
# =========================

FC_IN_FEATURES = 262
PART_NAME_LIST = ['dp3', 'mp3', 'pp3', 'radius', 'ulna', 'carpal', 'mc1', 'gp']


@inference_timer
def ba_predict(part_result,
               fc_reg_model_set,
               fc_cls_model_set,
               gender,
               device,
               whole_pred_month: float | None = None,
               whole_conf: float | None = None):
    """
    part_result: dict[str, tuple[np.ndarray, ...]]
      - 기존 파이프라인이 생성하는 결과 구조를 그대로 사용.
      - 각 part에서 part_result[part_name][0]이 1D feature array라고 가정.
    whole_pred_month: whole classifier 기대값(months) 또는 month-scale prediction
    whole_conf: whole classifier confidence (max softmax 등, 0~1)
    """
    feature_extract = []

    for part_name in PART_NAME_LIST:
        part_feature = part_result[part_name][0].tolist()
        feature_extract.extend(part_feature)

    # append whole features
    # (누락 시 0으로 채움: 학습/추론 안정성)
    feature_extract.append(float(whole_pred_month) if whole_pred_month is not None else 0.0)
    feature_extract.append(float(whole_conf) if whole_conf is not None else 0.0)

    feature_extract = np.array(feature_extract, dtype=np.float32).reshape(-1)
    if feature_extract.shape[0] != FC_IN_FEATURES:
        raise RuntimeError(
            f"[FC] feature dim mismatch: got {feature_extract.shape[0]}, expected {FC_IN_FEATURES}. "
            f"(Check PART_NAME_LIST features + whole features.)"
        )

    x = torch.from_numpy(feature_extract).unsqueeze(0).to(device, non_blocking=True)

    fc_reg_model = fc_reg_model_set[gender]
    fc_cls_model = fc_cls_model_set[gender]

    fc_reg_model.eval()
    fc_cls_model.eval()

    with torch.no_grad():
        reg_predict = fc_reg_model(x)[0].detach().cpu().numpy()
        cls_predict = fc_cls_model(x)[0].detach().cpu().numpy()

    # reg_predict shape: (1,) or (1,1) -> [0]
    return float(np.ravel(reg_predict)[0]), cls_predict


@load_timer
def load_fc_reg_model(device, ckpt_root="ckpts/fc_ckpt_v2"):
    """
    ckpt_root 예시:
      ckpts/fc_ckpt_v2/regression/female/ba_reg.pt
      ckpts/fc_ckpt_v2/regression/male/ba_reg.pt
    """
    model_set = {'f': None, 'm': None}
    for g in model_set.keys():
        model = Regressor(FC_IN_FEATURES, g)
        if g == 'f':
            p = f"{ckpt_root}/regression/female/ba_reg.pt"
        else:
            p = f"{ckpt_root}/regression/male/ba_reg.pt"
        sd = torch.load(p, map_location=device)
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
        model_set[g] = model
    return model_set


@load_timer
def load_fc_cls_model(device, ckpt_root="ckpts/fc_ckpt_v2"):
    """
    ckpt_root 예시:
      ckpts/fc_ckpt_v2/classification/female/ba_cls.pt
      ckpts/fc_ckpt_v2/classification/male/ba_cls.pt
    """
    model_set = {'f': None, 'm': None}
    for g in model_set.keys():
        model = Classifier(FC_IN_FEATURES, g)
        if g == 'f':
            p = f"{ckpt_root}/classification/female/ba_cls.pt"
        else:
            p = f"{ckpt_root}/classification/male/ba_cls.pt"
        sd = torch.load(p, map_location=device)
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
        model_set[g] = model
    return model_set


class Regressor(nn.Module):
    """
    기존 fc_prediction.py의 구조를 그대로 사용.
    (입력 차원만 262로 변경)
    """
    def __init__(self, in_features, gender):
        super().__init__()

        layers = []
        if gender == 'f':
            feature_list = [384]
            dropout_list = [0.04008342975594972]
        else:
            feature_list = [284]
            dropout_list = [0.08293688710677454]

        for feature, dropout in zip(feature_list, dropout_list):
            layers.append(nn.Linear(in_features, feature))
            layers.append(nn.BatchNorm1d(feature))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = feature

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, inputs):
        x = self.layers(inputs)
        y = self.fc(x)
        return y


class Classifier(nn.Module):
    """
    기존 fc_prediction.py의 구조를 그대로 사용.
    (입력 차원만 262로 변경)
    """
    def __init__(self, in_features, gender):
        super().__init__()

        layers = []
        if gender == 'f':
            feature_list = [267]
            dropout_list = [0.20869487120574273]
        else:
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
