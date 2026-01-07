import cv2
import os
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils.timer import load_timer, inference_timer
from .image_utils import clahe, normalize, padding


transforms = [A.Compose([A.Resize(384, 384), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
              A.Compose([A.Resize(384, 384), A.Normalize(
                  [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
              A.Compose([A.Resize(384, 384), A.Normalize(
                  [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
              A.Compose([A.Resize(384, 384), A.InvertImg(p=1.), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()])]


@inference_timer
def cls_predict(roi_bbox_dic, gp_hand_image, cls_model_set, gender, device, path, type):

    result_dic = {'dp3' : None, 'mp3' : None, 'pp3' : None, 'radius' : None, 'pp1' : None, 'mp5' : None, 'smi' : None}
    # {'dp3': 5, 'mp3': 6, 'pp3': 1, 'radius': 0, 'pp1': 4, 'mp5': 7, 'smi': 7}
    result_dic = {}
    # os.makedirs(f'path{type}', exist_ok=True)
    path_basename = os.path.basename(path)
    path_basename, ext = os.path.splitext(path_basename)

    for roi in roi_bbox_dic.keys():
        classifier = cls_model_set[roi][0]
        if roi_bbox_dic[roi] == []:
            result_dic[roi] = []
        else:
            roi_crop_img = roi_bbox_dic[roi][1]
            if roi in 'carpal':
                roi_crop_img = padding(roi_crop_img)
            probability, predict_result = cls_inference(roi=roi,
                                                        roi_crop_img=roi_crop_img,
                                                        model=classifier,
                                                        device=device)

            result_dic[roi] = [probability[0], predict_result]



            # cv2.imwrite(f'path{type}/{path_basename}_{roi}{ext}', roi_crop_img)

    if gender == 'f':
        classifier = cls_model_set['f'][0]
    elif gender == 'm':
        classifier = cls_model_set['m'][0]
    probability, predict_result = cls_inference(roi='gp',
                                                roi_crop_img=gp_hand_image,
                                                model=classifier,
                                                device=device)
    # cv2.imwrite(f'path{type}/{path_basename}_gp{ext}', gp_hand_image)





    result_dic['gp'] = [probability[0], predict_result]
    return result_dic


def cls_inference(roi, roi_crop_img, model, device):

    value_as = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5,
                11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]

    value_as_ulna = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11,
                     11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]

    value_as_carpal = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
                       8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]

    model.eval()
    with torch.no_grad():
        if roi == 'pp1':
            tmp = torch.tensor(np.zeros((1, 2))).to(device, non_blocking=True)
        elif roi == 'mp5':
            tmp = torch.tensor(np.zeros((1, 3))).to(device, non_blocking=True)
        elif roi == 'ulna':  # 26
            tmp = torch.tensor(np.zeros((1, 26))).to(device, non_blocking=True)
        elif roi == 'carpal':  # 30
            tmp = torch.tensor(np.zeros((1, 30))).to(device, non_blocking=True)
        else:
            tmp = torch.tensor(np.zeros((1, 34))).to(device, non_blocking=True)





        for i, transform in enumerate(transforms):
            if i == 1:
                clahe_img = clahe(roi_crop_img)
                transformed_img = transform(image=clahe_img)[
                    'image'].to(device, non_blocking=True)
            if i == 2:
                norm_img = normalize(roi_crop_img)
                transformed_img = transform(image=norm_img)[
                    'image'].to(device, non_blocking=True)
            else:
                transformed_img = transform(image=roi_crop_img)[
                    'image'].to(device, non_blocking=True)
            output = model(transformed_img.unsqueeze(dim=0))
            del transformed_img

            tmp += output

    prob = F.softmax(tmp, dim=1).cpu().detach().numpy()

    if roi in ['pp1', 'mp5']:
        class_predict = torch.argmax(
            tmp.squeeze(dim=1), dim=1).cpu().detach().numpy()[0]
    elif roi == 'ulna':
        class_predict = value_as_ulna[torch.argmax(
            tmp.squeeze(dim=1), dim=1).cpu().detach().numpy()[0]]
    elif roi == 'carpal':
        class_predict = value_as_carpal[torch.argmax(
            tmp.squeeze(dim=1), dim=1).cpu().detach().numpy()[0]]
    else:
        class_predict = value_as[torch.argmax(
            tmp.squeeze(dim=1), dim=1).cpu().detach().numpy()[0]]

    return prob, class_predict


@load_timer
def load_cls_model(device):
    import os
    import torch
    import timm

    cls_ckpt_path = 'ckpts/roi_cls_ckpt'

    ckpt_dic = {
        'dp3':   f'{cls_ckpt_path}/dp3.pt',
        'mp3':   f'{cls_ckpt_path}/mp3.pt',
        'pp3':   f'{cls_ckpt_path}/pp3.pt',
        'radius':f'{cls_ckpt_path}/radius.pt',
        'ulna':  f'{cls_ckpt_path}/ulna.pt',
        'mc1':   f'{cls_ckpt_path}/mc1.pt',
        'carpal':f'{cls_ckpt_path}/carpal.pt',
        'pp1':   f'{cls_ckpt_path}/pp1.pt',
        'mp5':   f'{cls_ckpt_path}/mp5.pt',
        'f':     f'{cls_ckpt_path}/f/gp.pt',
        'm':     f'{cls_ckpt_path}/m/gp.pt',
    }

    # Classifier는 class_n 필수
    class_n_dic = {
        'pp1': 2,
        'mp5': 3,
        'ulna': 26,
        'carpal': 30,
        # 나머지는 우선 1 (회귀 가정)
        'dp3': 1, 'mp3': 1, 'pp3': 1, 'radius': 1, 'mc1': 1, 'f': 1, 'm': 1,
    }

    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if 'model' in ckpt and isinstance(ckpt['model'], dict):
                return ckpt['model']
            if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                return ckpt['state_dict']
        return ckpt

    def _strip_module(sd):
        if any(k.startswith('module.') for k in sd.keys()):
            return {k.replace('module.', '', 1): v for k, v in sd.items()}
        return sd

    def _sig(sd):
        # backbone 규모를 구분하는 시그니처 (가능하면 2개 키로)
        k1 = "model.layers.1.downsample.norm.weight"
        k2 = "model.layers.2.downsample.norm.weight"
        if k1 in sd and k2 in sd:
            return (int(sd[k1].shape[0]), int(sd[k2].shape[0]))
        if k1 in sd:
            return (int(sd[k1].shape[0]), -1)
        return (-1, -1)

    def _candidate_arches():
        # ✅ timm 내에서 가능한 swin 계열 후보를 넓게 수집
        cands = []
        for pat in [
            "swin*",
            "swinv2*",
        ]:
            cands.extend(timm.list_models(pat))

        # 우선순위 정렬: giant/huge > large > base > tiny
        def score(name: str):
            n = name.lower()
            s = 0
            if "giant" in n: s += 1000
            if "huge" in n: s += 900
            if "g" in n and "swinv2" in n: s += 50
            if "large" in n: s += 300
            if "base" in n: s += 200
            if "small" in n: s += 100
            if "tiny" in n: s += 50
            if "384" in n: s += 80
            if "window12" in n: s += 20
            return -s  # ascending sort

        # 중복 제거
        cands = list(dict.fromkeys(cands))
        cands.sort(key=score)
        return cands

    ARCH_CACHE = {}  # signature -> arch

    model_set = {k: [] for k in ckpt_dic.keys()}
    arches = _candidate_arches()

    for roi, weight_path in ckpt_dic.items():
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"[ROI_CLS] missing ckpt: {weight_path}")

        class_n = int(class_n_dic.get(roi, 1))

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        sd = _strip_module(_extract_state_dict(ckpt))
        signature = _sig(sd)

        # 캐시된 arch가 있으면 우선 사용
        chosen_arch = ARCH_CACHE.get(signature, None)

        def _try_load_with_arch(arch_name: str):
            m = Classifier(class_n, arch=arch_name, pretrained=False).to(device)
            # shape mismatch는 strict=False여도 RuntimeError로 터짐 → try/except로 탐색
            m.load_state_dict(sd, strict=False)
            m.eval()
            return m

        if chosen_arch is not None:
            try:
                BA_model = _try_load_with_arch(chosen_arch)
                model_set[roi].append(BA_model)
                continue
            except Exception:
                chosen_arch = None  # 캐시 무효화 후 재탐색

        # arch 자동 탐색 (처음 성공하는 timm 모델명 채택)
        last_err = None
        BA_model = None
        for arch_name in arches:
            try:
                BA_model = _try_load_with_arch(arch_name)
                chosen_arch = arch_name
                break
            except Exception as e:
                last_err = e
                continue

        if BA_model is None:
            raise RuntimeError(
                f"[ROI_CLS] No compatible timm arch found for roi={roi}, sig={signature}. "
                f"Last error: {repr(last_err)}"
            )

        ARCH_CACHE[signature] = chosen_arch
        print(f"[ROI_CLS] roi={roi} sig={signature} -> arch='{chosen_arch}'")

        model_set[roi].append(BA_model)

    return model_set



class Classifier(nn.Module):
    def __init__(self, class_n, arch, pretrained=False):
        super().__init__()
        import timm
        self.class_n = int(class_n)
        self.arch = arch

        # backbone only
        self.model = timm.create_model(self.arch, pretrained=pretrained, num_classes=0)
        feat_dim = self.model.num_features

        # 기존 코드가 model.head.fc 를 쓰는 구조였다면 유지해야 하므로 fc로 둠
        self.model.head = nn.Identity()
        self.model.head.fc = nn.Linear(feat_dim, self.class_n)

    def forward(self, x):
        import torch

        # timm 모델이면 forward_features가 존재하는 경우가 많음
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(x)
        else:
            feat = self.model(x)

        # --- 핵심: 어떤 형태든 (B, C)로 정규화 ---
        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        if feat.ndim == 4:
            # (B, C, H, W) -> GAP
            feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3:
            # (B, N, C) or (B, C, N) 둘 다 대응
            if feat.shape[-1] == self.model.num_features:
                # (B, N, C) -> token mean
                feat = feat.mean(dim=1)
            elif feat.shape[1] == self.model.num_features:
                # (B, C, N) -> token mean
                feat = feat.mean(dim=2)
            else:
                # 마지막 fallback: flatten 후 avg
                feat = feat.flatten(1).mean(dim=1, keepdim=True)

        # 이제 feat는 (B, C)여야 함
        out = self.model.head.fc(feat)
        return out
