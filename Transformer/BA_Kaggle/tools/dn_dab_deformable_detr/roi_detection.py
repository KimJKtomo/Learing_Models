import cv2
import torch
import numpy as np
from PIL import Image

from utils.timer import load_timer, inference_timer

from .detr_transforms import *
from .box_ops import rescale_bboxes
from .dab_deformable_detr import build_dab_deformable_detr

import warnings
warnings.filterwarnings(action='ignore')


score_threshold = 0.85



@inference_timer
def roi_detect(image, det_model, device):
    CLASSES = ['N/A', 'dp3', 'mp3', 'pp3', 'radius',
               'ulna', 'mc1', 'carpal', 'pp1', 'mp5']

    roi_point_dict = {'dp3': [], 'mp3': [], 'pp3': [], 'radius': [
    ], 'ulna': [], 'mc1': [], 'carpal': [], 'pp1': [], 'mp5': []}

    transforms = Compose([
        RandomResize([800], max_size=800),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # copy to original
    original_copy = image.copy()

    # preprocess image for network
    image = Image.fromarray(image)
    transformed_img = transforms(image, target=None)[
        0].unsqueeze(0).to(device, non_blocking=True)

    # predict bboxes
    model = det_model.to(device)
    with torch.no_grad():
        outputs, _ = model(transformed_img, dn_args=0)
    del transformed_img

    # keep only predictions with threshold confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > score_threshold

    # convert boxes from [0; 1] to image scales
    boxes = rescale_bboxes(
        outputs['pred_boxes'][0, keep].detach().cpu(), (image.size[0], image.size[1]))

    scores = probas[keep].detach().cpu().numpy()
    labels = np.array([CLASSES[score.argmax()] for score in scores])

    check_label = {}

    for box, score, label in zip(boxes.numpy(), scores, labels):
        box = box.astype(int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        if label in check_label.keys():
            if score.max() >= check_label[label]:
                check_label[label] = score.max()

                roi_crop = original_copy[y1:y2, x1:x2, :]
                roi_point_dict[label] = [
                    [x1, y1, x2, y2], roi_crop
                ]
        else:
            check_label[label] = score.max()

            roi_crop = original_copy[y1:y2, x1:x2, :]
            roi_point_dict[label] = [
                [x1, y1, x2, y2], roi_crop
            ]

    return roi_point_dict, scores, boxes


@load_timer
def load_roi_detection_model(device):
    roi_ckpt_path = 'ckpts/roi_det_ckpt/dn_dab_deformable_detr.pth'
    roi_model, _, _ = build_dab_deformable_detr('resnet101')
    roi_model = roi_model.to(device)

    ckpt = torch.load(roi_ckpt_path, map_location=device, weights_only=False)
    # ckpt 구조: {'model': state_dict, ...} 형태를 가정
    roi_model.load_state_dict(ckpt['model'])

    roi_model.eval()
    return roi_model