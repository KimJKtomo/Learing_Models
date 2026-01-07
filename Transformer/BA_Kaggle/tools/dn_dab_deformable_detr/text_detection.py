import cv2
import torch
from PIL import Image

from utils.timer import load_timer, inference_timer

from .detr_transforms import *
from .box_ops import rescale_bboxes
from .dab_deformable_detr import build_dab_deformable_detr

import warnings
warnings.filterwarnings(action='ignore')


score_threshold = 0.999


@inference_timer
def text_detect(image, det_model, device):
    transforms = Compose([
        RandomResize([800], max_size=800),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # copy to original
    draw = image.copy()

    # preprocess image for network
    image = Image.fromarray(image)
    transformed_img = transforms(image, target=None)[
        0].unsqueeze(0).to(device, non_blocking=True)

    # predict bboxees
    model = det_model.to(device)
    with torch.no_grad():
        outputs, _ = model(transformed_img, dn_args=0)
    del transformed_img

    # keep only predictions with threshold confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > score_threshold
    scores = probas[keep].detach().cpu().numpy()

    # convert boxes from [0; 1] to image scales
    boxes = rescale_bboxes(
        outputs['pred_boxes'][0, keep].detach().cpu(), (image.size[0], image.size[1]))

    # masking
    for box, score in zip(boxes.numpy(), scores):
        score = score[score.argmax()]
        if score >= score_threshold:
            box = box.astype(int)
            cv2.rectangle(draw, (box[0], box[1]),
                          (box[2], box[3]), [0, 0, 0], -1)

    return draw


@load_timer
def load_text_detection_model(device):
    text_ckpt_path = 'ckpts/text_det_ckpt/dn_dab_deformable_detr.pth'
    text_model, _, _ = build_dab_deformable_detr('resnet101')
    text_model = text_model.to(device)
    ckpt = torch.load(text_ckpt_path, map_location=device, weights_only=False)
    text_model.load_state_dict(ckpt['model'])
    text_model.eval()
    return text_model
