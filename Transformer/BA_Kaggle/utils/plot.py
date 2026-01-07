import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

import utils.detr.detr_transforms as T
from utils.detr.box_ops import rescale_bboxes
from utils.timer import inference_timer


CLASSES = ['N/A', 'dp3', 'mp3', 'pp3', 'radius',
           'ulna', 'mc1', 'carpal', 'pp1', 'mp5']

color_list = [[0, 0, 0], [255, 0, 152], [150, 150, 0], [250, 250, 150], [0, 255, 0],
              [0, 150, 0], [90, 50, 90], [0, 0, 250], [0, 0, 150], [200, 0, 50]]


@inference_timer
def detect(img, model, device, threshold=0.5):
    transforms = T.Compose([
        T.RandomResize([800], max_size=800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformed_img = transforms(img, target=None)[0].unsqueeze(0).to(device)
    outputs, _ = model(transformed_img, dn_args=0)
    del transformed_img

    # keep only predictions with 0.5+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep].detach().cpu(), (img.size[0], img.size[1]))

    return probas[keep], bboxes_scaled


def get_patch(box):
    patch = patches.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1], edgecolor='red', fill=False)
    return patch


def get_text_loc(box):
    return (box[0], box[1]-30)


def draw_bbox(frame, scores, bboxes):
    """
    Draw bounding boxes
    """
    for box, score in zip(bboxes, scores):
        # pred label
        cls_idx = score.argmax()
        cls = CLASSES[cls_idx]

        # draw bbox on frame
        box = list(map(int, box))

        cv2.rectangle(frame, (box[0], box[1]), (box[2],
                      box[3]), color_list[cls_idx], thickness=4)

        # show classs and scores
        text = f"{cls} {score[cls_idx]:.3f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (box[0], box[1]-15-h),
                      (box[0]+w, box[1]), color_list[cls_idx], -1)
        cv2.putText(frame, text, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def plot_results(imgs, scores, boxes):
    figure = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(top=0.8)
    for idx, (img, score, box) in enumerate(zip(imgs, scores, boxes)):
        plt.subplot(4, 4, idx+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax) in zip(score, box):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax -
                         ymin, fill=False, color=color_list[cl], linewidth=3))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.3))

    return figure


def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim()
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (
            img.size(0), str(img.size()))
        img_perm = img.permute(1, 2, 0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2, 0, 1)
    else:  # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (
            img.size(1), str(img.size()))
        img_perm = img.permute(0, 2, 3, 1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0, 3, 1, 2)
