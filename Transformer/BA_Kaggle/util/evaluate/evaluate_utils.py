import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from models.pose_hrnet import PoseHRNet
from evaluate.joint_evaluate import get_max_preds
from utils.timer import load_timer, inference_timer


@load_timer
def load_keypoint_model(key_n, ckpt, device):
    model = PoseHRNet(key_n)
    model = model.to(device)
    model.init_weights(model, ckpt, device)
    model.eval()
    return model


# @inference_timer
def fine_answer(coco, file_name, key=None):
    length = len(coco.imgs.keys())
    img_ids_info = [coco.loadImgs(idx)[0] for idx in range(
        length) if file_name in coco.loadImgs(idx)[0]['file_name']]

    if len(img_ids_info) == 1:
        img_ids = [info['id'] for info in img_ids_info]
        annIds = coco.getAnnIds(imgIds=img_ids, iscrowd=False)
        objs = coco.loadAnns(annIds)[0]['keypoints']
        return objs
    else:
        img_ids = [info['id']
                   for info in img_ids_info if info['file_name'].split('_')[0] == str(key)]
        annIds = coco.getAnnIds(imgIds=img_ids, iscrowd=False)
        objs = coco.loadAnns(annIds)[0]['keypoints']
        return objs


def inference_transforms():
    return A.Compose([
        A.Resize(384, 288),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def resize_transforms(h, w):
    return A.Compose([
        A.Resize(h, w),
    ], keypoint_params=A.KeypointParams(format='xy'))


def hflip_transforms():
    return A.Compose([
        A.HorizontalFlip(p=1.),
    ], keypoint_params=A.KeypointParams(format='xy'))


# @inference_timer
def inference_keypoints(img, model, device):
    inference_transform = inference_transforms()

    h, w = img.shape[:2]

    transformed_img = inference_transform(image=img)['image'].to(device)
    raw_img = A.Resize(384, 288)(image=img)['image']

    with torch.no_grad():
        output = model(transformed_img.unsqueeze(0))

    pred, _ = get_max_preds(output.detach().cpu().numpy())

    points = []
    for point in pred[0]:
        point = list(map(int, point))
        point = [x*a for x, a in zip(point, [4, 4])]
        points.append(point)

    resize_transform = resize_transforms(h, w)
    resize_transformed = resize_transform(image=raw_img, keypoints=points)
    points = resize_transformed['keypoints']

    return points
