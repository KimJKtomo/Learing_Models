import cv2
import math
import copy
import torch
import numpy as np

from .pose_hrnet import PoseHighResolutionNet
from .transforms import rotation,rotation_, crop_img, get_point
from .transforms import test_transforms, resize_transform, box_to_center_scale
from .joint_evaluate import get_max_preds
import torchvision.transforms as transforms
from scipy.spatial import distance

from lib.core.inference import get_final_preds
from lib.utils.transforms import get_affine_transform

from utils.timer import load_timer, inference_timer

pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
    
@inference_timer
def keypoint_detect(image_numpy, keypoint_model, device):
    model = keypoint_model
    input = test_transforms()(image=image_numpy)[
        'image'].to(device, non_blocking=True)
    with torch.no_grad():
        output = model(input.unsqueeze(0))

        preds, maxvals = get_max_preds(output.detach().cpu().numpy())
        tmp = [p*[4, 4] for p in preds[0]]
        img = input.permute(1, 2, 0).detach().cpu().numpy()
        del input

        transformed_keypoint = resize_transform(
            image_numpy.shape[0], image_numpy.shape[1])(image=img, keypoints=tmp)['keypoints']

        top, middle, thumb = list(map(int, transformed_keypoint[0])), list(
            map(int, transformed_keypoint[1])), list(map(int, transformed_keypoint[2]))
        point_save_list = [top[0], top[1],
                           middle[0], middle[1], thumb[0], thumb[1]]
        h, w = image_numpy.shape[:2]
        if np.all(maxvals > 0.1):
            top, middle, thumb = check_outliner(point_save_list, h, w)

            rot_crop_img, draw_img, angle = position_checker(
                image_numpy, top, middle, thumb, option=True)
            missing = False
        else:
            print('point missing', maxvals)
            rot_crop_img = image_numpy.copy()
            draw_img = image_numpy.copy()
            missing = True
            angle = -1000

        return rot_crop_img, draw_img, point_save_list, missing, angle

@inference_timer
def keypoint_detect2(image_numpy, keypoint_model, device):
    image = image_numpy.copy()
    config = keypoint_model.cfg
    h, w, _ = image_numpy.shape
    bboxes = [[(0,0), (w, h)]]
    if config['DATASET']['COLOR_RGB']:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    centers = []
    scales = []
    for bbox in bboxes:
        center, scale = box_to_center_scale(bbox, config['MODEL']['IMAGE_SIZE'][0], config['MODEL']['IMAGE_SIZE'][1])
        centers.append(center)
        scales.append(scale)
    pred_keypoints = get_pose_estimation_prediction(config, keypoint_model, image, centers, scales, pose_transform, device)
    keypoints = pred_keypoints[0][0]
    scores = pred_keypoints[1][0]
    point_save_list, scores = determine_keypoint(h+w, keypoints, scores)
    


    top, middle, thumb = check_outliner(point_save_list, h, w)
    
    # image_numpy = cv2.circle(image_numpy, (int(top[0]), int(top[1])), 10, (0, 0, 255), -1)
    # image_numpy = cv2.circle(
    #     image_numpy, (int(middle[0]), int(middle[1])), 10, (0, 255, 0), -1)
    # image_numpy = cv2.circle(
    #     image_numpy, (int(thumb[0]), int(thumb[1])), 10, (255, 0, 0), -1)
    # import imutils
    # cv2.imshow('d', imutils.resize(image_numpy, height= 800))
    # cv2.waitKey(0)
    # top, middle, thumb = [678, 207], [678, 1021], [1016, 932]
    top, middle, thumb = [list(map(int, coord)) for coord in [top, middle, thumb]]
    # print(top, middle, thumb)

    rot_crop_img, draw_img, angle = position_checker2(
        image_numpy, top, middle, thumb, option=True, )
    missing = False
    # print(top, middle, thumb)

    # if np.all(scores > 0.1):
    #     top, middle, thumb = check_outliner(point_save_list, h, w)

    #     rot_crop_img, draw_img = position_checker(
    #         image_numpy, top, middle, thumb, option=True)
    #     missing = False

    # else:
    #     print('point missing', scores)
    #     rot_crop_img = image_numpy
    #     draw_img = image_numpy
    #     missing = True

    return rot_crop_img, draw_img, point_save_list, missing, angle



def determine_keypoint(dist, kps, scores):
    oneHand = False
    # 거리가매우 가까울 경우(손 한개)
    if distance.euclidean(kps[0], kps[3]) < dist / 20:
        oneHand = True
    if distance.euclidean(kps[1], kps[4]) < dist / 20:
        oneHand = True
    if oneHand:
        if scores[2] > scores[5]: # 왼손
            return [kps[0][0],kps[0][1], kps[1][0], kps[1][1], kps[2][0], kps[2][1]], np.array([scores[0][0], scores[1][0], scores[2][0]])
        else: # 오른손
            return [kps[3][0],kps[3][1], kps[4][0], kps[4][1], kps[5][0], kps[5][1]], np.array([scores[3][0], scores[4][0], scores[5][0]])
    else: # 왼손
        return [kps[0][0],kps[0][1], kps[1][0], kps[1][1], kps[2][0], kps[2][1]], np.array([scores[0][0], scores[1][0], scores[2][0]])
        
    # 거리가 일정 거리 이상일 경우 왼쪽 선택
    

@load_timer
def load_keypoint_model(device, type):
    if type ==1:
        ckpt_path = 'ckpts/keypoint_det_ckpt/pose_hrnet_w32_384x288_99.pt'
        yaml_name = 'tools/hrnet/pose_hrnet_32w.yaml'
        model = PoseHighResolutionNet(3, [yaml_name, ''])
    elif type== 2:
        ckpt_path = 'ckpts/keypoint_det_ckpt/pose_hrnet_w48_384x288.pth'
        yaml_name = 'tools/hrnet/pose_hrnet_48w.yaml'
        model = PoseHighResolutionNet(6, [yaml_name, ['MODEL','EXTRA']])
    model = model.to(device)
    model.init_weights(ckpt_path)
    model.eval()
    return model


def check_outliner(point, h, w):
    top = [point[0], point[1]]
    middle = [point[2], point[3]]
    thumb = [point[4], point[5]]
    if point[0] <= 0:
        top[0] = 0
    elif point[0] >= w:
        top[0] = w
    if point[2] <= 0:
        middle[0] = 0
    elif point[2] >= w:
        middle[0] = w
    if point[4] <= 0:
        thumb[0] = 0
    elif point[4] >= w:
        thumb[0] = w

    if point[1] <= 0:
        top[1] = 0
    elif point[1] >= h:
        top[1] = h
    if point[3] <= 0:
        middle[1] = 0
    elif point[3] >= h:
        middle[1] = h
    if point[5] <= 0:
        thumb[1] = 0
    elif point[5] >= h:
        thumb[1] = h
    return top, middle, thumb


def position_checker(img, top, middle, thumb, option):
    # atan2를 통해 라디안를 구하고, 각도로 변환한다.
    angle = math.atan2(top[0] - middle[0], middle[1] - top[1])
    angle = math.degrees(angle)
    # 시계방향 90도
    if 45 <= angle <= 135:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0], top[1] = top[1], image_height - top[0]
        middle[0], middle[1] = middle[1], image_height - middle[0]
        thumb[0], thumb[1] = thumb[1], image_height - thumb[0]

    # 반시계방향 90도
    elif -135 <= angle <= -45:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0], top[1] = image_width - top[1], top[0]
        middle[0], middle[1] = image_width - middle[1], middle[0]
        thumb[0], thumb[1] = image_width - thumb[1], thumb[0]

    # 180도
    elif 135 <= angle <= 225 or -225 <= angle <= -135:
        img = cv2.rotate(img, cv2.ROTATE_180)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0], top[1] = image_width - top[0], image_height - top[1]
        middle[0], middle[1] = image_width - \
            middle[0], image_height - middle[1]
        thumb[0], thumb[1] = image_width - thumb[0], image_height - thumb[1]

    # 좌우 대칭
    if middle[0] > thumb[0]:
        img = cv2.flip(img, 1)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0] = image_width - top[0]
        middle[0] = image_width - middle[0]
        thumb[0] = image_width - thumb[0]
    # 양손 사진일 경우 왼손만 인식하기 위해 사진크기 조정
    image_height, image_width = img.shape[0], img.shape[1]
    if middle[0] < image_width // 2 and top[0] < image_width // 2 and thumb[0] < image_width // 2:
        flip_thumb_x = image_width - thumb[0]
        cutting_x_position = (flip_thumb_x + thumb[0]) // 2
        img = img[0:image_height, 0: int(cutting_x_position)]

    angle = math.atan2(top[0] - middle[0], middle[1] - top[1])
    angle = math.degrees(angle)
    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas = cv2.circle(canvas, (int(top[0]), int(top[1])), 2, (0, 0, 255), -1)
    canvas = cv2.circle(
        canvas, (int(middle[0]), int(middle[1])), 2, (0, 255, 0), -1)
    canvas = cv2.circle(
        canvas, (int(thumb[0]), int(thumb[1])), 2, (255, 0, 0), -1)

    if option:
        draw_img = copy.deepcopy(img)
        draw_img = cv2.circle(
            draw_img, (int(top[0]), int(top[1])), 10, (0, 0, 255), -1)
        draw_img = cv2.circle(
            draw_img, (int(middle[0]), int(middle[1])), 10, (0, 255, 0), -1)
        draw_img = cv2.circle(
            draw_img, (int(thumb[0]), int(thumb[1])), 10, (255, 0, 0), -1)
    if -60 <= angle <= 60:
        point_im, canvas, b_w, b_h = rotation(img, angle, canvas)
        draw_img, _, _, _ = rotation(draw_img, angle, canvas)

    top, middle, thumb = get_point(canvas)
    point_list = [top, middle, thumb]

    point_im = crop_img(point_im, top, middle, thumb, b_w, b_h)
    draw_img = crop_img(draw_img, top, middle, thumb, b_w, b_h)

    return point_im, draw_img, angle

def position_checker2(img, top, middle, thumb, option):
    # atan2를 통해 라디안를 구하고, 각도로 변환한다.
    angle = math.atan2(top[0] - middle[0], middle[1] - top[1])
    angle = math.degrees(angle)
    print(angle)
    # 시계방향 90도
    if 45 <= angle <= 135:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0], top[1] = top[1], image_height - top[0]
        middle[0], middle[1] = middle[1], image_height - middle[0]
        thumb[0], thumb[1] = thumb[1], image_height - thumb[0]

    # 반시계방향 90도
    elif -135 <= angle <= -45:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0], top[1] = image_width - top[1], top[0]
        middle[0], middle[1] = image_width - middle[1], middle[0]
        thumb[0], thumb[1] = image_width - thumb[1], thumb[0]

    # 180도
    elif 135 <= angle <= 225 or -225 <= angle <= -135:
        img = cv2.rotate(img, cv2.ROTATE_180)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0], top[1] = image_width - top[0], image_height - top[1]
        middle[0], middle[1] = image_width - \
            middle[0], image_height - middle[1]
        thumb[0], thumb[1] = image_width - thumb[0], image_height - thumb[1]

    # 좌우 대칭
    if middle[0] > thumb[0]:
        img = cv2.flip(img, 1)
        image_height, image_width = img.shape[0], img.shape[1]
        top[0] = image_width - top[0]
        middle[0] = image_width - middle[0]
        thumb[0] = image_width - thumb[0]

    
    # 양손 사진일 경우 왼손만 인식하기 위해 사진크기 조정
    # image_height, image_width = img.shape[0], img.shape[1]
    # if middle[0] < image_width // 2 and top[0] < image_width // 2 and thumb[0] < image_width // 2:
    #     flip_thumb_x = image_width - thumb[0]
    #     cutting_x_position = (flip_thumb_x + thumb[0]) // 2
    #     img = img[0:image_height, 0: int(cutting_x_position)]

    angle = math.atan2(top[0] - middle[0], middle[1] - top[1])
    angle = math.degrees(angle)
    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas = cv2.circle(canvas, (int(top[0]), int(top[1])), 2, (0, 0, 255), -1)
    canvas = cv2.circle(
        canvas, (int(middle[0]), int(middle[1])), 2, (0, 255, 0), -1)
    canvas = cv2.circle(
        canvas, (int(thumb[0]), int(thumb[1])), 2, (255, 0, 0), -1)

    if option:
        draw_img = copy.deepcopy(img)
        draw_img = cv2.circle(
            draw_img, (int(top[0]), int(top[1])), 10, (0, 0, 255), -1)
        draw_img = cv2.circle(
            draw_img, (int(middle[0]), int(middle[1])), 10, (0, 255, 0), -1)
        draw_img = cv2.circle(
            draw_img, (int(thumb[0]), int(thumb[1])), 10, (255, 0, 0), -1)
    if -60 <= angle <= 60:
        point_im, canvas, b_w, b_h = rotation_(img, angle, canvas)
        draw_img, _, _, _ = rotation_(draw_img, angle, canvas)

    top, middle, thumb = get_point(canvas)
    point_list = [top, middle, thumb]

    point_im = crop_img(point_im, top, middle, thumb, b_w, b_h)
    draw_img = crop_img(draw_img, top, middle, thumb, b_w, b_h)

    return point_im, draw_img, angle



def get_pose_estimation_prediction(cfg, pose_model, image, centers, scales, transform, device):
    rotation = 0
    with torch.no_grad():

        # pose estimation transformation
        model_inputs = []

        for center, scale in zip(centers, scales):
            trans = get_affine_transform(center, scale, rotation, cfg['MODEL']['IMAGE_SIZE'])
            # Crop smaller image of people
            model_input = cv2.warpAffine(
                image,
                trans,
                (int(cfg['MODEL']['IMAGE_SIZE'][0]), int(cfg['MODEL']['IMAGE_SIZE'][1])),
                flags=cv2.INTER_LINEAR)
            model_inputcopy = model_input.copy()
            # hwc -> 1chw
            model_input = transform(model_input)#.unsqueeze(0)
            model_inputs.append(model_input)
            # cv2.circle(model_inputcopy, (int(cfg.MODEL.IMAGE_SIZE[0]/2), int(cfg.MODEL.IMAGE_SIZE[1]/2)), 10, (255, 0, 0), 2)
            # cv2.imshow('f', model_inputcopy)
            # cv2.waitKey(0)
           
        # n * 1chw -> nchw
        model_inputs = torch.stack(model_inputs)
        # compute output heatmap
        output = pose_model(model_inputs.to(device))
        coords, scores = get_final_preds(
            cfg,
            output.cpu().detach().numpy(),
            np.asarray(centers),
            np.asarray(scales))
        return coords, scores
