 
from scipy.spatial import distance
import pandas as pd
from imutils import paths
import timm
import argparse
import namespace
import os
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import csv
import torch.nn.functional as F
import numpy as np
import imgaug.augmenters as iaa

import time
import math
import json
from tqdm import tqdm
import imutils
import traceback
import shutil
from scipy.spatial.distance import cdist

if __name__ == "main":
    from ..models.torch_models_CLS import Classifier
    from ..models.pose_hrnet import PoseHighResolutionNet
    from ..evaluate.joint_evaluate import get_max_preds
    from ..lib.utils.transforms import *
    from ..lib.core.inference import get_final_preds
else : 
    from util.models.torch_models_CLS import Classifier
    from util.models.pose_hrnet import PoseHighResolutionNet
    from util.evaluate.joint_evaluate import get_max_preds
    from util.lib.utils.transforms import *
    from util.lib.core.inference import get_final_preds

images = sorted(paths.list_images('images'))

# 이미지 로드

# 모델 로드 (keypoint, 분류, detection, segmentation)

class classification():
    def __init__(self, features):
        # load model
        self.C_models = [None] * len(features)
        self.C_devices = [None] * len(features)
        self.C_namespace = [None] * len(features)
        for i in range(len(features)):
            self.C_namespace[i] = namespace()
            self.C_namespace[i].device = features[i][0]
            self.C_namespace[i].model = features[i][1]
            self.C_namespace[i].trained = features[i][2]
            self.C_namespace[i].class_n = features[i][3]
            self.C_namespace[i].img_size = features[i][4]
            
            if str(self.C_namespace[i].device) == "0":
                print(f'cuda:0')
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                self.C_devices[i] = torch.device(f'cuda:0')
            elif str(self.C_namespace[i].device) == "1":
                print(f'cuda:1')
                os.environ['CUDA_VISIBLE_DEVICES'] = "1"
                self.C_devices[i] = torch.device(f'cuda')
            else:             
                print(f'cuda:cpu')
                self.C_devices[i] = torch.device("cpu")
        
            self.C_models[i] = Classifier(self.C_namespace[i])
            self.C_models[i] = self.C_models[i].to(self.C_devices[i])
            self.C_models[i].load_state_dict(
                torch.load(self.C_namespace[i].trained, map_location=self.C_devices[i])
            )
            self.C_models[i].eval()
            
    def transform(self, idx):
        if idx == 0:
            return A.Compose([
                A.Resize(self.C_namespace[idx].img_size, self.C_namespace[idx].img_size),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(),
    ])


    def image_padding(self, img):
        max_value = max(img.shape[:2])
        pad_img = A.PadIfNeeded(
            min_height=max_value, min_width=max_value, border_mode=0, value=(0, 0, 0)
        )(image=img)["image"]
        return pad_img

    def cls_predict(self, image, idx, tidx, flip = False, padding = False,  heatmap=True):
        import time
        jet_heatmap = None
        if flip:
            image = cv2.flip(image, 0)
        if padding:
            image = self.image_padding(image.copy())

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # tmp = torch.tensor(np.zeros((1, class_n))).to(self.C_devices[idx], non_blocking=True)
        transformed_image = self.transform(tidx)(image=image)['image'].to(self.C_devices[idx])
        transformed_image = transformed_image.unsqueeze(0).to(self.C_devices[idx])
        t = time.time()
        output = self.C_models[idx](transformed_image)
        prob = F.softmax(output.squeeze(0), dim=0).detach().cpu().numpy()
        prob = [int(x * 100) for x in prob]
        pred = np.argmax(prob)
        self.C_models[idx].eval()
        if heatmap:
            from pytorch_grad_cam import HiResCAM
            target_layers = []
            for k, v in self.C_models[idx].named_modules():
                if 'norm2' in k and 'layers' in k and 'blocks' in k:
                    target_layers.append(v)
            cam = HiResCAM(model=self.C_models[idx], target_layers=target_layers, use_cuda=True,
                           reshape_transform=reshape_transform)

            grayscale_cam = cam(input_tensor=transformed_image, targets=None, eigen_smooth=False, aug_smooth=False)
            grayscale_cam = grayscale_cam[0, :]

            crop_image = A.Resize(self.C_namespace[idx].img_size, self.C_namespace[idx].img_size)(image=image)['image']
            crop_image, heatmap, jet_heatmap = heatmap_filter(grayscale_cam, crop_image)

            jet_heatmap = A.Resize(h, w)(image=jet_heatmap)['image']
            softmax = torch.nn.Softmax(dim=0)

        return list(prob), pred, jet_heatmap

def reshape_transform(tensor):
    if len(tensor.size()) == 4:
        if tensor.size(1) == tensor.size(2):
            # tensor shape이 아래와 같이 [B, H, W, C]
            # ex) tensor.shape = torch.Size([1, 7, 7, 768])
            result = tensor.transpose(2, 3).transpose(1, 2)
            # result.shape = torch.Size([1, 768, 7, 7])
        elif tensor.size(2) == tensor.size(3):
            # tensor shape이 아래와 같이 [B, C, H, W]이면 그대로 사용
            # ex) tensor.shape = torch.Size([1, 768, 7, 7])
            result = tensor

    elif len(tensor.size()) == 3:
        if math.sqrt(tensor.size(1)) % 1 == 0:
            height = width = int(math.sqrt(tensor.size(1)))
            result = tensor.reshape(tensor.size(0),
                                    height, width, tensor.size(2))
        else:
            height = width = int(math.sqrt(tensor.size(1) - 1))
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                              height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)

    return result


def heatmap_filter(grayscale_cam, image, image_normalization=True, transparency=0.01, blue_channel_ratio=0.6,
                   blur_kernel=39, contrast_type='gamma', contrast_value=1.15):
    """
        image_normalization : Min-Max normalization
        transparency : transparency
        blue_channel_ratio : blue channel ratio of grayscale_cam
        blue_kernel : kernel size of gaussain blur
        contrast_type : contrast type (gamma, linear, log)
        contrast_value : contrast value
    """
    # normalize image
    if image_normalization is True:
        image = (image - np.min(image))
        image = np.uint8(image / np.max(image) * 255.)

    # apply Gaussian Blur
    if blur_kernel and blur_kernel > 0:
        grayscale_cam = cv2.GaussianBlur(
            grayscale_cam, (blur_kernel, blur_kernel), 0)

    # apply contrast
    if contrast_type == 'gamma':
        grayscale_cam = iaa.GammaContrast(
            contrast_value).augment_image(grayscale_cam)
    elif contrast_type == 'linear':
        grayscale_cam = iaa.LinearContrast(
            contrast_value).augment_image(grayscale_cam)
    elif contrast_type == 'log':
        grayscale_cam = iaa.LogContrast(
            contrast_value).augment_image(grayscale_cam)

    # apply colormap to heatmap
    temp_heatmap = grayscale_cam.copy()
    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # apply blue channel filter
    if blue_channel_ratio and blue_channel_ratio < 1.:
        heatmap[:, :, 2] = np.uint8(heatmap[:, :, 2] * blue_channel_ratio)  # RGB
    heatmap = np.float32(heatmap) / 255.

    # heatmap blending
    cam_visual = transparency * image + (1 - transparency) * heatmap
    cam_visual = cam_visual / np.max(cam_visual)
    cam_visual = np.uint8(cam_visual * 255.)
    cam_visual = cv2.cvtColor(cam_visual, cv2.COLOR_BGR2RGB)
    temp_heatmap *= 255.
    return image, temp_heatmap, cam_visual









class keypointDetection():
    def __init__(self, features):
        # load model
        self.K_models = [None] * len(features)
        self.K_devices = [None] * len(features)
        self.K_namespace = [None] * len(features)
        for i in range(len(features)):
            self.K_namespace[i] = namespace()
            self.K_namespace[i].device = features[i][0]
            self.K_namespace[i].model = features[i][1]
            self.K_namespace[i].key_ckpt = features[i][2]
            self.K_namespace[i].key_n = features[i][3]
            self.K_namespace[i].img_height = features[i][4]
            self.K_namespace[i].img_width = features[i][5]
            self.K_namespace[i].labels = features[i][6]
            self.K_namespace[i].yaml_file = features[i][7]

            if str(self.K_namespace[i].device) == "0":
                print(f'cuda:0')
                os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                self.K_devices[i] = torch.device(f'cuda:0')
            elif str(self.K_namespace[i].device) == "1":
                print(f'cuda:1')
                os.environ['CUDA_VISIBLE_DEVICES'] = "1"
                self.K_devices[i] = torch.device(f'cuda')
            else:             
                print(f'cuda:cpu')
                self.K_devices[i] = torch.device("cpu")


            # os.environ['CUDA_VISIBLE_DEVICES'] = f"{self.K_namespace[i].device}"
            # self.K_devices[i] = torch.device(f'cuda:{self.K_namespace[i].device}')
            self.K_models[i] = PoseHighResolutionNet(self.K_namespace[i].yaml_file, self.K_namespace[i])
            self.K_models[i] = self.K_models[i].to(self.K_devices[i])
            self.K_models[i].init_weights(self.K_models[i], self.K_namespace[i].key_ckpt, self.K_devices[i])
            self.K_models[i] = torch.nn.DataParallel(self.K_models[i], device_ids=(0,) ).cuda()
            self.K_models[i].eval()
        
    def transform(self, idx):
        return A.Compose([
            A.Resize(self.K_namespace[idx].img_height, self.K_namespace[idx].img_width),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()]
        )

    def keypoint_detect(self, image, idx, threshold = 0.1):
        resultimage = image.copy()
        t = time.time()
        raw_img = A.Resize(int(self.K_namespace[idx].img_height), int(self.K_namespace[idx].img_width))(image=image)["image"]

        transformed_img = self.transform(idx)(image = image)['image']
        
        
        
        with torch.no_grad():
            outputs = self.K_models[idx](transformed_img.unsqueeze(0).to(self.K_devices[idx]))

        pred, probs = get_max_preds(outputs.detach().cpu().numpy())
        prediction = [(i, self.K_namespace[idx].labels[num]) if float(i[1] > threshold) else None for num, i in enumerate(zip(pred[0], probs[0]))]
        ######################## 1 

        # t = time.time()
        # pred = np.array([(np.array(list(map(int, p)))*np.array([4, 4])).tolist()
        #                 for p in pred[0]])
        # resize_transforms = A.Compose([A.Resize(image.shape[0], image.shape[1])], keypoint_params=A.KeypointParams(format='xy'))
        # predicted = resize_transforms(
        #     image = raw_img, keypoints = pred)

        # for num, p in enumerate(predicted['keypoints']):
        
        #     cv2.line(image, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (0, 0, 255), 40)
        #     print(int(p[0]), int(p[1]))
        #     # cv2.putText(image, str(num), (p[0]-10, p[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # print(time.time() - t, '11111111111111111')
        ###################################

        ######################## 2
        t = time.time()
        
        multiplyX = image.shape[1] / self.K_namespace[idx].img_width 
        multiplyY = image.shape[0] / self.K_namespace[idx].img_height
        coords = [(i[0][0][0]*multiplyX * 4, i[0][0][1] * multiplyY *4) if i != None else None for i in prediction]
        labels = [i[1]  if i != None else None for i in prediction]
        for p in coords:
            if p != None:
                cv2.line(resultimage, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (0, 0, 255), 40)
        # cv2.imwrite('result.jpg', resultimage)
        ###################################

        

        return image, coords, labels
    

    # 
    def keypoint_detect2(self, image, idx, box, m_width, m_height, threshold = 0.1):
        """
        idx : 모델 인덱스
        box : 박스 좌표들(원본 그대로 일 경우 이미지의 width, height)
        width : 이미지 width
        height : 이미지 height
        m_width : 모델 검증 width
        m_height : 모델 검증 height
        """
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255,0,0))
        with torch.no_grad():
            c, s = self._box2cs(box, m_width, m_height)
            r = 0

            trans = get_affine_transform(c, s, r, [m_width, m_height])
            input = cv2.warpAffine(
                image, 
                trans,
                (int(m_width), int(m_height)),
                flags = cv2.INTER_LINEAR
            )
            # cv2.imshow('input', input)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ])

            input = transform(input).unsqueeze(0)
            output = self.K_models[idx](input)
            preds, maxvals = self.get_final_preds(output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))
            coords = [list(i) if maxvals[0][num] > threshold else [None, None] for num, i in enumerate(preds[0]) ]
            labels = [i if maxvals[0][num] > threshold else None for num, i in enumerate(self.K_namespace[idx].labels) ]
            return image, coords, labels

    def _box2cs(self, box, image_width, image_height):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h, image_width, image_height)

    def _xywh2cs(self, x, y, w, h, image_width, image_height):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        aspect_ratio = image_width * 1.0 / image_height
        pixel_std = 200

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    


    def get_final_preds(self, batch_heatmaps, center, scale):
        coords, maxvals = get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        preds = coords.copy()
        post_process = True
        if post_process == True:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = batch_heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px],
                            ]
                        )
                        coords[n][p] += np.sign(diff) * 0.25

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

        return preds, maxvals


# 크롭, 회전, 밝기 조절 
class preprocessImage():
    
    def crop(self, image, coord, keypoints = None): 
        """
        
        이미지에 대한 크롭을 진행한 후 크롭 후의 이미지, 키포인트를 리턴하는 함수
        image (numpy) : input image
        coord (list) : [x0, y0, x1, y1] 
        keypoints (list) : keypoint 도 전처리 할 경우 사용([[x0, y0], [x1, y1], ....])
        """
        height, width, _ = image.shape
        b_l = 0; b_r = 0; b_t = 0; b_b = 0 # border_left, right, top, bottom
        if coord[0] < 0:
            b_l = abs(coord[0])
        if coord[1] < 0:
            b_t = abs(coord[1])
        if coord[2] > width:
            b_r = abs(coord[2] - width)
        if coord[3] > height:
            b_b = abs(coord[3] - height)
        cropped = cv2.copyMakeBorder(image, int(b_t), int(b_b), int(b_l), int(b_r), borderType=cv2.BORDER_CONSTANT)
        cropped = cropped[int(coord[1]) + int(b_t): int(coord[3]) + int(b_t), int(coord[0]) + int(b_l): int(coord[2]) + int(b_l)]
        show_cropped = cropped.copy()

        if keypoints != None:
            cropped_keypoints = []
            for keypoint in list(keypoints):
                if keypoint != None:
                    cropped_keypoints.append([keypoint[0] - coord[0] , keypoint[1] - coord[1] ])
                    cv2.circle(show_cropped, (int(keypoint[0] - coord[0] ), int(keypoint[1] - coord[1] )), 20, 255, -1)
                else:
                    cropped_keypoints.append(None)
            return cropped, cropped_keypoints     
        else: return cropped    
        
    def saveImage(self, image, folder, filename):
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(filename, image)
    
    def flip(self, image, axis, keypoints = None):
        """
        이미지에 대한 flip 을 진행 한 후 flip 후의 이미지, 키포인트를 리턴하는 함수
        image (numpy) : input image
        axis (int) : flip 방향 (1 - 좌우 반전, 0 - 상하반전)
        keypoints (list) : keypoint 도 전처리 할 경우 사용([[x0, y0], [x1, y1], ....])
        """
        height, width, _ = image.shape
        centerX = width/2 ; centerY = height/2
        flipped = cv2.flip(image, axis)
        show_flipped = flipped.copy()
        if keypoints != None:

            flipped_keypoints = []
            for keypoint in keypoints:
                if keypoint != None:
                    if axis == 1:
                        pointX = 2*centerX -keypoint[0]
                        pointY = keypoint[1]
                    else:
                        pointX = keypoint[0]
                        pointY = 2 * centerY - keypoint[1]
                    flipped_keypoints.append([pointX, pointY])
                    cv2.circle(show_flipped, (int(pointX), int(pointY)), 20,255,-1)

            return flipped, flipped_keypoints
        else:
            return flipped
        
    def rotate(self, image, mid, degree, keypoints = None):
        """
        이미지에 대한 rotate 를 진행 한 후 rotate 후의 이미지, 키포인트를 리턴하는 ㅎ마수
        image (numpy) : input image
        mid (tuple) : 중심 회전 좌표(x, y)
        degree (float) : 회전 각도
        keypoints (list) : keypoint 도 전처리 할 경우 사용([[x0, y0], [x1, y1], ....])
        """
        t = time.time()
        radian = math.radians(-degree)
        height, width, _ = image.shape
        M = cv2.getRotationMatrix2D((mid), degree, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height))
        show_rotated = rotated.copy()
        if keypoints != None:
            rotated_keypoints = []
            for keypoint in keypoints:
                pointX = mid[0] + math.cos(radian) * (keypoint[0] - mid[0]) - math.sin(radian) * (keypoint[1] - mid[1])
                pointY = mid[1] + math.sin(radian) * (keypoint[0] - mid[0]) + math.cos(radian) * (keypoint[1] - mid[1])
                rotated_keypoints.append([pointX, pointY])

                cv2.circle(show_rotated, (int(pointX), int(pointY)), 20,255,-1)
            return rotated, rotated_keypoints
        else:
            return rotated

    def draw_keypoint(self, image, coord, color):
        
        for num, c in enumerate(coord):
            if c !=[None, None]:
                cv2.circle(image, (int(c[0]), int(c[1])), 10, color, 3)
                cv2.putText(image, str(num), (int(c[0]), int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return image
    def clahe_image(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        image = clahe.apply(image[:, :, 0])
        image = np.stack([image, image, image], axis=2)

        return image
        
    def draw_polylines(self, image, poly_coords):
        image_output = image.copy()
        for coords in poly_coords:

            if (coords is not None) and (coords != ''):
                image_output = cv2.polylines(image_output, [np.array(coords, np.int32)], True, (255, 255, 255), 2)

        return image_output



class makelabelmeFile():
    def makeStructure(self, image, imagename):
        # labelme json 파일 틀 만들기
        height, width, _ = image.shape
        json_file = {}
        json_file['version'] = '5.2.1'
        json_file['flags'] = {}
        json_file['shapes'] = []
        json_file['imagePath'] = imagename
        json_file['imageData'] = None
        json_file['imageHeight'] = height
        json_file['imageWidth'] = width
        return json_file
        

    def addKeypoints(self, json_file, coords, labels, filename):
        """
        전처리 후의 좌표 json 파일에 쓰기
        """
        for coord, label in zip(coords, labels):
            if coord == None:
                continue
            # if coord[0] < 0:
            #     print('문제있음1', filename)
            #     continue
            # if coord[1] < 0:
            #     print('문제있음2', filename)

            #     continue
            # if coord[0] > json_file['imageWidth']:
            #     print('문제있음3', filename)

            #     continue
            # if coord[1] > json_file['imageHeight']:
            #     print('문제있음4', filename)
            #     continue

            json_shape = {}
            json_shape['label'] = label
            json_shape['points'] = [[float(coord[0]), float(coord[1])]]
            json_shape['group_id'] = None
            json_shape['description'] = ''
            json_shape['shape_type'] = 'point'
            json_shape['flags'] = {}
            json_file['shapes'].append(json_shape)
        return json_file

    def saveJson(self, json_file, outFolder, filename):
        os.makedirs(outFolder, exist_ok=True)
        with open(f'{outFolder}/{filename}.json', 'w') as save_file:
            json.dump(json_file, save_file, indent=2)

    def saveImage(self, outFolder, image, filename):
        cv2.imwrite(f'{outFolder}/{filename}.png', image)

# def saveImage(image, savePath, keypoints = None):
#     cv2.imwrite(savePath, image)

class readlabelmeFile():
    def readjson(self, filename):
        with open(filename, 'r') as f:
            json_data = json.load(f)
        return json_data
    
    # labelme 파일 넣기
    def labelAndCoord(self, json_data, labels):
        """
        input
        filename : labelme(json) 파일명
        image : 이미지 파일 명
        labels : 라벨 리스트

        ['라벨1' : [x좌표, y좌표]], 원본이미지

        # 문제점 : keypoint에서 좌표 2개 이상 같은 라벨로 라벨링 돼있을 경우 하나 날아감
        """
        key_dict = {key : None for key in labels}
        for shape in json_data['shapes']:

            if shape['shape_type'] == 'point':
                key_dict[shape['label']] = shape['points'][0]

        return key_dict
    
class calculation():
    def interval_calculation(self, coordinates, interval = False):
        idxes = [index for index, coord in enumerate(coordinates) if coord != [None, None]]
        x_interval = (coordinates[max(idxes)][0] - coordinates[min(idxes)][0]) / (max(idxes) - min(idxes))
        y_interval = (coordinates[max(idxes)][1] - coordinates[min(idxes)][1]) / (max(idxes) - min(idxes))
        xs = coordinates[min(idxes)][0] - x_interval * min(idxes)
        ys = coordinates[min(idxes)][1] - y_interval * min(idxes)
        xe = coordinates[max(idxes)][0] + x_interval * (len(coordinates)-1 - max(idxes))
        ye = coordinates[max(idxes)][1] + y_interval * (len(coordinates)-1 - max(idxes))
        if interval == False:
            return xs, ys, xe, ye
        else:
            return xs, ys, xe, ye, x_interval, y_interval
    def rotate_points(self, c, points, angle):
        # 특정 점을 중심으로 특정 각도만큼 회전시키는 코드
        rotated_coords = []
        for p in points:
            rx = c[0] + math.cos(math.radians(angle)) * (p[0] - c[0]) - math.sin(math.radians(angle)) * (p[1] - c[1])
            ry = c[1] + math.sin(math.radians(angle)) * (p[0] - c[0]) + math.cos(math.radians(angle)) * (p[1] - c[1])
            rotated_coords.append([int(rx), int(ry)])

        return rotated_coords
    
    def check_flipped(self, coordinates, count=2):
        flips = 0

        for small, big in coordinates:
            if small is not None and big is not None:
                if small > big:
                    flips += 1
        if flips >= count:
            return True
        else:
            return False
        
class CSVfile():
    def readCSVfile(self, filename):
        data = pd.read_csv(filename)
        return data
    
    def writeCSVfile(self, content, filename):
        f = open(f'{filename}.csv', 'w', newline="")
        wr = csv.writer(f)
        wr.writerows(content)
        f.close()

class statistics():
    def get_outlier(self, data, multiply_value = 1.5):
        sorted_data = np.sort(data)
        # 중앙값
        median = np.median(sorted_data)
        # Q1 (1st Quartile)
        q1 = np.percentile(sorted_data, 25)
        # Q3 (3rd Quartile)
        q3 = np.percentile(sorted_data, 75)
        # IQR (Interquartile Range)을 계산합니다.
        iqr = q3 - q1

        lower_bound = q1 - multiply_value * iqr
        upper_bound = q3 + multiply_value * iqr
        # 이상치를 식별하고 출력합니다.
        outliers = [(num, value) for num, value in enumerate(data) if value < lower_bound or value > upper_bound]
        return outliers, lower_bound, upper_bound


# 0 : gpu, 1 : pretrained, 2 : trained, 3 : num_classes, 4 : img size

def getfilename(path, dirs = False, exts = False):
    if exts == True:
        if dirs == False:
            filename = os.path.basename(path)

    elif exts == False:
        if dirs == False:
            filename =  os.path.splitext(os.path.basename(path))[0]
        else:
            filename =  os.path.splitext(path)[0]
    return filename

def find_farthest_points(coordinates):
    # (0, 0)을 제외한 좌표 배열 생성
    filtered_coordinates = np.array([coord for coord in coordinates if coord != (0.0, 0.0)])
    
    # 모든 좌표 쌍 간의 거리 계산
    distances = cdist(filtered_coordinates, filtered_coordinates, metric='euclidean')
    
    # 가장 먼 거리와 해당 좌표 쌍 인덱스 찾기
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    return filtered_coordinates[i], filtered_coordinates[j]

# Keymodels = \
#     [
#         [
#             0,
#             'pose_hrnet',
#             '/home/jiwoo/Desktop/code_room/use_models/lumber_1st.pt',
#             12,
#             384,
#             288,
#             ['l1c','l2c','l3c','l4c','l5c','l6c','l1e','l2e','l3e','l4e','l5e','l6e']
#         ],
#         [
#             0,
#             'pose_hrnet',
#             '/home/jiwoo/Desktop/code_room/use_models/lumber_2nd.pt',
#             12,
#             384,
#             288,
#             ['l1c','l2c','l3c','l4c','l5c','l6c','l1e','l2e','l3e','l4e','l5e','l6e']
#         ]
#     ]


# CLS = classification(CLSmodels)
# KEY = keypointDetection(Keymodels)
# PREP = preprocessImage()
# # MLM = makelabelmeFile()
# RLM = readlabelmeFile()
# CAL = calculation()
# CSV = CSVfile()

# CLSmodels = \
#     [
#         [
#             0,
#             'SwinT_l_384_21k',
#             '/home/jiwoo/Desktop/code_room/use_models/SwinT_l_384_21k_down_93_0.6048014178983597.pt',
#             4,
#             384
#         ],
#     ] 
# coords : [(), (), ()], labels : ['라벨1', '라벨2']...
# image, coords,labels = KEY.keypoint_detect('00231956_2008-01-10_RSC04903_C_1.png', 0)

# cv2.imwrite('result.jpg', image)


# output, pred = CLS.keypoint_detect('00231956_2008-01-10_RSC04903_C_1.png', 0)


# image, coords = PREP.crop(image, [100,100,2000,2000], coords)
# image, coords = PREP.flip(image, 1, coords)
# image, coords = PREP.rotate(image, (1230,1485), 30, coords)

# MLM.makeStructure(image, '00231956_2008-01-10_RSC04903_C_1.png')
# MLM.addKeypoints(coords, labels)
# MLM.saveJson('output', image)


# inputFolder, labellist, outputFolder
# folder = 'train'
# json_files = sorted(paths.list_files(f'{folder}', validExts='json'))
# for json_file in tqdm(json_files):
#     filename = json_file.split('/')[-1].split('.json')[0]
#     key_dict, image = RLM.labelAndCoord(f'{json_file}',f'{folder}/{filename}.png', ['c1c','c2c','c3c','c4c','c5c','c6c','c1e','c2e','c3e','c4e','c5e','c6e'])

#     image, coords,labels = KEY.keypoint_detect(f'{folder}/{filename}.png',0, 0.1)
#     xs, ys, xe, ye = CAL.interval_calculation(coords)
#     coords.extend([(xs, ys), (xe, ye)])

#     xcoords = [i[0] for i in coords if i != None]
#     ycoords = [i[1] for i in coords if i != None]

#     height = max(ycoords) - min(ycoords)
    
#     x0 = min(xcoords) - height * 0.3 
#     x1 = max(xcoords) + height * 0.3

#     y0 = min(ycoords) - height * 0.3
#     y1 = max(ycoords) + height * 0.3
#     image, coords = PREP.crop(image, [x0, y0, x1, y1], key_dict.values())
#     MLM.makeStructure(image, f'{filename}.png')
#     MLM.addKeypoints(coords, key_dict.keys(), filename)
    
    # MLM.saveJson('output', image)


# for json_fil2023.09.10 (일) 09:20	
#     key_dict, image = RLM.labelAndCoord(f'{json_file}',f'{folder}/{filename}.png', ['c1c','c2c','c3c','c4c','c5c','c6c','c1e','c2e','c3e','c4e','c5e','c6e'])
#     xs, ys, xe, ye = CAL.interval_calculation(dict(list(key_dict.items())[0:6]), 0, 5)
#     Xcoords = [i[0] for i in key_dict.values() if i != None]
#     Xcoords.extend([xs, xe])
#     height = abs(ye - ys)
#     image, coords = PREP.crop(image, (int(min(Xcoords) - height*0.3), int(ys - height * 0.3), int(max(Xcoords) + height*0.3), int(ye + height * 0.3)), key_dict.values())
#     MLM.makeStructure(image, f'{filename}.png')
#     MLM.addKeypoints(coords, key_dict.keys(), filename)
#     MLM.saveJson('output', image)


# for json_file in json_files:
#     filename = json_file.split('/')[-1].split('.json')[0]

#     key_dict, image = RLM.labelAndCoord(f'{json_file}',f'{folder}/{filename}.png', ['c1c','c2c','c3c','c4c','c5c','c6c','c1e','c2e','c3e','c4e','c5e','c6e'])
#     xs, ys, xe, ye = CAL.interval_calculation(dict(list(key_dict.items())[0:6]), 0, 5)
#     Xcoords = [i[0] for i in # for json_file in json_files:
#     filename = json_file.split('/')[-1].split('.json')[0]

#     key_dict, image = RLM.labelAndCoord(f'{json_file}',f'{folder}/{filename}.png', ['c1c','c2c','c3c','c4c','c5c','c6c','c1e','c2e','c3e','c4e','c5e','c6e'])
#     xs, ys, xe, ye = CAL.interval_calculation(dict(list(key_dict.items())[0:6]), 0, 5)
#     Xcoords = [i[0] for i in key_dict.values() if i != None]
#     Xcoords.extend([xs, xe])
#     height = abs(ye - ys)
#     image, coords = PREP.crop(image, (int(min(Xcoords) - height*0.3), int(ys - height * 0.3), int(max(Xcoords) + height*0.3), int(ye + height * 0.3)), key_dict.values())
#     MLM.makeStructure(image, f'{filename}.png')
#     MLM.addKeypoints(coords, key_dict.keys())
#     MLM.saveJson('output', image)key_dict.values() if i != None]
#     Xcoords.extend([xs, xe])
#     height = abs(ye - ys)
#     image, coords = PREP.crop(image, (int(min(Xcoords) - height*0.3), int(ys - height * 0.3), int(max(Xcoords) + height*0.3), int(ye + height * 0.3)), key_dict.values())
#     MLM.makeStructure(image, f'{filename}.png')
#     MLM.addKeypoints(coords, key_dict.keys())
#     MLM.saveJson('output', image)


# for json_file in json_files:
#     filename = json_file.split('/')[-1].split('.json')[0]

#     key_dict, image = RLM.labelAndCoord(f'{json_file}',f'{folder}/{filename}.png', ['c1c','c2c','c3c','c4c','c5c','c6c','c1e','c2e','c3e','c4e','c5e','c6e'])
#     # print()   
#     print(key_dict)
#     image, coords = PREP.flip(image, 1, key_dict.values()) 
#     key_X_none = [key for key, value in key_dict.items() if value is not None]
#     MLM.makeStructure(image, f'{filename}.png')

#     MLM.addKeypoints(coords, key_X_none)
#     MLM.saveJson('output', image)

