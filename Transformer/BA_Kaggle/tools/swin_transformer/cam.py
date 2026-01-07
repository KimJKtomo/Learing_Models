import os
import cv2
import numpy as np

import albumentations as A
import imgaug.augmenters as iaa
from albumentations.pytorch.transforms import ToTensorV2

from pytorch_grad_cam import HiResCAM

from datamodel.roi import RoiFileMap
from .image_utils import padding
from utils.timer import inference_timer


# define transforms
transforms = A.Compose([A.Resize(384, 384), A.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()])


def heatmap_filter(grayscale_cam, image, image_normalization=True, transparency=0.01, blue_channel_ratio=0.6, blur_kernel=39, contrast_type='gamma', contrast_value=1.15):
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
    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # apply blue channel filter
    if blue_channel_ratio and blue_channel_ratio < 1.:
        heatmap[:, :, 2] = np.uint8(heatmap[:, :, 2]*blue_channel_ratio)  # RGB
    heatmap = np.float32(heatmap) / 255.

    # heatmap blending
    cam_visual = transparency * image + (1-transparency) * heatmap
    cam_visual = cam_visual / np.max(cam_visual)
    cam_visual = np.uint8(cam_visual * 255.)
    cam_visual = cv2.cvtColor(cam_visual, cv2.COLOR_BGR2RGB)

    return image, heatmap, cam_visual


# height, width = model window
def reshape_transform(tensor, height=24, width=24):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


@inference_timer
def roi_cam(roi_bbox_dic, cls_model_set, cam_path, f_name, replace_img):

    for roi in roi_bbox_dic.keys():
        if roi_bbox_dic[roi] == []:
            file_prefix = os.path.splitext(os.path.basename(f_name))[0]
            cv2.imwrite(cam_path +
                        file_prefix + "_" + getattr(RoiFileMap, roi) + '_cam.jpg', replace_img)
        else:
            roi_crop_img = roi_bbox_dic[roi][1]
            if roi in ['carpal']:
                roi_crop_img = padding(roi_crop_img)
            h, w = roi_crop_img.shape[:2]

            # define & load model (Swin Transformer)
            model = cls_model_set[roi][0]
            model.eval()

            # target layers
            target_layers = []
            for k, v in model.model.named_modules():
                if 'norm' in k and 'layers.2' in k and 'blocks' in k:
                    target_layers.append(v)

            # Construct the CAM object once, and then re-use it on many images
            cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=True,
                           reshape_transform=reshape_transform)

            # Create an input tensor image for model
            input_tensor = transforms(image=roi_crop_img)['image']
            input_tensor = input_tensor.unsqueeze(dim=0)

            # You can also pass aug_smooth=True, and eigen_smooth=True,
            # to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=None,
                                eigen_smooth=False, aug_smooth=False)
            grayscale_cam = grayscale_cam[0, :]
            del input_tensor

            roi_crop_img = A.Resize(384, 384)(image=roi_crop_img)['image']
            roi_crop_img, _, cam_visual = heatmap_filter(
                grayscale_cam, roi_crop_img)

            # save cam image
            cam_visual = A.Resize(h, w)(image=cam_visual)['image']
            file_prefix = os.path.splitext(os.path.basename(f_name))[0]
            cv2.imwrite(cam_path + 
                        file_prefix + "_" + getattr(RoiFileMap, roi) + "_cam.jpg", cam_visual)


@inference_timer
def gp_cam(gp_image, cls_model_set, gender, cam_path, f_name):

    gp_hand_image = padding(gp_image)
    h, w = gp_hand_image.shape[:2]

    # define & load model (Swin Transformer)
    model = cls_model_set[gender][0]
    model.eval()

    # target layers
    target_layers = []
    for k, v in model.model.named_modules():
        if 'norm' in k and 'layers.2' in k and 'blocks' in k:
            target_layers.append(v)

    # Construct the CAM object once, and then re-use it on many images
    cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=True,
                   reshape_transform=reshape_transform)

    # Create an input tensor image for model
    input_tensor = transforms(image=gp_hand_image)['image']
    input_tensor = input_tensor.unsqueeze(dim=0)

    # You can also pass aug_smooth=True, and eigen_smooth=True,
    # to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None,
                        eigen_smooth=False, aug_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    del input_tensor

    gp_hand_image = A.Resize(384, 384)(image=gp_hand_image)['image']
    gp_hand_image, _, cam_visual = heatmap_filter(
        grayscale_cam, gp_hand_image)

    # save cam image
    cam_visual = A.Resize(h, w)(image=cam_visual)['image']
    cam_visual = padding(cam_visual)
    file_prefix = os.path.splitext(os.path.basename(f_name))[0]
    cv2.imwrite(cam_path +
                file_prefix + "_cam.jpg", cam_visual)
