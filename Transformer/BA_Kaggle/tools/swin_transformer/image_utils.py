import cv2
import numpy as np


def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 3:
        img2 = clahe.apply(image[:, :, 0])
    else:
        img2 = clahe.apply(image[:, :])
    image = np.stack([img2, img2, img2], axis=2)
    return image


def normalize(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image


def padding(img):
    height, width = img.shape[0:2]
    margin = [np.abs(height - width)//2, np.abs(height-width)//2]

    if np.abs(height - width) % 2 != 0:
        margin[0] += 1

    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]
    
    if len(img.shape) == 3:
        margin_list.append([0, 0])
    
    img = np.pad(img, margin_list, mode='constant')
    return img
    