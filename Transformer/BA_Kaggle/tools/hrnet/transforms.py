import cv2
import math
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def test_transforms():
    return A.Compose([
        A.Resize(384, 288),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def resize_transform(height, width):
    return A.Compose([
        A.Resize(height, width)
    ], keypoint_params=A.KeypointParams(format='xy'))


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
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


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def rotation(image, angleInDegrees, canvas):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)
    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))
    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])
    # outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_CUBIC)
    # canvas = cv2.warpAffine(canvas, rot, (b_w, b_h), flags=cv2.INTER_CUBIC)
    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    canvas = cv2.warpAffine(canvas, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg, canvas, b_w, b_h

def rotation_(img, angle, canvas):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    outImg = cv2.warpAffine(img, M, tuple(size_new.astype(int)), flags=cv2.INTER_CUBIC)
    canvas = cv2.warpAffine(canvas, M, tuple(size_new.astype(int)), flags = cv2.INTER_CUBIC)
    return outImg, canvas, outImg.shape[1], outImg.shape[0]



def crop_img(img, top, middle, thumb, w, h):
    mid_line = (top[0] + middle[0]) / 2
    gap_width = abs(thumb[0] - mid_line)
    left = mid_line - gap_width * 2.0
    right = mid_line + gap_width * 2.0

    if mid_line > thumb[0]:
        add_left = right
        add_right = left
    else:
        add_left = left
        add_right = right

    if add_left <= 0:
        add_left = 0
    if add_right >= w:
        add_right = w

    gap_height = abs(middle[1] - top[1])
    if top[1] > middle[1]:
        up = middle[1]
        mid = top[1]
    else:
        up = top[1]
        mid = middle[1]

    add_up = up - gap_height / 4
    add_down = mid + gap_height / 2
    # print(gap_height, add_up, add_down)

    if add_up <= 0:
        add_up = 0
    if add_down >= h:
        add_down = h

    crop_img = img[int(add_up):int(add_down), int(add_left):int(add_right)]
    return crop_img


def get_point(canvas):
    top = np.where((canvas == [0, 0, 255]).all(axis=2))
    middle = np.where((canvas == [0, 255, 0]).all(axis=2))
    thumb = np.where((canvas == [255, 0, 0]).all(axis=2))
    return [top[1][0], top[0][0]], [middle[1][0], middle[0][0]], [thumb[1][0], thumb[0][0]]

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """


    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25


    return center, scale


