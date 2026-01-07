import argparse
import os
import cv2
import traceback
from tqdm import tqdm
from glob import glob

import torch

from tools.hrnet.keypoint_detection import load_keypoint_model, keypoint_detect
from tools.dn_dab_deformable_detr.roi_detection import load_roi_detection_model, roi_detect
from tools.dn_dab_deformable_detr.text_detection import load_text_detection_model, text_detect


except_list = []


def crop_img(args):
    # device
    if args.device == 'gpu0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    elif args.device == 'gpu1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load models
    keypoint_model = load_keypoint_model(device=device)  # 1803 MB
    roi_model = load_roi_detection_model(device=device)  # 2363 MB
    text_model = load_text_detection_model(device=device)  # 2363 MB

    for mode in ['train', 'test']:
        for gender in ['female', 'male']:
            img_list = glob(f'{args.data_path}/{mode}/{gender}/*')

            for img_path in tqdm(img_list):
                try:
                    img = cv2.imread(img_path)

                    # keypoint detection
                    img, _, _ = keypoint_detect(img, keypoint_model, device)

                    # roi detection
                    roi_bbox_dict, scores, boxes = roi_detect(
                        img, roi_model, device)

                    # text detection & remove
                    img = text_detect(img, text_model, device)
                    save_path = os.path.join(args.save_path, mode, gender)
                    file_name = os.path.splitext(os.path.basename(img_path))[0]
                    cv2.imwrite(save_path+'/'+file_name+'_gp.jpg', img)

                    # save roi images
                    # missing value -> pass
                    for k, v in roi_bbox_dict.items():
                        if v == []:
                            break

                    for roi in roi_bbox_dict.keys():
                        roi_crop_img = roi_bbox_dict[roi][1]
                        save_path = os.path.join(args.save_path, mode, gender)
                        file_name = os.path.splitext(
                            os.path.basename(img_path))[0]
                        cv2.imwrite(save_path+'/'+file_name +
                                    '_'+roi+'.jpg', roi_crop_img)

                except Exception as e:
                    error_message = str(traceback.format_exc())
                    except_list.append({'file_name': img_path,
                                        'error_message': error_message})
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument(
        "--device", type=str, default="gpu0", help="device type (default: gpu0)")

    # path & dataset
    parser.add_argument(
        "--data_path", type=str, default='/home/crescom/BA/data', help="dataset path")
    parser.add_argument(
        "--save_path", type=str, default='./results', help="save path for probs json file")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    print(args)

    crop_img(args)
