from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import json
import math
import logging
import numpy as np
import albumentations as A

from utils.timer import inference_timer

from .table.table_load import Table
from .hrnet.keypoint_detection import keypoint_detect
from .dn_dab_deformable_detr.text_detection import text_detect
from .dn_dab_deformable_detr.roi_detection import roi_detect
from .swin_transformer.roi_classification import cls_predict, file_roi_dict
from .swin_transformer.cam import roi_cam, gp_cam
from .swin_transformer.image_utils import clahe, padding
from .fc_model.fc_prediction import ba_predict


def round_25(x):
    if x - math.floor(x) < 0.5:
        if (x - math.floor(x)) // 0.25 == 1:
            x = math.floor(x) + 0.5
        else:
            x = math.floor(x)

    elif x - math.floor(x) > 0.5:
        if (x - math.floor(x)) // 0.75 == 1:
            x = math.floor(x) + 1
        else:
            x = math.floor(x) + 0.5

    return x


def replace_process(gender, roi, median, replaced_ba, cls_result_dic, miss_and_outlier_dic, replace_type):
    if replace_type == 'detection_error':
        original_value = median
    else:
        original_value = cls_result_dic[roi][1]

    if roi == 'ulna':
        if replaced_ba <= 5.5:
            replaced_ba = 5.5

        miss_and_outlier_dic[roi] = [original_value, replaced_ba]

    elif roi == 'carpal':
        if gender == 'f':
            if replaced_ba >= 14.5:
                replaced_ba = 14.5
            miss_and_outlier_dic[roi] = [original_value, replaced_ba]

        else:
            if replaced_ba >= 16:
                replaced_ba = 16
            miss_and_outlier_dic[roi] = [original_value, replaced_ba]

    elif roi == 'pp1':
        miss_and_outlier_dic[roi] = [original_value, -1]

    elif roi == 'mp5':
        miss_and_outlier_dic[roi] = [original_value, -1]

    else:
        miss_and_outlier_dic[roi] = [original_value, replaced_ba]

    return miss_and_outlier_dic


def ba_mapping(result_dict, female_mapping_table, d_name, f_name):
    for roi in result_dict.keys():
        if roi in ['pp1', 'mp5', 'gp']:
            continue
        else:
            if result_dict[roi] != []:
                male_bone_age = float(result_dict[roi][1])
                mapping_age = female_mapping_table[roi][str(float(result_dict[roi][1]))]
                result_dict[roi][1] = mapping_age
                logging.info(
                    f'id={d_name + f_name} message=FemaleBoneAgeMapping roi={roi} MaleBoneAge={male_bone_age} MappingBoneAge={result_dict[roi][1]}')

    return result_dict


@inference_timer
def demo_inference(path, device, gender, keypoint_model, text_model, roi_model,
                   cls_model_set, fc_reg_model, fc_cls_model, d_name, f_name,
                   other_drive, backup_drive=None):
    d_name = ''
    replace_img = cv2.imread('tools/table/missing_img.jpg')

    im = cv2.imread(path)

    # calculate shape ratio
    if len(im.shape) == 2:
        h, w = im.shape
    else:
        h, w, _ = im.shape
    shape_ratio = int(round(w * 100 / h, 0))

    # keypoint detection
    im, _, _ = keypoint_detect(im, keypoint_model, device)

    # raw image
    raw_image = np.copy(im)

    # roi detection
    roi_bbox_dict, scores, boxes = roi_detect(
        im, roi_model, device)

    # save roi images
    clahe_img = clahe(raw_image)
    for roi in roi_bbox_dict.keys():
        if roi_bbox_dict[roi] == []:
            file_prefix = os.path.splitext(os.path.basename(f_name))[0]
            cv2.imwrite(other_drive + "/statics/roid/" + d_name + file_prefix + "_" + file_roi_dict[roi] + '.jpg',
                        replace_img)
        else:
            x1, y1, x2, y2 = roi_bbox_dict[roi][0]
            roi_crop_img = clahe_img[y1:y2, x1:x2, :]
            if roi == 'carpal':
                roi_crop_img = padding(roi_crop_img)
            file_prefix = os.path.splitext(os.path.basename(f_name))[0]
            cv2.imwrite(other_drive + "/statics/roid/" + d_name +
                        file_prefix + "_" + file_roi_dict[roi] + '.jpg', roi_crop_img)

    # text detection
    im = text_detect(im, text_model, device)
    cv2.imwrite(other_drive + '/statics/Uploads/' + d_name + '/' + f_name, im)

    # roi classfication
    cls_result_dic = cls_predict(
        roi_bbox_dict, im, cls_model_set, gender, device)

    original_ba = ba_predict(
        cls_result_dic, fc_reg_model, fc_cls_model, gender, device)[0]

    if gender == 'f':
        cls_result_dic = ba_mapping(cls_result_dic, Table.female_mapping_table, d_name, f_name)

    # calculate median
    cls_result_lst = []
    for roi in cls_result_dic.keys():
        if cls_result_dic[roi] == []:
            continue
        if roi in ['mp5', 'pp1']:
            continue
        cls_result_lst.append(cls_result_dic[roi][1])

    median = round_25(np.median(cls_result_lst))

    # Replace Missing value and Outlier
    miss_and_outlier_dic = {}

    for roi in roi_bbox_dict.keys():
        if roi_bbox_dict[roi] == []:
            miss_and_outlier_dic = replace_process(
                gender, roi, median, median, cls_result_dic, miss_and_outlier_dic, 'detection_error')
            logging.info(
                f'id={d_name + f_name} message=DetectionFailed roi={roi} replaced={miss_and_outlier_dic[roi][1]}')

        else:
            if roi in ['pp1', 'mp5', 'gp']:
                continue

            else:
                outlier = Table.outlier_table[gender][roi][str(float(median))]

                if roi == 'dp3' and median >= 11:
                    if abs(float(cls_result_dic[roi][1]) - median) >= 3:
                        if abs(float(cls_result_dic[roi][1]) - median) >= outlier + 3:
                            miss_and_outlier_dic = replace_process(
                                gender, roi, median, median, cls_result_dic, miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier2 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')
                        else:
                            miss_and_outlier_dic = replace_process(
                                gender, roi, median, round_25((median + cls_result_dic[roi][1]) / 2), cls_result_dic,
                                miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier1 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')

                elif roi == 'carpal':
                    if abs(float(cls_result_dic[roi][1]) - median) >= outlier + 1.5:
                        if abs(float(cls_result_dic[roi][1]) - median) >= outlier + 4.5:
                            miss_and_outlier_dic = replace_process(
                                gender, roi, median, median, cls_result_dic, miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier2 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')
                        else:
                            miss_and_outlier_dic = replace_process(
                                gender, roi, median, round_25((median + cls_result_dic[roi][1]) / 2), cls_result_dic,
                                miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier1 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')
                else:
                    if abs(float(cls_result_dic[roi][1]) - median) >= outlier:
                        if abs(float(cls_result_dic[roi][1]) - median) >= outlier + 2:
                            miss_and_outlier_dic = replace_process(
                                gender, roi, median, median, cls_result_dic, miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier2 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')
                        else:
                            miss_and_outlier_dic = replace_process(
                                gender, roi, median, round_25((median + cls_result_dic[roi][1]) / 2), cls_result_dic,
                                miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier1 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')

    for roi in miss_and_outlier_dic.keys():
        replace_ba = float(miss_and_outlier_dic[roi][1])

        if roi == 'pp1':
            cls_result_dic[roi] = [np.array([0.99, 0.01]), -1]
        elif roi == 'mp5':
            cls_result_dic[roi] = [np.array([0.9, 0.1, 0.0]), -1]
        else:
            if gender == 'f':
                if len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values) == 1:
                    male_replace_ba = Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                                Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values[0]
                    cls_result_dic[roi] = [np.array(Table.ba_average_prob_table[roi][str(male_replace_ba)]), replace_ba]

                elif len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values) > 1:
                    male_replace_ba = max(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                                Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values)
                    cls_result_dic[roi] = [np.array(Table.ba_average_prob_table[roi][str(male_replace_ba)]), replace_ba]

                elif len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values) == 0:
                    diff_list = abs(Table.male_mapping_table[Table.male_mapping_table['roi'] == roi][
                                        'female_ba'] - replace_ba).tolist()
                    ba_idx = max(list(filter(lambda x: diff_list[x] == min(diff_list), range(len(diff_list)))))
                    replace_ba = Table.male_mapping_table[Table.male_mapping_table['roi'] == roi]['female_ba'].tolist()[
                        ba_idx]
                    male_replace_ba = Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                                Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values[0]
                    cls_result_dic[roi] = [np.array(Table.ba_average_prob_table[roi][str(male_replace_ba)]), replace_ba]
                else:
                    raise Exception()

            else:
                cls_result_dic[roi] = [np.array(Table.ba_average_prob_table[roi][str(replace_ba)]), replace_ba]

    # roi cam
    roi_cam(roi_bbox_dict, cls_model_set,
            other_drive, d_name, f_name, replace_img)

    # gp cam
    gp_cam(im, cls_model_set, gender, other_drive, d_name, f_name)

    # predict boneage
    ba_reg, ba_cls = ba_predict(
        cls_result_dic, fc_reg_model, fc_cls_model, gender, device)

    # ulna predicted 5.5 class -> UI
    if 'ulna' in miss_and_outlier_dic.keys():
        if miss_and_outlier_dic['ulna'][0] == 5.5:
            outlier = Table.outlier_table[gender]['ulna'][str(float(median))]
            if abs(5.5 - median) < outlier + 2:
                cls_result_dic['ulna'][1] = 5.5

    if 'dp3' in miss_and_outlier_dic.keys():
        outlier = Table.outlier_table[gender]['dp3'][str(float(median))]
        if abs(abs(float(miss_and_outlier_dic['dp3'][0]) - median) < outlier + 3.5):
            cls_result_dic['dp3'][1] = miss_and_outlier_dic['dp3'][0]
            logging.info(
                f"id={d_name + f_name} message=Replaced_Original roi=dp3 replaced={miss_and_outlier_dic['dp3'][1]} original={miss_and_outlier_dic['dp3'][0]}")

    for roi, value in roi_bbox_dict.items():
        if value == []:
            if roi == "dp3":
                bbox_dp = []
            elif roi == "mp3":
                bbox_mp = []
            elif roi == "pp3":
                bbox_pp = []
            elif roi == "radius":
                bbox_ra = []
            elif roi == "ulna":
                bbox_ul = []
            elif roi == "mc1":
                bbox_mc = []
            elif roi == "carpal":
                bbox_car = []
            elif roi == "pp1":
                bbox_pp1 = []
            elif roi == "mp5":
                bbox_mp5 = []
        else:
            bbox = value[0]
            if roi == "dp3":
                bbox_dp = [int(bbox[0]), int(bbox[1]),
                           int(bbox[2]), int(bbox[3])]
            elif roi == "mp3":
                bbox_mp = [int(bbox[0]), int(bbox[1]),
                           int(bbox[2]), int(bbox[3])]
            elif roi == "pp3":
                bbox_pp = [int(bbox[0]), int(bbox[1]),
                           int(bbox[2]), int(bbox[3])]
            elif roi == "radius":
                bbox_ra = [int(bbox[0]), int(bbox[1]),
                           int(bbox[2]), int(bbox[3])]
            elif roi == "ulna":
                bbox_ul = [int(bbox[0]), int(bbox[1]),
                           int(bbox[2]), int(bbox[3])]
            elif roi == "mc1":
                bbox_mc = [int(bbox[0]), int(bbox[1]),
                           int(bbox[2]), int(bbox[3])]
            elif roi == "carpal":
                bbox_car = [int(bbox[0]), int(bbox[1]),
                            int(bbox[2]), int(bbox[3])]
            elif roi == "pp1":
                bbox_pp1 = [int(bbox[0]), int(bbox[1]),
                            int(bbox[2]), int(bbox[3])]
            elif roi == "mp5":
                bbox_mp5 = [int(bbox[0]), int(bbox[1]),
                            int(bbox[2]), int(bbox[3])]

    return round(original_ba, 2), round(ba_reg, 2), cls_result_dic
