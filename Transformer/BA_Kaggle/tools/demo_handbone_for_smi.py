from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import math
import logging
import numpy as np
import json

from utils.timer import inference_timer
from .table.table_load import Table

from .hrnet.keypoint_detection import keypoint_detect, keypoint_detect2
from .dn_dab_deformable_detr.text_detection import text_detect
from .dn_dab_deformable_detr.roi_detection import roi_detect
from .swin_transformer.roi_classification import cls_predict
from .swin_transformer.image_utils import clahe, padding
from .fc_model.fc_prediction import ba_predict

from .smi.smi_predict import predicted_smi

from util.all import *
MLM = makelabelmeFile()


color_palette = [
    (0, 0, 255),   # Red
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255), # Magenta
    (255, 255, 0), # Cyan
    (0, 128, 0),   # Maroon
    (128, 0, 0),   # Olive
    (128, 0, 128),   # Navy
    (0, 128, 128), # Olive Green
    (128, 128, 0), # Purple
    (0, 0, 128), # Teal
    (0, 165, 255), # Orange
    (128, 128, 128), # Gray
    (255, 255, 255), # White
    (0, 0, 0),     # Black
    (203, 192, 255), # Pink
    (42, 165, 42),  # Brown
    (127, 255, 0),  # Spring Green
    (180, 130, 70),  # Steel Blue
    (71, 99, 255),   # Tomato
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green
    (113, 179, 60),  # Medium Sea Green

    
]




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

def cal_median(ba_result_dict):
    cls_result_lst = []
    for roi in ba_result_dict.keys():
        if ba_result_dict[roi] == []:
            continue
        if roi in ['mp5', 'pp1']:
            continue
        cls_result_lst.append(ba_result_dict[roi][1])

    median = round_25(np.median(cls_result_lst))
    return median

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
def demo_inference_make_keypoint(args, path, device, gender, keypoint_model, text_model,
                   roi_model, cls_model_set, fc_reg_model, fc_cls_model,
                   d_name, f_name,
                   other_drive, backup_drive=None):
    d_name = ''
    replace_img = cv2.imread('/home/jiwoo/Desktop/Company_projects/BA/hrnet/not_detected_240220_240313/211343_SEOUL_D_115_20240221_2223698_111118.jpg')
    im = cv2.imread(path)

    json_file = MLM.makeStructure(im, os.path.basename(path))
    _, img_ext = os.path.splitext(path)
    # calculate shape ratio
    if len(im.shape) == 2:
        h, w = im.shape
    else:
        h, w, _ = im.shape
    shape_ratio = int(round(w * 100 / h, 0))

    original_im = im

    # keypoint detection
    # im, draw_img, _, missing = keypoint_detect(im, keypoint_model, device)
    im, draw_img, point_save_list, missing = keypoint_detect2(im, keypoint_model, device)
    
    result_dict = {}
    point_list = [point_save_list[i:i+2] for i in range(len(point_save_list))[::2]]
    # top, middle, thumb
    json_file = MLM.addKeypoints(json_file, point_list, ['top', 'middle', 'thumb'], path)
    path_base = os.path.basename(path)
    MLM.saveJson(json_file, 'output_2.6_oflip', path_base.split(img_ext)[0])




@inference_timer
def demo_inference(args, path, device, gender, keypoint_model, text_model,
                   roi_model, cls_model_set, fc_reg_model, fc_cls_model,
                   d_name, f_name,
                   other_drive, type, backup_drive=None):
    d_name = ''

    im = cv2.imread(path)

    # calculate shape ratio
    if len(im.shape) == 2:
        h, w = im.shape
    else:
        h, w, _ = im.shape
    # shape_ratio = int(round(w * 100 / h, 0))

    # original_im = im

    # keypoint detection
    if type ==1:
        point_im, draw_img, point_save_list, missing, angle = keypoint_detect(im, keypoint_model, device)
    elif type==2:
        point_im, draw_img, point_save_list, missing, angle = keypoint_detect2(im, keypoint_model, device)

    # """ labelme 파일 만들기 """
    
    # json_file = MLM.makeStructure(im, os.path.basename(path))
    # _, img_ext = os.path.splitext(path)
    # point_list = [point_save_list[i:i+2] for i in range(len(point_save_list))[::2]]
    # json_file = MLM.addKeypoints(json_file, point_list, ['top', 'middle', 'thumb'], path)
    # path_base = os.path.basename(path)
    # MLM.saveJson(json_file, '240429_output_2.6_xflip_102', path_base.split(img_ext)[0])

    # """"""""""""""""""""""""""
    

    # if missing == True :

    # raw image
    raw_image = np.copy(im)

    # roi detection
    roi_bbox_dict, scores, boxes = roi_detect(
        point_im, roi_model, device)
    # save roi detection results
    for num, (label, box) in enumerate(zip(list(roi_bbox_dict.keys()), list(roi_bbox_dict.values()))):
        if box != []:
            cv2.rectangle(draw_img, (int(box[0][0]), int(box[0][1])), (int(box[0][2]), int(box[0][3])), color_palette[num], 3)
            cv2.putText(draw_img, label, (int(box[0][0]), int(box[0][1]-30)), cv2.FONT_HERSHEY_SIMPLEX,2, color_palette[num], 3)
    cv2.imwrite(os.path.join(args.save_path, f_name), draw_img)

    # cv2.imwrite(args.save_path + f'{f_name.replace(".jpg", "")}_bbox.jpg', bbox_img)

    # save roi images
    clahe_img = clahe(raw_image)
    for roi in roi_bbox_dict.keys():
        if roi_bbox_dict[roi] == []:
            file_prefix = os.path.splitext(os.path.basename(f_name))[0]
        else:
            x1, y1, x2, y2 = roi_bbox_dict[roi][0]
            roi_crop_img = clahe_img[y1:y2, x1:x2, :]
            if roi == 'carpal':
                roi_crop_img = padding(roi_crop_img)
            # if roi == 'pp1':
            #     os.makedirs(args.save_path+'pp1_not_clahe/', exist_ok=True)
            #     cv2.imwrite(args.save_path+f'pp1_not_clahe/{f_name.replace(".jpg","")}_pp1.jpg', raw_image[y1:y2, x1:x2, :])
            # file_prefix = os.path.splitext(os.path.basename(f_name))[0]
            # cv2.imwrite(args.save_path + f'{f_name.replace(".jpg", "")}_{roi}.jpg', roi_crop_img)

    # text detection
    im = text_detect(im, text_model, device)


    # roi classfication
    cls_result_dic = cls_predict(
        roi_bbox_dict, im, cls_model_set, gender, device, path, type)
    # print(
    #     f'ORIGINAL CLS RESULT DICT dp3:{cls_result_dic["dp3"][1]} mp3:{cls_result_dic["mp3"][1]} pp3:{cls_result_dic["pp3"][1]} radius:{cls_result_dic["radius"][1]} pp1:{cls_result_dic["pp1"][1]}')
    
    # SMI ROI dict
    median = cal_median(cls_result_dic)

    # Female : ROI male b-age -> female b-age mapping
    if gender == 'f':
        cls_result_dic = ba_mapping(
            cls_result_dic, Table.female_mapping_table, d_name, f_name)
        median = cal_median(cls_result_dic)

    # print(
    #     f'CLS RESULT DICT dp3:{cls_result_dic["dp3"][1]} mp3:{cls_result_dic["mp3"][1]} pp3:{cls_result_dic["pp3"][1]} radius:{cls_result_dic["radius"][1]}')

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
                                gender, roi, median, round_25(
                                    (median + cls_result_dic[roi][1]) / 2), cls_result_dic,
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
                                gender, roi, median, round_25(
                                    (median + cls_result_dic[roi][1]) / 2), cls_result_dic,
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
                                gender, roi, median, round_25(
                                    (median + cls_result_dic[roi][1]) / 2), cls_result_dic,
                                miss_and_outlier_dic, 'outlier')
                            logging.info(
                                f'id={d_name + f_name} message=Outlier1 roi={roi} median={median} predicted={cls_result_dic[roi][1]} replaced={miss_and_outlier_dic[roi][1]}')

    for roi in miss_and_outlier_dic.keys():
        replace_ba = float(miss_and_outlier_dic[roi][1])
        print(roi)
        print(replace_ba)

        if roi == 'pp1':
            cls_result_dic[roi] = [np.array([0.99, 0.01]), -1]
        elif roi == 'mp5':
            cls_result_dic[roi] = [np.array([0.9, 0.1, 0.0]), -1]
        else:
            if gender == 'f':
                if len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values) == 1:
                    male_replace_ba = Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values[0]
                    cls_result_dic[roi] = [
                        np.array(Table.ba_average_prob_table[roi][str(male_replace_ba)]), replace_ba]

                elif len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values) > 1:
                    male_replace_ba = max(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values)
                    cls_result_dic[roi] = [
                        np.array(Table.ba_average_prob_table[roi][str(male_replace_ba)]), replace_ba]

                elif len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values) == 0:
                    diff_list = abs(
                        Table.male_mapping_table[Table.male_mapping_table['roi'] == roi]['female_ba'] - replace_ba).tolist()
                    ba_idx = max(
                        list(filter(lambda x: diff_list[x] == min(diff_list), range(len(diff_list)))))
                    replace_ba = Table.male_mapping_table[Table.male_mapping_table['roi'] == roi]['female_ba'].tolist()[
                        ba_idx]
                    male_replace_ba = Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == replace_ba)]['ba'].values[0]
                    cls_result_dic[roi] = [
                        np.array(Table.ba_average_prob_table[roi][str(male_replace_ba)]), replace_ba]
                else:
                    raise Exception()

            else:
                cls_result_dic[roi] = [
                    np.array(Table.ba_average_prob_table[roi][str(replace_ba)]), replace_ba]

        # print('Outlier Replaced :',f'CLS RESULT DICT dp3:{cls_result_dic["dp3"][1]} mp3:{cls_result_dic["mp3"][1]} pp3:{cls_result_dic["pp3"][1]} radius:{cls_result_dic["radius"][1]}')

    # predict boneage
    ba_reg, ba_cls = ba_predict(
        cls_result_dic, fc_reg_model, fc_cls_model, gender, device)

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


    cls_result_dic_for_smi = {}

    for roi in cls_result_dic.keys():
        if gender == 'f':
            if roi not in ['pp1', 'mp5', 'gp']:
                female_ba = float(cls_result_dic[roi][1])
                if len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == female_ba)]['ba'].values) == 1:
                    replaced_male_ba = Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                            Table.male_mapping_table['female_ba'] == female_ba)]['ba'].values[0]

                elif len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == female_ba)]['ba'].values) > 1:
                    replaced_male_ba = max(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                            Table.male_mapping_table['female_ba'] == female_ba)]['ba'].values)

                elif len(Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                        Table.male_mapping_table['female_ba'] == female_ba)]['ba'].values) == 0:
                    diff_list = abs(
                        Table.male_mapping_table[Table.male_mapping_table['roi'] == roi][
                            'female_ba'] - female_ba).tolist()
                    ba_idx = max(
                        list(filter(lambda x: diff_list[x] == min(diff_list), range(len(diff_list)))))
                    female_ba = Table.male_mapping_table[Table.male_mapping_table['roi'] == roi]['female_ba'].tolist()[
                        ba_idx]
                    replaced_male_ba = Table.male_mapping_table[(Table.male_mapping_table['roi'] == roi) & (
                            Table.male_mapping_table['female_ba'] == female_ba)]['ba'].values[0]
                else:
                    raise Exception()
                cls_result_dic_for_smi[roi] = replaced_male_ba
            else:
                cls_result_dic_for_smi[roi] = cls_result_dic[roi][1]

        else:
            cls_result_dic_for_smi[roi] = cls_result_dic[roi][1]

    # predict SMI
    # TODO: 미검출 SMI ROI 있는 경우 smi_result_dict['smi']=''으로 따로 내보내야 함
    smi_result_dict = predicted_smi(gender, cls_result_dic_for_smi, ba_reg, Table.smi_mapping_table, Table.smi_standard_ba_table)
    print(ba_reg)
    print(f'CLS RESULT DICT dp3:{cls_result_dic["dp3"][1]} mp3:{cls_result_dic["mp3"][1]} pp3:{cls_result_dic["pp3"][1]} radius:{cls_result_dic["radius"][1]}')
    print(f'SMI RESULT DICT dp3:{cls_result_dic_for_smi["dp3"]} mp3:{cls_result_dic_for_smi["mp3"]} pp3:{cls_result_dic_for_smi["pp3"]} radius:{cls_result_dic_for_smi["radius"]}')
    print(f'smi_result_dict : {smi_result_dict}')
    return round(ba_reg, 2), cls_result_dic, smi_result_dict, cls_result_dic_for_smi, angle

