import traceback

import os
import pandas as pd
import torch
import argparse
import logging
from glob import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error
from imutils import paths
from tools.hrnet.keypoint_detection import load_keypoint_model
from tools.dn_dab_deformable_detr.text_detection import load_text_detection_model
from tools.dn_dab_deformable_detr.roi_detection import load_roi_detection_model
from tools.swin_transformer.roi_classification import load_cls_model
from tools.fc_model.fc_prediction import load_fc_reg_model, load_fc_cls_model
from tools.demo_handbone_for_smi import demo_inference

type = 2
def inference(args):

    # answer_table = pd.read_csv('/home/crescom/BA/data/smi_img_2/DENTAL_SMI_2_FLOOR.csv')
    # answer_table = pd.read_csv('data/dental_smi_fix_0517.csv')
    if 'severance' in args.test_name :
        print('Inference Severance Dataset')
        answer_table = pd.read_csv('/home/crescom/BA/data/severance_smi_validation/severance_validation.csv')

    if 'miso' in args.test_name :
        answer_table = pd.read_csv('/home/crescom/BA/data/miso_dental/miso_dental_3_0914_재판독.csv')

    if args.device == 'gpu0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    elif args.device == 'gpu1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    keypoint_model = load_keypoint_model(device=device, type=type)  # 1803MB
    text_model = load_text_detection_model(device=device)  # 2363 MB
    roi_model = load_roi_detection_model(device=device)  # 2363 MB
    cls_model_set = load_cls_model(device=device)  # 9989 MB
    fc_reg_model = load_fc_reg_model(device=device)  # 1689 MB
    fc_cls_model = load_fc_cls_model(device=device)  # 1687 MB
    # total model memory 11337 MB

    # Logging setting
    formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(args.save_path + f'{args.test_name}.log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    results = []

    if args.one_img == True :
        test_img = args.data_path
        other_drive = '/home/crescom/boneage-ai/test'
        d_name = ''
        f_name = test_img.split('/')[-1]

        ba, result_dict, smi_dict, smi_ba_dict, angle = demo_inference(
            args,
            test_img, device, args.gender,
            keypoint_model, text_model, roi_model, cls_model_set, fc_reg_model, fc_cls_model,
            d_name, f_name, other_drive)

        print(f'Predicted B-age = {ba}')
        print(f'Predicted ROI B-age = {result_dict}')
        print(f'Predicted SMI = {smi_dict}')

    else :
        for gender in ['female', 'male']:
            test_img_list = glob(f'{args.data_path}/{gender}/*')
            print(f'{args.data_path}/{gender}/*')

            if gender == 'female':
                args.gender = 'f'
            else:
                args.gender = 'm'
            print(args.gender)
            for test_img in tqdm(test_img_list):





                try:
                    other_drive = '/home/crescom/boneage-ai/test'
                    d_name = ''
                    f_name = test_img.split('/')[-1]
                    # if f_name not in checkfiles:
                    #     continue




                    ba, result_dict, smi_dict, smi_ba_dict, angle = demo_inference(
                        args,
                        test_img, device, args.gender,
                        keypoint_model, text_model, roi_model, cls_model_set, fc_reg_model, fc_cls_model,
                        d_name, f_name, other_drive, type)
                    
                    # demo_inference(
                    #     args,
                    #     test_img, device, args.gender,
                    #     keypoint_model, text_model, roi_model, cls_model_set, fc_reg_model, fc_cls_model,
                    #     d_name, f_name, other_drive, type)
                    # continue
                    if 'severance' in args.test_name :
                        test_name = int(test_img.split('/')[-1].replace('.jpg',''))
                    else :
                        test_name = test_img.split('/')[-1]

                    if 'miso' in args.test_name or 'severance' in args.test_name:
                        print(args.test_name)
                        results.append({'file_name': test_name,
                                        'gender': gender,
                                        'angle' : angle,
                                        'dp3': result_dict['dp3'][1],
                                        'dp3_smi': smi_dict['dp3'],
                                        'mp3': result_dict['mp3'][1],
                                        'mp3_smi': smi_dict['mp3'],
                                        'pp3': result_dict['pp3'][1],
                                        'pp3_smi': smi_dict['pp3'],
                                        'radius': result_dict['radius'][1],
                                        'radius_smi': smi_dict['radius'],
                                        'ulna': result_dict['ulna'][1],
                                        'carpal': result_dict['carpal'][1],
                                        'mc1': result_dict['mc1'][1],
                                        'pp1': result_dict['pp1'][1],
                                        'pp1_smi': smi_dict['pp1'],
                                        'mp5': result_dict['mp5'][1],
                                        'mp5_smi': smi_dict['mp5'],
                                        'gp': result_dict['gp'][1],
                                        'pred_ba': ba,
                                        'pred_smi': smi_dict['smi'],
                                        'doctor_smi': answer_table[answer_table['file_name']==test_name]['doctor_smi'].values[0]
                                        })
                    else :
                        results.append({'file_name': test_name,
                                        'gender': gender,
                                        'angle' : angle,
                                        'dp3': result_dict['dp3'][1],
                                        'dp3_smi': smi_dict['dp3'],
                                        'mp3': result_dict['mp3'][1],
                                        'mp3_smi': smi_dict['mp3'],
                                        'pp3': result_dict['pp3'][1],
                                        'pp3_smi': smi_dict['pp3'],
                                        'radius': result_dict['radius'][1],
                                        'radius_smi': smi_dict['radius'],
                                        'ulna': result_dict['ulna'][1],
                                        'carpal': result_dict['carpal'][1],
                                        'mc1': result_dict['mc1'][1],
                                        'pp1': result_dict['pp1'][1],
                                        'pp1_smi': smi_dict['pp1'],
                                        'mp5': result_dict['mp5'][1],
                                        'mp5_smi': smi_dict['mp5'],
                                        'gp': result_dict['gp'][1],
                                        'pred_ba': ba,
                                        'pred_smi': smi_dict['smi'],
                                        })

                except Exception as e:
                    print(traceback.format_exc())
                    logging.exception(
                        f"id={d_name + f_name} message={e}")
                    pass

        result_df = pd.DataFrame(results)

        result_df.to_csv(f'{args.save_path}{args.test_name}_inference_results.csv', index=False)
        print('Saved in :', args.save_path)

        if 'miso' in args.test_name or 'severance' in args.test_name :
            result_df['diff'] = abs(result_df['pred_smi'] - result_df['doctor_smi'])
            print(round(mean_absolute_error(result_df['doctor_smi'], result_df['pred_smi']),3))
            print(round(np.mean(result_df['diff']),3))
            print(np.mean(result_df['diff']))
            print(result_df.groupby('gender')['diff'].mean())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # device
    parser.add_argument(
        "--device", type=str, default="gpu1", help="device type (default: gpu0)")

    # path & dataset
    parser.add_argument(
        "--data_path", type=str, default='/home/jiwoo/Desktop/Company_projects/BA_test-main/what', help="dataset path")

    parser.add_argument(
        "--save_path", type=str, default=f'result{type}', help="save path for probs json file")
    parser.add_argument(
        "--test_name", type=str, default='test', help="dataset mode(train/test)")
    parser.add_argument(
        "--one_img", action="store_true", help="Test one img")
    parser.add_argument(
        "--gender", type=str, default='f')
    parser.add_argument(
        "--correction", action="store_true", help="Repeat smi correction algorithm")

    args = parser.parse_args()
    args.save_path = f'{args.save_path}/{args.test_name}/'
    os.makedirs(args.save_path, exist_ok=True)  
    print(args)

    inference(args)

# one img code
# python test_for_smi.py --data_path /home/crescom/BA/data/smi_img/dental_smi_img/female/145.jpg --gender f --one_img