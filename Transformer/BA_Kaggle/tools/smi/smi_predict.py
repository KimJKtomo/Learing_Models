import logging

def predicted_smi(gender, roi_ba_result_dict, pred_ba, ROI_BA_SMI_MAPPING_TABLE, SMI_BA_TABLE):

    if gender == 'f':
        gender = 'female'
    else :
        gender = 'male'

    if -1 in roi_ba_result_dict.values() :
        smi_result_dict = {}
        smi_result_dict['smi'] = ''

    else :
        # SMI Mapping
        smi_result_dict = {}
        smi_result_dict ={'dp3': -1, 'mp3': -1, 'pp3': -1, 'radius': -1, 'pp1': -1, 'mp5': -1}

        for roi in ['dp3', 'mp3', 'pp3', 'radius', 'pp1', 'mp5']:
            smi_result_dict[roi] = ROI_BA_SMI_MAPPING_TABLE[roi][str(float(roi_ba_result_dict[roi]))]

        pred_smi = max(smi_result_dict.values())
        smi_values = sorted(smi_result_dict.values())

        logging.info(
            f"SMI result dict = {smi_result_dict}")

        if pred_smi == 7:
            if smi_values[-2] < 3:
                pred_smi = 3

        # SMI upper
        if pred_smi in [0, 1, 4, 5, 7, 8, 9, 10]:
            if pred_ba > SMI_BA_TABLE['BA'][gender][str(pred_smi + 1)][0] + SMI_BA_TABLE['BA'][gender][str(pred_smi + 1)][2]:
                if roi_ba_result_dict[SMI_BA_TABLE['ROI_BA'][str(pred_smi + 1)][0]] == (SMI_BA_TABLE['ROI_BA'][str(pred_smi + 1)][1] - 0.5):
                    pred_smi = pred_smi + 1

        # SMI 단계 하향 조정
        if pred_smi in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
            print('예측 골연령 =',pred_ba)
            print('기준 =',SMI_BA_TABLE['BA'][gender][str(pred_smi)][0] - (SMI_BA_TABLE['BA'][gender][str(pred_smi)][1] * 2))
            if pred_ba < SMI_BA_TABLE['BA'][gender][str(pred_smi)][0] - (SMI_BA_TABLE['BA'][gender][str(pred_smi)][1] * 2):
                if pred_smi  in [3, 7]:
                    pred_smi = smi_values[-2]
                else:
                    roi = SMI_BA_TABLE['ROI_BA'][str(pred_smi)][0]
                    roi_ba = SMI_BA_TABLE['ROI_BA'][str(pred_smi)][1]
                    pred_roi_ba = roi_ba_result_dict[roi]
                    if pred_roi_ba == roi_ba:
                        logging.info(
                            f"message=SMI Correction {pred_smi} to {smi_values[-2]}")
                        pred_smi = smi_values[-2]

        smi_result_dict['smi'] = pred_smi

        logging.info(
            f"SMI result dict = {smi_result_dict}")

    return smi_result_dict