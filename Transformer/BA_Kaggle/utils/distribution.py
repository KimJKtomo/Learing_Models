import json
import argparse
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', 500)


roi_b_age_df = pd.read_csv('data/ROI_BA_VER2.2_TABLE.csv')
roi_b_age_df['file_name'] = [file_name +
                             '.jpg' for file_name in roi_b_age_df['file_name']]

value_as = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
            10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]
value_as_ulna = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
                 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]
value_as_carpal = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
                   10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]

idx_list = [i for i in range(34)]
idx_list_ulna = [i for i in range(26)]
idx_list_carpal = [i for i in range(30)]

roi_list = ['dp3', 'mp3', 'pp3', 'radius', 'ulna', 'carpal', 'mc1', 'gp']


def json_to_dataframe(json_file_path):
    with open(json_file_path, 'r') as f:
        json_file = json.load(f)

    annotation_lst = []

    for annotation in json_file['annotations']:
        file_name = annotation['file_name']
        for roi in roi_list:
            if roi == 'ulna':
                idxs = idx_list_ulna
                class_list = value_as_ulna
            elif roi == 'carpal':
                idxs = idx_list_carpal
                class_list = value_as_carpal
            else:
                idxs = idx_list
                class_list = value_as

            for idx in idxs:
                try:
                    annotation_lst.append({
                        'file_name': file_name,
                        'img_set': file_name.split('_')[0],
                        'gender': roi_b_age_df[(roi_b_age_df['file_name'] == file_name)]['gender'].values[0],
                        'idx': class_list[idx],
                        'roi': roi,
                        'probs': annotation[roi][idx],
                        'predicted_class': class_list[np.argmax(annotation[roi])],
                        'answer_class': roi_b_age_df[(roi_b_age_df['file_name'] == file_name) &
                                                     (roi_b_age_df['roi'] == roi)]['ba'].values[0]
                    })
                except:
                    pass

    return pd.DataFrame(annotation_lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    args = parser.parse_args()
    print(args)

    data = json_to_dataframe(args.json_path)
    data.to_csv(args.save_csv)
