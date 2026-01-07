import json
import pandas as pd

class Table:
    # ROI별 Class별 평균 확률분포 TABLE
    with open('tools/table/BA_AVERAGE_PROB_TABLE.json', 'r') as f:
        ba_average_prob_table = json.load(f)

    # Outlier 기준 Table
    with open('tools/table/OUTLIER_DETECTION.json', 'r') as f:
        outlier_table = json.load(f)

    # ROI female b-age -> male b-age Mapping Table
    with open('tools/table/ROI_BA_MAPPING_TABLE.json', 'r') as f:
        female_mapping_table = json.load(f)

    # ROI female b-age -> male b-age Mapping Table
    male_mapping_table = pd.read_csv('tools/table/ROI_BA_MAPPING_TABLE.csv')

    # ROI b-age -> SMI 단계 Mapping Table
    with open('tools/table/BA_SMI_MAPPING.json', 'r') as f:
        smi_mapping_table = json.load(f)

    # SMI 단계별 b-age 기준 Table
    with open('tools/table/SMI_BA_TABLE.json') as f:
        smi_standard_ba_table = json.load(f)