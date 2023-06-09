import os
import json
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

# 상수 선언
CSV_PATH = "/opt/ml/level2_cv_semanticsegmentation-cv-15/codebook/ensemble"
MAX_LINES = 29 * 300

# 파일 가져오기
csv_list = os.listdir(CSV_PATH)
df_list = []
for name in csv_list:
    df_list.append(pd.read_csv(os.path.join(CSV_PATH, name)).fillna(""))

# 예측 클래스 확인
for df in df_list:
    if len(df) != MAX_LINES:
        print("예측이 안된 클래스가 존재합니다! 다시 확인해 보세요")
        exit(0)

# 저장할 dataframe 생성
result_df = pd.read_csv(os.path.join(CSV_PATH, csv_list[0])).fillna("")

# ensemble voting
for line in tqdm(range(MAX_LINES)):
    pixel_list = []
    pred_list = []
    result = ""
    cnt = 0
    for df in df_list:
        if df["rle"][line] == "":
            continue
        cnt += 1
        split_numbers = list(map(int, df["rle"][line].split(" ")))
        for i in range(0, len(split_numbers), 2):
            for j in range(split_numbers[i+1]):
                pixel_list.append(j + split_numbers[i])
    pixel_counter = Counter(pixel_list)
    pixel_keys = pixel_counter.keys()
    for key in pixel_keys:
        if pixel_counter[key] >= cnt // 2:
            pred_list.append(key)
    pred_list.sort()

    idx = 0
    while idx < len(pred_list):
        result += "%d " % pred_list[idx]
        num = pred_list[idx]
        start_idx = idx
        while idx < len(pred_list) - 1 and num + 1 == pred_list[idx+1]:
            num += 1
            idx += 1
        idx += 1
        result += "%d " % (idx - start_idx)
    result = result[:-1]
    result_df["rle"][line] = result

result_df.to_csv("/opt/ml/level2_cv_semanticsegmentation-cv-15/codebook/ensemble.csv", index=False)

        