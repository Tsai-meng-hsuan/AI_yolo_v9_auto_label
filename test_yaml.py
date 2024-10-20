import yaml
import pandas as pd
import os


def arrange_df(conf_filter_df):
    conf_filter_df = conf_filter_df.drop(columns=['conf', 'class'])
    new_df = pd.DataFrame({})
    new_df["class_name"] = conf_filter_df["class_name"]
    colume_name_list = [col for col in conf_filter_df.columns if col != "class_name"]
    for colume_name in colume_name_list:
        new_df[colume_name] = conf_filter_df[colume_name]
    return new_df


# 讀取 YAML 檔案
with open(r'E:\1082_mhtsai\B2408_AI_lable_tool\yolov9\gap_meter-6\data.yaml', 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)

# 顯示讀取的資料
print(yaml_data["names"])

data = {
    'x1': [1996, 1556, 2422, 2366, 1510, 971],
    'y1': [1921, 2190, 1645, 2187, 1645, 1748],
    'x2': [2044, 1620, 2491, 2534, 1669, 1124],
    'y2': [1964, 2236, 1692, 2232, 1692, 1813],
    'conf': [0.253492, 0.569397, 0.776721, 0.787404, 0.824773, 0.853442],
    'class': [3, 0, 0, 1, 1, 2]
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 定義字典映射
# class_mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
class_mapping = {}
for label_index in range(len(yaml_data["names"])):
    class_mapping[label_index] = yaml_data["names"][label_index]

# 新增一列
df['class_name'] = df['class'].map(class_mapping)

df = arrange_df(df)

# 顯示結果
print(df)

# 儲存為 TXT 檔案
df.to_csv('output.txt', sep='\t', header=False, index=False)

file_name = os.path.basename(r'E:\1082_mhtsai\B2408_AI_lable_tool\yolov9\gap_meter-6\data.yaml')
print(file_name)

df = pd.read_csv(r'E:\1082_mhtsai\B2408_AI_lable_tool\yolov9\runs\detect\output_image28\labels\IMG_1481.txt', sep=' ', header=None)
print(df)
# 定義要插入的新資料
new_data = [10, 20, 30, 40, 50, 60]  # 新資料列表

# 在第一列插入新資料
df.insert(0, 'new_column', new_data)

# 顯示更新後的 DataFrame
print(df)