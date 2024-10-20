import shutil
import os
import cv2
import yaml
import pandas as pd
from colorama import Fore, init

from yolov9 import detect



def read_init_label_txt(yolo_root, txt_folder):
    # 依據auto label資料彙整彙整結果
    detect_path = os.path.join(yolo_root, "runs", "detect")
    folders = os.listdir(detect_path)
    latest_dir = max(folders, key=lambda d: os.path.getmtime(os.path.join(detect_path, d))) #找到最新的資料夾
    label_path = os.path.join(detect_path, latest_dir, "labels")
    all_files = os.listdir(label_path)
    label_file = [file for file in all_files if file.endswith('.txt')][0]
    label_file_path = os.path.join(label_path, label_file) #這個是yolo v9 原生輸出的檔案
    init_txt_df = pd.read_csv(label_file_path, sep=' ', header=None)
    return init_txt_df


def drew_object(image_path, detection_df, yaml_path, show_image_size):
    # 定義框框的顏色和位置
    color_list = [(0, 0, 255), (0, 255, 0), (201, 42, 172), 
                  (255, 240, 142), (208, 168, 103), (171, 111, 220), 
                  (138, 115,38), (38, 205,38), (38, 205,38), (38, 205,38)]
    conf_list = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    #讀取yaml檔以對應物件名稱
    with open(yaml_path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    class_mapping = {}
    object_name_list = yaml_data["names"]
    for class_index in range(len(object_name_list)):
        class_mapping[class_index] = object_name_list[class_index]
    detection_df['class_name'] = detection_df['class'].map(class_mapping)

    for conf in conf_list:
        image = cv2.imread(image_path)
        # conf = float(conf)
        # 依據conf進行篩選
        conf_filter_df = detection_df[detection_df["conf"] >= conf]
        print(Fore.RED + f"detection conf: {conf}")
        print(conf_filter_df)
        # 抓取唯一的 class 值
        unique_classes = conf_filter_df['class'].unique()
        # 計算 class 的數量
        num_classes = len(unique_classes)

        for class_index in range(num_classes):
            # 避免超出顏色列表，目前寫法最多20種，看要不要加一個while重複執行減法
            if class_index >= len(color_list):
                color_index = class_index - len(color_index)
            else:
                color_index = class_index

            box_color = color_list[color_index]  # BGR格式
            text_color = color_list[color_index]  # BGR格式，這裡是藍色
            one_class_objects = conf_filter_df[conf_filter_df["class"] == class_index]
            for one_detection_object_index in range(one_class_objects.shape[0]):
                one_detection_object = one_class_objects.iloc[one_detection_object_index]
                box_coords = (int(one_detection_object.loc["x1"]), 
                            int(one_detection_object.loc["y1"]), 
                            int(one_detection_object.loc["x2"]), 
                            int(one_detection_object.loc["y2"]))  # (x1, y1, x2, y2)
                # 繪製矩形框
                cv2.rectangle(image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), box_color, 7)

                text = f"class: {class_index} conf: {one_detection_object.loc['conf'].round(2)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                thickness = 5
                # 設置矩形的底色和文字顏色
                bg_color = color_list[color_index]
                text_color = (0, 0, 0)       # 黑色文字

                # 計算文字的大小
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # 繪製矩形作為底色
                cv2.rectangle(image, 
                            (box_coords[0], box_coords[1] - text_height - 10), 
                            (box_coords[0] + text_width, box_coords[1]), 
                            bg_color, 
                            cv2.FILLED)
                # 在矩形上添加文字
                cv2.putText(image, text, (box_coords[0], box_coords[1] - 5), font, font_scale, text_color, thickness)

        resized_image = cv2.resize(image, show_image_size)
        # 顯示修改後的圖像
        cv2.imshow('resized_image with Rectangle and Text', resized_image)
        # 等待按鍵，然後關閉所有窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        key_in = input("輸入y結束, 按enter繼續: ")
        if key_in == "y" or conf == "Y":
            conf_threshold = conf
            print(Fore.RED + f"finish, conf_threshold: {conf_threshold}")
            break
        else:
            continue

    if key_in == "":
        print(Fore.RED + "unable detection")
        return False
    else:
        print(Fore.RED + "Save the dataframe")
        return (conf_filter_df, conf_threshold)
    

def arrange_df(init_txt_df, yaml_path, conf_threshold, use_class_index_file):
    # 更新 DataFrame 的列標題
    new_headers = ['class', 'x', 'y', 'w', 'h', 'conf']
    init_txt_df.columns = new_headers
    #依據conf進行篩選
    init_txt_df = init_txt_df[init_txt_df["conf"] >= conf_threshold]
    if use_class_index_file:
        new_df = init_txt_df.drop(columns=['conf'])
    else:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        class_mapping = {}
        object_name_list = yaml_data["names"]
        for class_index in range(len(object_name_list)):
            class_mapping[class_index] = object_name_list[class_index]
        # 物件名稱對照
        init_txt_df['class_name'] = init_txt_df['class'].map(class_mapping)
        
        init_txt_df = init_txt_df.drop(columns=['class', 'conf'])
        new_df = pd.DataFrame({})
        new_df["class_name"] = init_txt_df["class_name"]
        colume_name_list = [col for col in init_txt_df.columns if col != "class_name"]
        for colume_name in colume_name_list:
            new_df[colume_name] = init_txt_df[colume_name]

    return new_df


def save_label_txt(image_path, folder_path, new_df):
    file_name = os.path.basename(image_path)
    # 去掉附檔名
    file_name_without_extension = os.path.splitext(file_name)[0]
    txt_path = os.path.join(folder_path, file_name_without_extension + ".txt")
    new_df.to_csv(txt_path, sep=' ', header=False, index=False)


def main(weight_path, image_path, yolo_root, txt_folder, yaml_path, use_class_index_file, show_image_size):
    init(autoreset=True)
    
    detection_df = detect.main(weight_path, image_path) #yolo v9 辨識結果
    result = drew_object(image_path, detection_df, yaml_path, show_image_size) #顯示辨識結果
    if result == False:
        print(Fore.RED + "detection error !!")
    else:
        conf_filter_df, conf_threshold = result
    init_txt_df = read_init_label_txt(yolo_root, txt_folder) #彙整辨識結果txt檔
    new_df = arrange_df(init_txt_df, yaml_path, conf_threshold, use_class_index_file)
    save_label_txt(image_path, txt_folder, new_df)
    print(new_df)


if __name__ =="__main__":
    weight_path = r"E:\1082_mhtsai\B2408_AI_lable_tool\yolov9\weights\gape_meter_gelan-m_20240911.pt"
    image_path = r"E:\1082_mhtsai\B2408_AI_lable_tool\test_image\IMG_1481.JPG"
    yolo_root = r"E:\1082_mhtsai\B2408_AI_lable_tool\yolov9"
    txt_folder = r"E:\1082_mhtsai\B2408_AI_lable_tool\yolov9\auto_label"
    yaml_path = r'E:\1082_mhtsai\B2408_AI_lable_tool\yolov9\gap_meter-6\data.yaml'
    show_image_size = (600, 800)
    use_class_index_file = True #True的話會使用數字index編列物件
    main(weight_path, image_path, yolo_root, txt_folder, yaml_path, use_class_index_file, show_image_size)

