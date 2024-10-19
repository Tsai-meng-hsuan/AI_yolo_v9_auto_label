import cv2
import pandas as pd

def drew_object(image_path, detection_df):
    # 讀取圖像
    # init_image = cv2.imread(image_path)
    # 定義框框的顏色和位置
    color_list = [(0, 0, 255), (0, 255, 0), (201, 42, 172), 
                  (255, 240, 142), (208, 168, 103), (171, 111, 220), 
                  (138, 115,38), (38, 205,38), (38, 205,38), (38, 205,38)]
    conf_list = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    for conf in conf_list:
        print(f"detection conf: {conf}")
        key_in = input("輸入y結束, 按enter繼續: ")
        if key_in == "y" or conf == "Y":
            print("finish")
            break
        else:
            image = cv2.imread(image_path)
            # conf = float(conf)
            conf_filter_df = detection_df[detection_df["conf"] >= conf]
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
                # print(type(one_class_objects))

                for one_detection_object_index in range(one_class_objects.shape[0]):
                    one_detection_object = one_class_objects.iloc[one_detection_object_index]
                    box_coords = (int(one_detection_object.loc["x1"]), 
                                int(one_detection_object.loc["y1"]), 
                                int(one_detection_object.loc["x2"]), 
                                int(one_detection_object.loc["y2"]))  # (x1, y1, x2, y2)
                    # print(box_coords)
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

            resized_image = cv2.resize(image, (600, 800))
            # 顯示修改後的圖像
            cv2.imshow('resized_image with Rectangle and Text', resized_image)

            # 等待按鍵，然後關閉所有窗口
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# 指定圖像檔案路徑
image_path = r'E:\1082_mhtsai\B2408_AI_lable_tool\test_image\IMG_2650_for_test.JPG'

# 假設數據存在一個 DataFrame 中
data = {
    'x1': [1996, 1556, 2422, 2366, 1510, 971],
    'y1': [1921, 2190, 1645, 2187, 1645, 1748],
    'x2': [2044, 1620, 2491, 2534, 1669, 1124],
    'y2': [1964, 2236, 1692, 2232, 1692, 1813],
    'conf': [0.253492, 0.569397, 0.776721, 0.787404, 0.824773, 0.853442],
    'class': [3, 0, 0, 1, 1, 2]
}
detection_df = pd.DataFrame(data)

drew_object(image_path, detection_df)