from roboflow import Roboflow
rf = Roboflow(api_key="6vi6277zjuyrW8KAIZMq")
project = rf.workspace("tsaimenghsuan-cfdts").project("gap_meter")
version = project.version(6)
dataset = version.download("yolov9")

# import os
# HOME = os.getcwd()
# print(HOME)


# train yolov9 models
# python train_dual.py --workers 8 --device 0 --batch 16 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# train gelan models
# python train.py --workers 8 --device 0 --batch 32 --data data/coco.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights '' --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15
# python train.py --workers 8 --device 0 --batch 16 --data .\new_bubbles-10\data.yaml --img 640 --cfg models/detect/gelan-c.yaml --weights .\weights\gelan-c.pt --name gelan-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 25 --close-mosaic 15

# 影像辨識用
# python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './weights/yolov9-c-converted.pt' --name yolov9_c_c_640_detect
# python detect.py --source './data/images/horses.jpg' --img 640 --device cpu --weights './weights/yolov9-c-converted.pt' --name yolov9_c_c_640_detect
# python detect.py --source './data/images/horses.jpg' --img 640 --device cpu --weights './weights/bubble_20240903.pt' --name yolov9_c_c_640_detect