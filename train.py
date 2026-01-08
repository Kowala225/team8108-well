import ultralytics
from ultralytics import YOLO

model = YOLO('yolov9m.pt') #初次訓練使用YOLO官方的預訓練模型，如要使用自己的模型訓練可以將'yolo12n.pt'替換掉
results = model.train(data="./aortic_valve_colab.yaml",
            epochs=100, #跑幾個epoch，這邊設定10做範例測試，可依需求調整
            batch=4, #batch_size
            imgsz=640, #圖片大小640*640
            device=0 #使用GPU進行訓練
            )