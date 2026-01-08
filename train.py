import ultralytics
from ultralytics import YOLO

model = YOLO('yolov9m.pt') 
results = model.train(data="./aortic_valve_colab.yaml",
            epochs=100, 
            batch=4, 
            imgsz=640,
            device=0
            )