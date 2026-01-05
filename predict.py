import ultralytics
from ultralytics import YOLO

model = YOLO(r'.\best.pt')
results = model.predict(source=r"C:\Users\user\Desktop\dataset\aicup\predict\images",
              save=True,
              imgsz=640,
              device=0
              )