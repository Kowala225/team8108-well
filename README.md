# AICUP 2025 心臟瓣膜辨識

## 簡介
AICUP 2025 心臟瓣膜辨識競賽，使用 YOLOv9 物件偵測模型來辨識超音波影像中的主動脈瓣膜。本github包含完整的資料前處理、資料增強、模型訓練與預測功能。

## 環境需求

### Python 版本
- Python 3.8+

### 套件依賴
```bash
ultralytics==8.3.248
```

安裝方式：
```bash
pip install -r requirements.txt
```

## 使用說明

### 1. 資料前處理

#### 標註正規化
確保 YOLO 格式標註檔案的座標值在有效範圍內（0-1之間）：

```bash
python normalize_yolo.py <標註資料夾路徑> [輸出資料夾路徑]
```

功能：
- 限制類別編號為非負整數
- 確保中心點座標 (x, y) 在 [0, 1] 範圍內
- 確保寬高 (w, h) 在 [0.001, 1] 範圍內
- 防止邊界框超出圖片範圍

#### 資料增強
使用 [augment_yolo.py](augment_yolo.py) 進行資料增強，支援多種增強方法：

```python
from augment_yolo import YOLODataAugmentor

augmentor = YOLODataAugmentor(
    img_dir='./datasets/train/images',
    label_dir='./datasets/train/labels', 
    output_dir='./datasets/augmented'
)

# 執行資料增強
augmentor.augment_dataset(augmentation_per_image=3)
```

支援的增強方法：
- 水平翻轉
- 隨機旋轉（-15° ~ +15°）
- 隨機亮度對比調整
- 高斯雜訊
- 影像模糊
- 格式驗證與標準化

### 2. 模型訓練

使用 [train.py](train.py) 訓練模型：

```bash
python train.py
```

訓練參數：
- **預訓練模型**: `yolov9m.pt`（YOLO官方預訓練權重）
- **資料集配置**: [aortic_valve_colab.yaml](aortic_valve_colab.yaml)
- **訓練輪數**: 100 epochs
- **批次大小**: 4
- **圖片尺寸**: 640×640
- **運算裝置**: GPU (device=0)

訓練輸出：
- 模型權重儲存於 `runs/train/train/weights/`
- 訓練指標記錄於 `runs/train/train/results.csv`
- TensorBoard 事件檔案可用於視覺化分析

### 3. 模型預測

使用 [predict.py](predict.py) 進行預測：

```bash
python predict.py
```

預測參數：
- **模型權重**: `best.pt`
- **預測來源**: 指定的圖片資料夾
- **圖片尺寸**: 640×640
- **運算裝置**: GPU (device=0)
- **結果儲存**: 自動儲存標註後的圖片至 `runs/detect/`

## 訓練結果

根據 [results.csv](runs/train/train/results.csv) 顯示的訓練指標，模型已完成100個epoch的訓練，表現如下：

### 最佳性能指標

| 指標 | 數值 | Epoch |
|------|------|-------|
| 最佳 mAP50 | 96.7% | 第81輪 |
| 最佳 mAP50-95 | 70.9% | 第81、86輪 |
| 最佳 Precision | 92.0% | 第41輪 |
| 最佳 Recall | 95.4% | 第47輪 |
| 最終 mAP50 | 96.0% | 第100輪 |
| 最終 mAP50-95 | 70.4% | 第100輪 |

### 損失函數趨勢

| 損失類型 | 初始值 (Epoch 1) | 最終值 (Epoch 100) | 改善幅度 |
|---------|-----------------|-------------------|---------|
| Box Loss | 1.778 | 0.643 | ↓63.8% |
| Class Loss | 2.560 | 0.310 | ↓87.9% |
| DFL Loss | 1.547 | 0.936 | ↓39.5% |

## 資料集配置

[aortic_valve_colab.yaml](aortic_valve_colab.yaml) 定義資料集路徑與類別：

```yaml
train: "./datasets/train/images"
val: "./datasets/val/images"
test: "./datasets/test/images"

names:
  0: aortic_valve
```

## 模型架構

- **基礎架構**: YOLOv9m-nano
- **偵測類別**: 單一類別（主動脈瓣膜）
- **輸入尺寸**: 640×640 像素
- **輸出格式**: Bounding box 座標與信心分數