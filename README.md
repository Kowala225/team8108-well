# AICUP 2025 心臟瓣膜辨識專案

## 專案簡介
AICUP 2025 心臟瓣膜辨識競賽，使用 YOLOv9m 物件偵測模型來辨識電腦斷層CT影像中的主動脈瓣膜。專案包含完整的資料前處理、資料集切分、資料增強、模型訓練與預測功能。

## 環境需求

### Python 版本
- Python 3.10

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

#### 資料集切分

本專案使用兩階段的資料集切分策略，確保資料分配的合理性與一致性。

**按 Patient 順序切分（40:10）**

使用 [dataprocess/move.py](dataprocess/move.py) 將有標註的資料集按 patient 順序切分：

```bash
python dataprocess/move.py
```

**功能特點**：
- 以 txt 標註檔案為基準進行切分
- 前 40 個 patient → 訓練集
- 後 10 個 patient → 驗證集
- 自動配對對應的圖片檔案（.png）

**配置方式**：編輯腳本頂部的 `CONFIG` 字典
```python
CONFIG = {
    'image_dir': r"C:\path\to\training_image",    # 圖片資料夾
    'label_dir': r"C:\path\to\training_label",    # 標註資料夾
    'output_dir': r"./datasets",                  # 輸出目錄
    'train_ratio': 40,                            # 訓練集patient數量
    'val_ratio': 10,                              # 驗證集patient數量
}
```

**隨機挑選剩餘資料（2787張）**

使用 [dataprocess/moveother.py](dataprocess/moveother.py) 從剩餘圖片中隨機挑選：

```bash
python dataprocess/moveother.py
```

#### 標註正規化

確保 YOLO 格式標註檔案的座標值在有效範圍內（0-1之間）：

```bash
python dataprocess/normalize_yolo.py <標註資料夾路徑> [輸出資料夾路徑]

```

#### 資料增強

使用 [dataprocess/augment_yolo.py](dataprocess/augment_yolo.py) 進行資料增強

**支援的增強方法**：
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

**訓練參數**：
- **預訓練模型**: `yolov9m.pt`（YOLO官方預訓練權重）
- **資料集配置**: [aortic_valve_colab.yaml](aortic_valve_colab.yaml)
- **訓練輪數**: 100 epochs
- **批次大小**: 4
- **圖片尺寸**: 640×640
- **運算裝置**: GPU (device=0)

### 3. 模型預測

使用 [predict.py](predict.py) 進行預測：

```bash
python predict.py
```

**預測參數**：
- **模型權重**: `best.pt`
- **預測來源**: 指定的圖片資料夾
- **圖片尺寸**: 640×640
- **運算裝置**: GPU (device=0)
- **結果儲存**: 自動儲存標註後的圖片至 `runs/detect/`

## 訓練結果

根據 [results.csv](runs/train/train/results.csv) 顯示的訓練指標，模型已完成100個epoch的訓練，表現如下：

### 最佳指標

| 指標 | 數值 | Epoch |
|------|------|-------|
| 最佳 mAP50 | 96.7% | 第81輪 |
| 最佳 mAP50-95 | 70.9% | 第81輪 |
| 最佳 Precision | 92.0% | 第41輪 |
| 最佳 Recall | 92.1% | 第49輪 |

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

- **基礎架構**: YOLOv9m
- **偵測類別**: 單一類別（主動脈瓣膜）
- **輸入尺寸**: 640×640 像素
- **輸出格式**: Bounding box 座標與信心分數

## 參考資源

- [Ultralytics YOLO 官方文件](https://docs.ultralytics.com/)
- [AICUP 2025 競賽官網](https://www.aicup.tw/)
- [YOLOv9 GitHub Repository](https://github.com/WongKinYiu/yolov9)

## 授權聲明

本專案僅供學術研究與競賽使用。

