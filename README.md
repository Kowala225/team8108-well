# AICUP 2025 心臟瓣膜辨識專案

## 簡介
AICUP 2025 心臟瓣膜辨識競賽，使用 YOLOv9m 物件偵測模型來辨識電腦斷層CT影像中的主動脈瓣膜。專案包含完整的資料前處理、資料集切分、資料增強、模型訓練、預測與後處理功能。

## 結構

```
team8108/
├── dataprocess/             # 資料處理工具集
│   ├── move.py              # 資料集切分（有標註，40:10比例）
│   ├── moveother.py         # 資料集切分（無標註，2787張）
│   ├── normalize_yolo.py    # 標註座標正規化
│   ├── augment.py           # 資料增強（翻轉、旋轉、縮放）
│   ├── post.py              # 預測後處理（保留最高分）
│   └── filter.py            # 連續性過濾（視頻序列誤判過濾）
├── train.py                 # 模型訓練腳本
├── predict.py               # 模型預測腳本
├── best.pt                  # 訓練好的模型權重
└── aortic_valve_colab.yaml  # 資料集配置檔
```

## dataprocess 資料處理工具說明

### 1. [move.py](dataprocess/move.py) - 資料集切分（有標註）
**功能**：按 patient 順序切分有標註的資料集（40:10 比例）
- 以 txt 標註檔案為基準進行切分
- 前 40 個 patient → 訓練集
- 後 10 個 patient → 驗證集
- 自動配對對應的圖片檔案（.png）
- 使用 `shutil.move` 直接移動檔案

### 2. [moveother.py](dataprocess/moveother.py) - 資料集切分（無標註）
**功能**：從剩餘圖片中隨機挑選 2787 張（無標註資料）
- 隨機挑選 2787 張圖片（seed=42 可重現）
- 根據 patient 編號自動分類：
  - Patient 1-40 → 訓練集
  - Patient 41-50 → 驗證集
  - Patient 51+ → 跳過
- 只處理圖片，無標註檔案


### 3. [normalize_yolo.py](dataprocess/normalize_yolo.py) - 標註正規化
**功能**：確保 YOLO 格式標註檔案的座標值在有效範圍內
- 限制類別編號為非負整數
- 確保中心點座標 (x, y) 在 [0, 1] 範圍內
- 確保寬高 (w, h) 在 [0.001, 1] 範圍內
- 防止邊界框超出圖片範圍

### 4. [augment_yolo.py](dataprocess/augment_yolo.py) - 資料增強
**功能**：針對有標註的影像進行資料增強，擴充訓練資料
- **鏡像反轉**：水平翻轉影像與標註
- **左右旋轉**：旋轉 ±15 度
- **以 bbox 中心縮放**：
  - 0.8x 縮放（放大視野）
  - 1.2x 縮放（聚焦目標）
- 只處理有標註檔案的影像，無標註者自動跳過
- 每張圖片可生成 5 張增強圖片

### 5. [post.py](dataprocess/post.py) - 預測後處理
**功能**：處理模型預測結果，每張圖片只保留信心分數最高的預測框
- 讀取所有預測框及其信心分數
- 按信心分數排序
- 只保留分數最高的預測框
- 刪除其他較低分數的預測框
- 自動備份原始預測結果

**適用場景**：當模型對同一張圖片產生多個預測框時，只保留最可信的那一個

### 6. [filter.py](dataprocess/filter.py) - 連續性過濾
**功能**：過濾視頻序列中的誤判，只保留連續出現 N 張的預測結果
- 從檔名提取序號（支援多種命名格式）
- 分析預測結果的連續性
- 只保留連續 N 張（預設 30 張）都有預測的片段
- 刪除孤立或不連續的預測（視為誤判）
- 顯示所有連續片段的起止範圍

## 模型訓練

使用 [train.py](train.py) 訓練模型： 100 個 epoch 的訓練：

**訓練參數**：
- **預訓練模型**: `yolov9m.pt`（YOLO官方預訓練權重）
- **資料集配置**: [aortic_valve_colab.yaml](aortic_valve_colab.yaml)
- **訓練輪數**: 100 epochs
- **批次大小**: 4
- **圖片尺寸**: 640×640
- **運算裝置**: GPU (device=0)

## 模型預測

使用 [predict.py](predict.py) 進行預測：

**預測參數**：
- **模型權重**: `best.pt`
- **預測來源**: 指定的圖片資料夾
- **圖片尺寸**: 640×640
- **運算裝置**: GPU (device=0)
- **結果儲存**: 自動儲存標註後的圖片至 `runs/detect/`

## 預測結果後處理

**1 保留最高分預測框**（處理多框問題）

```bash
python dataprocess/postprocess_predictions.py
```

適用場景：當模型對同一張圖片產生多個預測框時

**2 連續性過濾**（處理視頻序列誤判）

```bash
python dataprocess/filter_continuous.py
```

適用場景：視頻序列預測，只保留連續出現 30 張的預測結果，過濾孤立誤判


## 資料集配置

[aortic_valve_colab.yaml](aortic_valve_colab.yaml) 定義資料集路徑與類別：


## yaml
```
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

