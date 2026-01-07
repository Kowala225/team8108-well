# AICUP 2025 心臟瓣膜辨識專案

## 專案簡介
本專案為參加 AICUP 2025 心臟瓣膜辨識競賽的解決方案，使用 YOLOv12 物件偵測模型來辨識超音波影像中的主動脈瓣膜（Aortic Valve）。專案包含完整的資料前處理、資料增強、模型訓練與預測功能。

## 專案結構
```
team8108/
├── train.py                    # 模型訓練腳本
├── predict.py                  # 模型預測腳本
├── augment_yolo.py            # YOLO格式資料增強工具
├── normalize_yolo.py          # YOLO標註正規化工具
├── aortic_valve_colab.yaml    # 資料集配置檔案
├── requirements.txt           # Python套件依賴
├── best.pt                    # 最佳訓練模型權重
├── README.md                  # 專案說明文件
├── pre/                       # 前處理資料資料夾
├── datasets/                  # 資料集資料夾
│   ├── train/images          # 訓練集圖片
│   ├── val/images            # 驗證集圖片
│   └── test/images           # 測試集圖片
└── runs/                      # 訓練與預測結果
    ├── train/train/          # 訓練輸出
    │   ├── weights/          # 模型權重
    │   │   ├── best.pt      # 最佳模型
    │   │   └── last.pt      # 最後一次訓練模型
    │   ├── results.csv       # 訓練指標記錄
    │   └── args.yaml         # 訓練參數配置
    └── detect/               # 預測結果輸出
```

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
- **預訓練模型**: `yolo12n.pt`（YOLO官方預訓練權重）
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

根據 [results.csv](runs/train/train/results.csv) 顯示的訓練指標，模型表現如下：

| 指標 | 數值 |
|------|------|
| 最佳 mAP50 | 94.3% (第13輪) |
| 最佳 mAP50-95 | 64.8% (第13輪) |
| 最佳 Precision | 87.4% (第6輪) |
| 最佳 Recall | 94.5% (第13輪) |

訓練過程觀察：
- 模型在前10個epoch快速收斂
- 在13-19個epoch達到穩定的高準確率
- 驗證損失函數逐漸降低，顯示良好的泛化能力

## 資料集配置

[aortic_valve_colab.yaml](aortic_valve_colab.yaml) 定義資料集路徑與類別：

```yaml
train: "./datasets/train/images"
val: "./datasets/val/images"
test: "./datasets/test/images"

names:
  0: aortic_valve
```

## 注意事項

1. **資料集路徑**：執行前請確認 YAML 檔案中的路徑設定正確
2. **GPU 需求**：訓練與預測預設使用 GPU，請確保有 CUDA 支援的 GPU
3. **記憶體需求**：建議至少 8GB RAM 和 4GB VRAM
4. **標註格式**：請確保標註檔案使用標準 YOLO 格式（每行：`<class> <x> <y> <w> <h>`）
5. **資料增強**：使用前請先備份原始資料

## 模型架構

- **基礎架構**: YOLOv12-nano
- **偵測類別**: 單一類別（主動脈瓣膜）
- **輸入尺寸**: 640×640 像素
- **輸出格式**: Bounding box 座標與信心分數

## 疑難排解

### 常見問題

1. **找不到 CUDA/GPU**
   - 確認已安裝 CUDA 工具包
   - 檢查 PyTorch 是否支援 CUDA
   - 可改用 `device='cpu'` 進行訓練（速度較慢）

2. **記憶體不足**
   - 減少 batch_size 參數
   - 降低圖片尺寸
   - 關閉其他佔用記憶體的程式

3. **標註格式錯誤**
   - 使用 `normalize_yolo.py` 修正標註檔案
   - 確認每行有且僅有 5 個數值

## 參考資源

- [Ultralytics YOLO 官方文件](https://docs.ultralytics.com/)
- [AICUP 2025 競賽官網](https://www.aicup.tw/)

## 授權聲明

本專案僅供學術研究與競賽使用。

## 聯絡資訊

如有問題或建議，歡迎透過 GitHub Issues 回報。