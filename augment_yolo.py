"""
YOLO格式資料增強工具
支援標準化格式檢查和多種資料增強方法
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.core.composition import BboxParams


class YOLODataAugmentor:
    """YOLO格式資料增強器"""
    
    def __init__(self, img_dir: str, label_dir: str, output_dir: str):
        """
        初始化資料增強器
        
        Args:
            img_dir: 圖片資料夾路徑
            label_dir: 標註檔案資料夾路徑
            output_dir: 輸出資料夾路徑
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        
        # 創建輸出資料夾
        self.output_img_dir = self.output_dir / 'images'
        self.output_label_dir = self.output_dir / 'labels'
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.output_label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化完成:")
        print(f"  圖片來源: {self.img_dir}")
        print(f"  標註來源: {self.label_dir}")
        print(f"  輸出位置: {self.output_dir}")
    
    def validate_yolo_format(self, label_path: str) -> Tuple[bool, str]:
        """
        驗證YOLO標註格式是否標準
        
        Args:
            label_path: 標註檔案路徑
            
        Returns:
            (是否有效, 錯誤訊息)
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return True, "空標註檔案(無物件)"
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    return False, f"第{i+1}行格式錯誤: 應有5個值(class x y w h)，實際有{len(parts)}個"
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 檢查歸一化值範圍
                    if not (0 <= x_center <= 1):
                        return False, f"第{i+1}行: x_center={x_center} 超出範圍[0,1]"
                    if not (0 <= y_center <= 1):
                        return False, f"第{i+1}行: y_center={y_center} 超出範圍[0,1]"
                    if not (0 <= width <= 1):
                        return False, f"第{i+1}行: width={width} 超出範圍[0,1]"
                    if not (0 <= height <= 1):
                        return False, f"第{i+1}行: height={height} 超出範圍[0,1]"
                    if class_id < 0:
                        return False, f"第{i+1}行: class_id={class_id} 不能為負數"
                    
                except ValueError as e:
                    return False, f"第{i+1}行數值轉換錯誤: {str(e)}"
            
            return True, "格式正確"
            
        except Exception as e:
            return False, f"讀取檔案錯誤: {str(e)}"
    
    def normalize_yolo_labels(self, label_path: str) -> List[List[float]]:
        """
        讀取並標準化YOLO標註
        
        Args:
            label_path: 標註檔案路徑
            
        Returns:
            標註列表 [[class_id, x_center, y_center, width, height], ...]
        """
        labels = []
        
        if not os.path.exists(label_path):
            return labels
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 確保值在合理範圍內
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width = np.clip(width, 0, 1)
                    height = np.clip(height, 0, 1)
                    
                    labels.append([class_id, x_center, y_center, width, height])
        
        return labels
    
    def yolo_to_pascal(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        YOLO格式轉Pascal VOC格式
        
        Args:
            yolo_bbox: [x_center, y_center, width, height] (歸一化)
            img_width: 圖片寬度
            img_height: 圖片高度
            
        Returns:
            [x_min, y_min, x_max, y_max] (絕對座標)
        """
        x_center, y_center, width, height = yolo_bbox
        
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height
        
        return [x_min, y_min, x_max, y_max]
    
    def pascal_to_yolo(self, pascal_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Pascal VOC格式轉YOLO格式
        
        Args:
            pascal_bbox: [x_min, y_min, x_max, y_max] (絕對座標)
            img_width: 圖片寬度
            img_height: 圖片高度
            
        Returns:
            [x_center, y_center, width, height] (歸一化)
        """
        x_min, y_min, x_max, y_max = pascal_bbox
        
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # 確保值在合理範圍內
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        return [x_center, y_center, width, height]
    
    def save_yolo_labels(self, labels: List[List[float]], output_path: str):
        """
        保存YOLO格式標註
        
        Args:
            labels: 標註列表
            output_path: 輸出檔案路徑
        """
        with open(output_path, 'w') as f:
            for label in labels:
                class_id = int(label[0])
                x_center, y_center, width, height = label[1:]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def get_augmentation_pipeline(self, aug_type: str) -> A.Compose:
        """
        獲取增強管道
        
        Args:
            aug_type: 增強類型
            
        Returns:
            Albumentations Compose物件
        """
        bbox_params = BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
        
        if aug_type == 'flip_horizontal':
            transform = A.Compose([
                A.HorizontalFlip(p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'flip_vertical':
            transform = A.Compose([
                A.VerticalFlip(p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'rotate_90':
            transform = A.Compose([
                A.Rotate(limit=(90, 90), p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'brightness':
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'contrast':
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'blur':
            transform = A.Compose([
                A.Blur(blur_limit=7, p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'noise':
            transform = A.Compose([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'hsv':
            transform = A.Compose([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
            ], bbox_params=bbox_params)
        
        elif aug_type == 'mixed':
            # 混合多種增強
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.Blur(blur_limit=3, p=0.3),
            ], bbox_params=bbox_params)
        
        else:
            raise ValueError(f"未知的增強類型: {aug_type}")
        
        return transform
    
    def augment_image(self, img_path: str, label_path: str, aug_type: str, suffix: str):
        """
        對單張圖片進行增強
        
        Args:
            img_path: 圖片路徑
            label_path: 標註路徑
            aug_type: 增強類型
            suffix: 輸出檔案後綴
        """
        # 讀取圖片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️ 無法讀取圖片: {img_path}")
            return
        
        # 讀取標註
        labels = self.normalize_yolo_labels(str(label_path))
        
        # 如果沒有標註，仍然可以增強圖片
        if not labels:
            bboxes = []
            class_labels = []
        else:
            bboxes = [[l[1], l[2], l[3], l[4]] for l in labels]
            class_labels = [int(l[0]) for l in labels]
        
        # 應用增強
        try:
            transform = self.get_augmentation_pipeline(aug_type)
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_class_labels = transformed['class_labels']
            
            # 生成輸出檔名
            img_stem = Path(img_path).stem
            img_ext = Path(img_path).suffix
            
            output_img_name = f"{img_stem}_{suffix}{img_ext}"
            output_label_name = f"{img_stem}_{suffix}.txt"
            
            # 保存增強後的圖片
            output_img_path = self.output_img_dir / output_img_name
            cv2.imwrite(str(output_img_path), aug_image)
            
            # 保存增強後的標註
            if aug_bboxes:
                aug_labels = []
                for class_id, bbox in zip(aug_class_labels, aug_bboxes):
                    aug_labels.append([class_id] + list(bbox))
                
                output_label_path = self.output_label_dir / output_label_name
                self.save_yolo_labels(aug_labels, str(output_label_path))
            else:
                # 即使沒有bbox也創建空標註檔案
                output_label_path = self.output_label_dir / output_label_name
                open(output_label_path, 'w').close()
            
            print(f"  ✓ {img_stem}{img_ext} -> {output_img_name}")
            
        except Exception as e:
            print(f"  ⚠️ 增強失敗 {img_path}: {str(e)}")
    
    def batch_augment(self, aug_types: List[str] = None, copy_original: bool = True):
        """
        批次增強所有圖片
        
        Args:
            aug_types: 增強類型列表，預設使用所有類型
            copy_original: 是否複製原始檔案
        """
        if aug_types is None:
            aug_types = ['flip_horizontal', 'flip_vertical', 'brightness', 'contrast', 'hsv']
        
        # 獲取所有圖片檔案
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_files = []
        for ext in img_extensions:
            img_files.extend(list(self.img_dir.glob(f'*{ext}')))
            img_files.extend(list(self.img_dir.glob(f'*{ext.upper()}')))
        
        if not img_files:
            print("❌ 未找到圖片檔案！")
            return
        
        print(f"\n找到 {len(img_files)} 張圖片")
        print(f"將使用以下增強方法: {', '.join(aug_types)}")
        print("\n開始處理...\n")
        
        # 統計
        valid_count = 0
        invalid_count = 0
        augmented_count = 0
        
        for img_file in img_files:
            img_stem = img_file.stem
            label_file = self.label_dir / f"{img_stem}.txt"
            
            # 驗證標註格式
            if label_file.exists():
                is_valid, message = self.validate_yolo_format(str(label_file))
                if not is_valid:
                    print(f"⚠️ 格式錯誤 {img_stem}.txt: {message}")
                    invalid_count += 1
                    continue
                valid_count += 1
            
            print(f"處理: {img_file.name}")
            
            # 複製原始檔案
            if copy_original:
                shutil.copy(str(img_file), str(self.output_img_dir / img_file.name))
                if label_file.exists():
                    shutil.copy(str(label_file), str(self.output_label_dir / label_file.name))
                print(f"  ✓ 原始檔案已複製")
            
            # 對每種增強類型進行處理
            for aug_type in aug_types:
                self.augment_image(
                    str(img_file),
                    str(label_file),
                    aug_type,
                    aug_type
                )
                augmented_count += 1
            
            print()
        
        # 輸出統計
        print("="*50)
        print("資料增強完成！")
        print(f"原始圖片數量: {len(img_files)}")
        print(f"有效標註: {valid_count}")
        print(f"無效標註: {invalid_count}")
        print(f"生成增強圖片: {augmented_count}")
        if copy_original:
            print(f"總圖片數量: {len(img_files) + augmented_count}")
        else:
            print(f"總圖片數量: {augmented_count}")
        print(f"\n輸出位置:")
        print(f"  圖片: {self.output_img_dir}")
        print(f"  標註: {self.output_label_dir}")
        print("="*50)


def main():
    """主程式"""
    # 設定路徑
    img_dir = r"path/to/images"  # 修改為你的圖片資料夾路徑
    label_dir = r"path/to/labels"  # 修改為你的標註資料夾路徑
    output_dir = r"path/to/output"  # 修改為你的輸出資料夾路徑
    
    # 創建增強器
    augmentor = YOLODataAugmentor(img_dir, label_dir, output_dir)
    
    # 選擇增強類型
    # 可用類型: 'flip_horizontal', 'flip_vertical', 'rotate_90', 
    #          'brightness', 'contrast', 'blur', 'noise', 'hsv', 'mixed'
    aug_types = [
        'flip_horizontal',
        'flip_vertical', 
        'brightness',
        'contrast',
        'hsv'
    ]
    
    # 執行批次增強
    augmentor.batch_augment(
        aug_types=aug_types,
        copy_original=True  # 是否複製原始檔案到輸出資料夾
    )


if __name__ == "__main__":
    main()
