"""
YOLO格式資料增強工具
針對有標註的影像進行資料增強
支援：
1. 針對bounding box中心座標的影像縮放
2. 左右旋轉影像
3. 鏡像反轉
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


# ============ 配置區 - 請修改以下路徑 ============
CONFIG = {
    'img_dir': r"C:\Users\user\Desktop\team8108\datasets\train\images",      # 圖片資料夾路徑
    'label_dir': r"C:\Users\user\Desktop\team8108\datasets\train\labels",    # 標註資料夾路徑
    'output_dir': r"C:\Users\user\Desktop\team8108\datasets\augmented",      # 輸出資料夾路徑
    'copy_original': True,                                                   # 是否複製原始檔案
}
# =================================================


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
        print(f"\n增強方法:")
        print(f"  1. 鏡像反轉（水平翻轉）")
        print(f"  2. 左右旋轉（±15度）")
        print(f"  3. 以bbox中心為基準的縮放（0.8x, 1.2x）")
        print(f"  ⚠️ 只處理有標註檔案的影像")
    
    def read_yolo_labels(self, label_path: str) -> List[List[float]]:
        """
        讀取YOLO標註
        
        Args:
            label_path: 標註檔案路徑
            
        Returns:
            標註列表 [[class_id, x_center, y_center, width, height], ...]
        """
        labels = []
        
        if not os.path.exists(label_path):
            return labels
        
        try:
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
                        
                        labels.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            print(f"  ⚠️ 讀取標註失敗 {label_path}: {str(e)}")
            return []
        
        return labels
    
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
                # 確保值在合理範圍內
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0.001, 1)
                height = np.clip(height, 0.001, 1)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def horizontal_flip(self, image: np.ndarray, labels: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        """
        鏡像反轉（水平翻轉）
        
        Args:
            image: 原始圖片
            labels: 標註列表
            
        Returns:
            (翻轉後的圖片, 翻轉後的標註)
        """
        flipped_image = cv2.flip(image, 1)
        flipped_labels = []
        
        for label in labels:
            class_id, x_center, y_center, width, height = label
            # 水平翻轉：x座標對稱
            new_x_center = 1.0 - x_center
            flipped_labels.append([class_id, new_x_center, y_center, width, height])
        
        return flipped_image, flipped_labels
    
    def rotate_image(self, image: np.ndarray, labels: List[List[float]], angle: float) -> Tuple[np.ndarray, List[List[float]]]:
        """
        旋轉影像（繞中心旋轉）
        
        Args:
            image: 原始圖片
            labels: 標註列表
            angle: 旋轉角度（正值為逆時針）
            
        Returns:
            (旋轉後的圖片, 旋轉後的標註)
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # 獲取旋轉矩陣
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋轉圖片
        rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # 旋轉標註
        rotated_labels = []
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # 轉換為絕對座標
            x_abs = x_center * w
            y_abs = y_center * h
            
            # 相對於中心的座標
            x_rel = x_abs - center[0]
            y_rel = y_abs - center[1]
            
            # 旋轉座標
            x_new = x_rel * cos_a - y_rel * sin_a + center[0]
            y_new = x_rel * sin_a + y_rel * cos_a + center[1]
            
            # 轉回歸一化座標
            x_center_new = x_new / w
            y_center_new = y_new / h
            
            # 旋轉寬高（近似處理，對小角度旋轉足夠準確）
            width_new = width
            height_new = height
            
            # 檢查是否在圖片範圍內
            if 0 <= x_center_new <= 1 and 0 <= y_center_new <= 1:
                rotated_labels.append([class_id, x_center_new, y_center_new, width_new, height_new])
        
        return rotated_image, rotated_labels
    
    def crop_around_bbox_center(self, image: np.ndarray, labels: List[List[float]], scale: float) -> Tuple[np.ndarray, List[List[float]]]:
        """
        以bounding box中心為基準進行影像裁切/縮放
        
        Args:
            image: 原始圖片
            labels: 標註列表
            scale: 縮放比例（<1為放大視野，>1為縮小視野）
            
        Returns:
            (處理後的圖片, 處理後的標註)
        """
        if not labels:
            return image, labels
        
        h, w = image.shape[:2]
        
        # 計算所有bbox的平均中心點
        centers_x = [label[1] for label in labels]
        centers_y = [label[2] for label in labels]
        avg_center_x = np.mean(centers_x)
        avg_center_y = np.mean(centers_y)
        
        # 計算裁切區域
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        
        # 確保裁切區域不超出原圖
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # 計算裁切起點（以bbox中心為基準）
        center_x_abs = int(avg_center_x * w)
        center_y_abs = int(avg_center_y * h)
        
        x1 = max(0, center_x_abs - crop_w // 2)
        y1 = max(0, center_y_abs - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        # 調整起點確保裁切區域大小一致
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        # 裁切圖片
        cropped = image[y1:y2, x1:x2]
        
        # 縮放回原尺寸
        scaled_image = cv2.resize(cropped, (w, h))
        
        # 調整標註座標
        scaled_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # 轉換為絕對座標
            x_abs = x_center * w
            y_abs = y_center * h
            
            # 調整到裁切區域的相對位置
            x_new = (x_abs - x1) / (x2 - x1)
            y_new = (y_abs - y1) / (y2 - y1)
            
            # 調整寬高
            width_new = width * w / (x2 - x1)
            height_new = height * h / (y2 - y1)
            
            # 檢查是否在有效範圍內
            if 0 <= x_new <= 1 and 0 <= y_new <= 1 and width_new > 0 and height_new > 0:
                scaled_labels.append([class_id, x_new, y_new, width_new, height_new])
        
        return scaled_image, scaled_labels
    
    def process_single_image(self, img_path: Path, label_path: Path):
        """
        對單張圖片進行所有增強處理
        
        Args:
            img_path: 圖片路徑
            label_path: 標註路徑
        """
        # 讀取圖片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️ 無法讀取圖片: {img_path}")
            return 0
        
        # 讀取標註
        labels = self.read_yolo_labels(str(label_path))
        
        if not labels:
            return 0
        
        img_stem = img_path.stem
        img_ext = img_path.suffix
        aug_count = 0
        
        try:
            # 1. 鏡像反轉（水平翻轉）
            flip_img, flip_labels = self.horizontal_flip(image, labels)
            output_name = f"{img_stem}_flip{img_ext}"
            cv2.imwrite(str(self.output_img_dir / output_name), flip_img)
            self.save_yolo_labels(flip_labels, str(self.output_label_dir / f"{img_stem}_flip.txt"))
            aug_count += 1
            
            # 2. 左旋轉15度
            rot_left_img, rot_left_labels = self.rotate_image(image, labels, 15)
            if rot_left_labels:  # 確保旋轉後仍有有效標註
                output_name = f"{img_stem}_rot_left{img_ext}"
                cv2.imwrite(str(self.output_img_dir / output_name), rot_left_img)
                self.save_yolo_labels(rot_left_labels, str(self.output_label_dir / f"{img_stem}_rot_left.txt"))
                aug_count += 1
            
            # 3. 右旋轉15度
            rot_right_img, rot_right_labels = self.rotate_image(image, labels, -15)
            if rot_right_labels:  # 確保旋轉後仍有有效標註
                output_name = f"{img_stem}_rot_right{img_ext}"
                cv2.imwrite(str(self.output_img_dir / output_name), rot_right_img)
                self.save_yolo_labels(rot_right_labels, str(self.output_label_dir / f"{img_stem}_rot_right.txt"))
                aug_count += 1
            
            # 4. 以bbox中心縮放 0.8x（放大視野）
            scale_out_img, scale_out_labels = self.crop_around_bbox_center(image, labels, 0.8)
            if scale_out_labels:
                output_name = f"{img_stem}_scale_out{img_ext}"
                cv2.imwrite(str(self.output_img_dir / output_name), scale_out_img)
                self.save_yolo_labels(scale_out_labels, str(self.output_label_dir / f"{img_stem}_scale_out.txt"))
                aug_count += 1
            
            # 5. 以bbox中心縮放 1.2x（縮小視野，聚焦）
            scale_in_img, scale_in_labels = self.crop_around_bbox_center(image, labels, 1.2)
            if scale_in_labels:
                output_name = f"{img_stem}_scale_in{img_ext}"
                cv2.imwrite(str(self.output_img_dir / output_name), scale_in_img)
                self.save_yolo_labels(scale_in_labels, str(self.output_label_dir / f"{img_stem}_scale_in.txt"))
                aug_count += 1
            
            print(f"  ✓ {img_path.name} -> 生成 {aug_count} 張增強圖片")
            return aug_count
            
        except Exception as e:
            print(f"  ⚠️ 處理失敗 {img_path}: {str(e)}")
            return 0
    
    def augment_dataset(self, copy_original: bool = True):
        """
        批次增強所有有標註的圖片
        
        Args:
            copy_original: 是否複製原始檔案到輸出資料夾
        """
        # 獲取所有標註檔案
        label_files = list(self.label_dir.glob('*.txt'))
        
        if not label_files:
            print("❌ 未找到標註檔案！")
            return
        
        print(f"\n找到 {len(label_files)} 個標註檔案")
        print("\n開始處理...\n")
        
        # 統計
        processed_count = 0
        skipped_count = 0
        total_augmented = 0
        
        for label_file in label_files:
            label_stem = label_file.stem
            
            # 尋找對應的圖片檔案
            img_file = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                potential_img = self.img_dir / f"{label_stem}{ext}"
                if potential_img.exists():
                    img_file = potential_img
                    break
            
            if img_file is None:
                print(f"⚠️ 找不到對應圖片: {label_stem}")
                skipped_count += 1
                continue
            
            print(f"處理: {img_file.name}")
            
            # 複製原始檔案
            if copy_original:
                import shutil
                shutil.copy(str(img_file), str(self.output_img_dir / img_file.name))
                shutil.copy(str(label_file), str(self.output_label_dir / label_file.name))
                print(f"  ✓ 原始檔案已複製")
            
            # 進行資料增強
            aug_count = self.process_single_image(img_file, label_file)
            
            if aug_count > 0:
                processed_count += 1
                total_augmented += aug_count
            else:
                print(f"  ⚠️ 跳過（無有效標註）")
                skipped_count += 1
            
            print()
        
        # 輸出統計
        print("="*60)
        print("資料增強完成！")
        print(f"原始圖片數量: {len(label_files)}")
        print(f"成功處理: {processed_count}")
        print(f"跳過: {skipped_count}")
        print(f"生成增強圖片: {total_augmented}")
        if copy_original:
            print(f"總圖片數量: {processed_count + total_augmented}")
        else:
            print(f"總圖片數量: {total_augmented}")
        print(f"\n輸出位置:")
        print(f"  圖片: {self.output_img_dir}")
        print(f"  標註: {self.output_label_dir}")
        print("="*60)


def main():
    """主程式"""
    print("="*60)
    print("YOLO 資料增強工具")
    print("="*60)
    
    print(f"\n當前配置：")
    print(f"  圖片來源: {CONFIG['img_dir']}")
    print(f"  標註來源: {CONFIG['label_dir']}")
    print(f"  輸出位置: {CONFIG['output_dir']}")
    print(f"  複製原始: {CONFIG['copy_original']}")
    
    # 檢查路徑是否存在
    img_path = Path(CONFIG['img_dir'])
    label_path = Path(CONFIG['label_dir'])
    
    if not img_path.exists():
        print(f"\n❌ 錯誤：圖片資料夾不存在！")
        print(f"   路徑：{CONFIG['img_dir']}")
        print(f"\n請修改程式開頭的 CONFIG['img_dir'] 路徑")
        input("\n按 Enter 鍵結束...")
        return
    
    if not label_path.exists():
        print(f"\n❌ 錯誤：標註資料夾不存在！")
        print(f"   路徑：{CONFIG['label_dir']}")
        print(f"\n請修改程式開頭的 CONFIG['label_dir'] 路徑")
        input("\n按 Enter 鍵結束...")
        return
    
    print(f"\n✓ 路徑檢查通過")
    
    # 顯示增強方法
    print(f"\n增強方法說明：")
    print(f"  1. 鏡像反轉（水平翻轉）")
    print(f"  2. 左旋轉 15 度")
    print(f"  3. 右旋轉 15 度")
    print(f"  4. 以 bbox 中心縮放 0.8x（放大視野）")
    print(f"  5. 以 bbox 中心縮放 1.2x（聚焦）")
    print(f"\n⚠️ 注意：只會處理有標註檔案(.txt)的圖片")
    
    user_input = input(f"\n是否開始資料增強？(y/n): ").strip().lower()
    
    if user_input != 'y':
        print("已取消操作")
        return
    
    # 創建增強器
    augmentor = YOLODataAugmentor(
        img_dir=CONFIG['img_dir'],
        label_dir=CONFIG['label_dir'],
        output_dir=CONFIG['output_dir']
    )
    
    # 執行資料增強
    augmentor.augment_dataset(copy_original=CONFIG['copy_original'])
    
    print(f"\n完成！按 Enter 鍵結束...")
    input()


if __name__ == "__main__":
    main()
