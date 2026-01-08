"""
資料集切分工具
根據patient順序將資料集切分為訓練集和驗證集
切分比例：40:10 (train:val)
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

# ============================================
# 配置區域 - 請在此修改您的路徑設定
# ============================================
CONFIG = {
    'image_dir': r"C:\Users\user\Desktop\dataset\aicup\training_image",        # 圖片根目錄（包含 patientxxxx 資料夾）
    'label_dir': r"C:\Users\user\Desktop\dataset\aicup\training_label",        # 標註根目錄（包含 patientxxxx 資料夾）
    'output_dir': r"./datasets",     # 輸出目錄
    'train_ratio': 40,               # 訓練集patient數量
    'val_ratio': 10,                 # 驗證集patient數量
}
# ============================================


class DatasetSplitter:
    """資料集切分器"""
    
    def __init__(self, image_dir: str, label_dir: str, output_dir: str, 
                 train_ratio: int = 40, val_ratio: int = 10):
        """
        初始化資料集切分器
        
        Args:
            image_dir: 圖片根目錄（包含 patientxxxx 子資料夾）
            label_dir: 標註根目錄（包含 patientxxxx 子資料夾）
            output_dir: 輸出目錄
            train_ratio: 訓練集數量
            val_ratio: 驗證集數量
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # 創建輸出資料夾結構
        self.train_img_dir = self.output_dir / 'train' / 'images'
        self.train_label_dir = self.output_dir / 'train' / 'labels'
        self.val_img_dir = self.output_dir / 'val' / 'images'
        self.val_label_dir = self.output_dir / 'val' / 'labels'
        
        for dir_path in [self.train_img_dir, self.train_label_dir, 
                         self.val_img_dir, self.val_label_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化完成:")
        print(f"  圖片來源: {self.image_dir}")
        print(f"  標註來源: {self.label_dir}")
        print(f"  輸出位置: {self.output_dir}")
        print(f"  切分比例: {self.train_ratio}:{self.val_ratio} (train:val)")
    
    def get_patient_folders(self) -> List[str]:
        """
        獲取所有patient資料夾並排序
        
        Returns:
            排序後的patient資料夾名稱列表
        """
        # 從標註資料夾獲取所有patient資料夾（以txt為準）
        patient_folders = []
        
        if not self.label_dir.exists():
            raise FileNotFoundError(f"標註資料夾不存在: {self.label_dir}")
        
        for item in self.label_dir.iterdir():
            if item.is_dir() and item.name.startswith('patient'):
                # 檢查資料夾內是否有txt檔案
                txt_files = list(item.glob('*.txt'))
                if txt_files:
                    patient_folders.append(item.name)
        
        # 排序patient資料夾
        patient_folders.sort()
        
        print(f"\n找到 {len(patient_folders)} 個patient資料夾")
        return patient_folders
    
    def split_patients(self, patient_folders: List[str]) -> Tuple[List[str], List[str]]:
        """
        根據比例切分patient資料夾
        
        Args:
            patient_folders: 所有patient資料夾列表
            
        Returns:
            (訓練集patient列表, 驗證集patient列表)
        """
        total = len(patient_folders)
        split_idx = self.train_ratio
        
        if split_idx > total:
            print(f"警告: 訓練集數量 ({self.train_ratio}) 超過總數 ({total})")
            print(f"       將使用前 {total - self.val_ratio} 個作為訓練集")
            split_idx = total - self.val_ratio
        
        train_patients = patient_folders[:split_idx]
        val_patients = patient_folders[split_idx:split_idx + self.val_ratio]
        
        print(f"\n切分結果:")
        print(f"  訓練集: {len(train_patients)} 個patients")
        print(f"  驗證集: {len(val_patients)} 個patients")
        
        if len(val_patients) < self.val_ratio:
            print(f"  警告: 驗證集實際數量 ({len(val_patients)}) 少於預期 ({self.val_ratio})")
        
        return train_patients, val_patients
    
    def move_patient_data(self, patient_name: str, is_train: bool) -> Tuple[int, int]:
        """
        移動單個patient的資料到對應資料夾
        
        Args:
            patient_name: patient資料夾名稱
            is_train: True表示訓練集，False表示驗證集
            
        Returns:
            (成功移動的圖片數量, 成功移動的標註數量)
        """
        # 確定目標資料夾
        if is_train:
            target_img_dir = self.train_img_dir
            target_label_dir = self.train_label_dir
        else:
            target_img_dir = self.val_img_dir
            target_label_dir = self.val_label_dir
        
        label_src_dir = self.label_dir / patient_name
        image_src_dir = self.image_dir / patient_name
        
        img_count = 0
        label_count = 0
        
        # 遍歷標註檔案（以txt為準）
        for txt_file in label_src_dir.glob('*.txt'):
            # 移動標註檔案
            label_dst = target_label_dir / txt_file.name
            shutil.move(str(txt_file), str(label_dst))
            label_count += 1
            
            # 尋找對應的圖片檔案
            base_name = txt_file.stem  # 不含副檔名
            png_file = image_src_dir / f"{base_name}.png"
            
            if png_file.exists():
                # 移動圖片檔案
                img_dst = target_img_dir / png_file.name
                shutil.move(str(png_file), str(img_dst))
                img_count += 1
            else:
                print(f"  警告: 找不到對應的圖片 {png_file.name}")
        
        return img_count, label_count
    
    def split_dataset(self):
        """執行資料集切分"""
        print("=" * 60)
        print("開始資料集切分")
        print("=" * 60)
        
        # 1. 獲取所有patient資料夾
        patient_folders = self.get_patient_folders()
        
        if not patient_folders:
            print("錯誤: 沒有找到任何patient資料夾")
            return
        
        # 2. 切分patient列表
        train_patients, val_patients = self.split_patients(patient_folders)
        
        # 3. 移動訓練集資料
        print("\n處理訓練集...")
        train_img_total = 0
        train_label_total = 0
        
        for i, patient in enumerate(train_patients, 1):
            img_count, label_count = self.move_patient_data(patient, is_train=True)
            train_img_total += img_count
            train_label_total += label_count
            print(f"  [{i}/{len(train_patients)}] {patient}: {img_count} 圖片, {label_count} 標註")
        
        # 4. 移動驗證集資料
        print("\n處理驗證集...")
        val_img_total = 0
        val_label_total = 0
        
        for i, patient in enumerate(val_patients, 1):
            img_count, label_count = self.move_patient_data(patient, is_train=False)
            val_img_total += img_count
            val_label_total += label_count
            print(f"  [{i}/{len(val_patients)}] {patient}: {img_count} 圖片, {label_count} 標註")
        
        # 5. 輸出統計資訊
        print("\n" + "=" * 60)
        print("切分完成！")
        print("=" * 60)
        print(f"\n訓練集統計:")
        print(f"  Patient數量: {len(train_patients)}")
        print(f"  圖片數量: {train_img_total}")
        print(f"  標註數量: {train_label_total}")
        print(f"  儲存位置:")
        print(f"    - 圖片: {self.train_img_dir}")
        print(f"    - 標註: {self.train_label_dir}")
        
        print(f"\n驗證集統計:")
        print(f"  Patient數量: {len(val_patients)}")
        print(f"  圖片數量: {val_img_total}")
        print(f"  標註數量: {val_label_total}")
        print(f"  儲存位置:")
        print(f"    - 圖片: {self.val_img_dir}")
        print(f"    - 標註: {self.val_label_dir}")
        
        print(f"\n總計:")
        print(f"  Patient數量: {len(train_patients) + len(val_patients)}")
        print(f"  圖片數量: {train_img_total + val_img_total}")
        print(f"  標註數量: {train_label_total + val_label_total}")


def main():
    """主程式"""
    import sys
    
    # 從命令列參數讀取（如果有提供）
    if len(sys.argv) >= 3:
        print("使用命令列參數指定路徑\n")
        image_dir = sys.argv[1]
        label_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else CONFIG['output_dir']
        train_ratio = int(sys.argv[4]) if len(sys.argv) >= 5 else CONFIG['train_ratio']
        val_ratio = int(sys.argv[5]) if len(sys.argv) >= 6 else CONFIG['val_ratio']
    else:
        # 使用配置檔案中的路徑
        print("=" * 60)
        print("使用配置檔案中的路徑設定")
        print("=" * 60)
        print(f"圖片資料夾: {CONFIG['image_dir']}")
        print(f"標註資料夾: {CONFIG['label_dir']}")
        print(f"輸出資料夾: {CONFIG['output_dir']}")
        print(f"切分比例: {CONFIG['train_ratio']}:{CONFIG['val_ratio']}")
        print("\n如需修改，請編輯腳本頂部的 CONFIG 配置區域")
        print("或使用命令列參數：")
        print(f"  python {os.path.basename(sys.argv[0])} <圖片資料夾> <標註資料夾> [輸出資料夾] [訓練集數量] [驗證集數量]\n")
        
        # 詢問是否繼續
        response = input("按 Enter 繼續，或輸入 'n' 取消: ").strip().lower()
        if response == 'n':
            print("已取消操作")
            return
        
        image_dir = CONFIG['image_dir']
        label_dir = CONFIG['label_dir']
        output_dir = CONFIG['output_dir']
        train_ratio = CONFIG['train_ratio']
        val_ratio = CONFIG['val_ratio']
    
    try:
        # 創建切分器並執行
        splitter = DatasetSplitter(
            image_dir=image_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        
        splitter.split_dataset()
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
