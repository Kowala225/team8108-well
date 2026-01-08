"""
剩餘資料集切分工具
從剩餘的照片中隨機挑選2787張進行切分
根據patient編號決定放入train或val：
- patient 1-40: 放入train
- patient 41-50: 放入val
- patient 51+: 不處理

注意：此腳本只處理圖片檔案，不需要標註檔案
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Dict


# ============================================
# 配置區域 - 請在此修改您的路徑設定
# ============================================
CONFIG = {
    'image_dir': r"C:\Users\user\Desktop\dataset\aicup\training_image",  # 剩餘圖片資料夾
    'output_dir': r"./datasets",     # 輸出目錄（會添加到現有的train/val資料夾）
    'total_samples': 2787,           # 要隨機挑選的總數量
    'patient_train_max': 40,         # train集的patient上限（1-40）
    'patient_val_min': 41,           # val集的patient下限（41-50）
    'patient_val_max': 50,           # val集的patient上限
}
# ============================================


class RemainingDatasetSplitter:
    """剩餘資料集切分器（僅處理圖片）"""
    
    def __init__(self, image_dir: str, output_dir: str,
                 total_samples: int = 2787, patient_train_max: int = 40,
                 patient_val_min: int = 41, patient_val_max: int = 50):
        """
        初始化剩餘資料集切分器
        
        Args:
            image_dir: 圖片資料夾路徑
            output_dir: 輸出目錄
            total_samples: 要隨機挑選的總數量
            patient_train_max: train集的patient上限
            patient_val_min: val集的patient下限
            patient_val_max: val集的patient上限
        """
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.patient_train_max = patient_train_max
        self.patient_val_min = patient_val_min
        self.patient_val_max = patient_val_max
        
        # 輸出資料夾（只需要圖片資料夾）
        self.train_img_dir = self.output_dir / 'train' / 'images'
        self.val_img_dir = self.output_dir / 'val' / 'images'
        
        # 確保輸出資料夾存在
        for dir_path in [self.train_img_dir, self.val_img_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化完成:")
        print(f"  圖片來源: {self.image_dir}")
        print(f"  輸出位置: {self.output_dir}")
        print(f"  挑選數量: {self.total_samples} 張")
        print(f"  Patient分配: 1-{self.patient_train_max} → train, "
              f"{self.patient_val_min}-{self.patient_val_max} → val")
    
    def extract_patient_number(self, filename: str) -> int:
        """
        從檔名中提取patient編號
        例如: patient0001_0001.png -> 1
        
        Args:
            filename: 檔案名稱
            
        Returns:
            patient編號（整數）
        """
        try:
            # 假設格式為 patientXXXX_YYYY.ext
            if filename.startswith('patient'):
                patient_part = filename.split('_')[0]  # 取得 patientXXXX
                patient_num = int(patient_part.replace('patient', ''))  # 取得數字
                return patient_num
        except:
            pass
        return -1  # 無法解析
    
    def collect_available_samples(self) -> Dict[str, List[Path]]:
        """
        收集所有可用的圖片樣本
        
        Returns:
            字典 {'train': [檔案列表], 'val': [檔案列表], 'skip': [檔案列表]}
        """
        samples = {
            'train': [],
            'val': [],
            'skip': []
        }
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"圖片資料夾不存在: {self.image_dir}")
        
        print("\n掃描可用圖片...")
        
        # 先檢查是否有子資料夾
        subdirs = [d for d in self.image_dir.iterdir() if d.is_dir()]
        
        if subdirs and any(d.name.startswith('patient') for d in subdirs):
            # 資料夾結構: images/patientXXXX/*.png
            for patient_dir in subdirs:
                if not patient_dir.name.startswith('patient'):
                    continue
                    
                for png_file in patient_dir.glob('*.png'):
                    patient_num = self.extract_patient_number(png_file.name)
                    
                    # 根據patient編號分類
                    if 1 <= patient_num <= self.patient_train_max:
                        samples['train'].append(png_file)
                    elif self.patient_val_min <= patient_num <= self.patient_val_max:
                        samples['val'].append(png_file)
                    else:
                        samples['skip'].append(png_file)
        else:
            # 扁平結構: images/*.png
            for png_file in self.image_dir.glob('*.png'):
                patient_num = self.extract_patient_number(png_file.name)
                
                # 根據patient編號分類
                if 1 <= patient_num <= self.patient_train_max:
                    samples['train'].append(png_file)
                elif self.patient_val_min <= patient_num <= self.patient_val_max:
                    samples['val'].append(png_file)
                else:
                    samples['skip'].append(png_file)
        
        print(f"  Train候選: {len(samples['train'])} 張")
        print(f"  Val候選: {len(samples['val'])} 張")
        print(f"  跳過: {len(samples['skip'])} 張 (patient > {self.patient_val_max})")
        print(f"  總計: {len(samples['train']) + len(samples['val'])} 張可用")
        
        return samples
    
    def random_select_samples(self, samples: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
        """
        從可用樣本中隨機挑選指定數量
        維持train和val的相對比例
        
        Args:
            samples: 可用樣本字典
            
        Returns:
            挑選後的樣本字典
        """
        total_available = len(samples['train']) + len(samples['val'])
        
        if total_available < self.total_samples:
            print(f"\n警告: 可用樣本 ({total_available}) 少於需求 ({self.total_samples})")
            print(f"      將使用所有可用樣本")
            return samples
        
        # 計算train和val的比例
        train_ratio = len(samples['train']) / total_available
        val_ratio = len(samples['val']) / total_available
        
        # 計算要選取的數量（按比例）
        train_count = int(self.total_samples * train_ratio)
        val_count = self.total_samples - train_count
        
        print(f"\n隨機挑選樣本...")
        print(f"  Train: {train_count} 張 (從 {len(samples['train'])} 張中挑選)")
        print(f"  Val: {val_count} 張 (從 {len(samples['val'])} 張中挑選)")
        
        # 隨機挑選
        random.shuffle(samples['train'])
        random.shuffle(samples['val'])
        
        selected = {
            'train': samples['train'][:train_count],
            'val': samples['val'][:val_count],
            'skip': []
        }
        
        return selected
    
    def move_samples(self, samples: Dict[str, List[Path]]):
        """
        移動挑選的圖片到目標資料夾
        
        Args:
            samples: 要移動的樣本字典
        """
        print("\n開始移動檔案...")
        
        # 移動train樣本
        print("\n處理Train集...")
        for i, png_file in enumerate(samples['train'], 1):
            # 移動圖片
            img_dst = self.train_img_dir / png_file.name
            shutil.move(str(png_file), str(img_dst))
            
            if i % 100 == 0 or i == len(samples['train']):
                print(f"  已處理: {i}/{len(samples['train'])}")
        
        # 移動val樣本
        print("\n處理Val集...")
        for i, png_file in enumerate(samples['val'], 1):
            # 移動圖片
            img_dst = self.val_img_dir / png_file.name
            shutil.move(str(png_file), str(img_dst))
            
            if i % 100 == 0 or i == len(samples['val']):
                print(f"  已處理: {i}/{len(samples['val'])}")
    
    def split_dataset(self):
        """執行資料集切分"""
        print("=" * 60)
        print("開始剩餘資料集切分")
        print("=" * 60)
        
        # 1. 收集可用樣本
        samples = self.collect_available_samples()
        
        total_available = len(samples['train']) + len(samples['val'])
        if total_available == 0:
            print("\n錯誤: 沒有找到任何可用樣本")
            return
        
        # 2. 隨機挑選指定數量
        selected = self.random_select_samples(samples)
        
        # 3. 移動檔案
        self.move_samples(selected)
        
        # 4. 輸出統計
        print("\n" + "=" * 60)
        print("切分完成！")
        print("=" * 60)
        print(f"\n已移動樣本統計:")
        print(f"  Train: {len(selected['train'])} 張")
        print(f"  Val: {len(selected['val'])} 張")
        print(f"  總計: {len(selected['train']) + len(selected['val'])} 張")
        print(f"\n儲存位置:")
        print(f"  Train圖片: {self.train_img_dir}")
        print(f"  Val圖片: {self.val_img_dir}")


def main():
    """主程式"""
    import sys
    
    # 從命令列參數讀取（如果有提供）
    if len(sys.argv) >= 2:
        print("使用命令列參數指定路徑\n")
        image_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else CONFIG['output_dir']
        total_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else CONFIG['total_samples']
    else:
        # 使用配置檔案中的路徑
        print("=" * 60)
        print("使用配置檔案中的路徑設定")
        print("=" * 60)
        print(f"圖片資料夾: {CONFIG['image_dir']}")
        print(f"輸出資料夾: {CONFIG['output_dir']}")
        print(f"挑選數量: {CONFIG['total_samples']}")
        print("\n如需修改，請編輯腳本頂部的 CONFIG 配置區域")
        print("或使用命令列參數：")
        print(f"  python {os.path.basename(sys.argv[0])} <圖片資料夾> [輸出資料夾] [挑選數量]\n")
        
        # 詢問是否繼續
        response = input("按 Enter 繼續，或輸入 'n' 取消: ").strip().lower()
        if response == 'n':
            print("已取消操作")
            return
        
        image_dir = CONFIG['image_dir']
        output_dir = CONFIG['output_dir']
        total_samples = CONFIG['total_samples']
    
    try:
        # 設定隨機種子以確保可重現性（可選）
        random.seed(42)
        
        # 創建切分器並執行
        splitter = RemainingDatasetSplitter(
            image_dir=image_dir,
            output_dir=output_dir,
            total_samples=total_samples,
            patient_train_max=CONFIG['patient_train_max'],
            patient_val_min=CONFIG['patient_val_min'],
            patient_val_max=CONFIG['patient_val_max']
        )
        
        splitter.split_dataset()
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
