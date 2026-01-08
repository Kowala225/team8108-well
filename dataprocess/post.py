"""
YOLO 預測結果後處理工具
針對每張圖片只保留信心分數最高的預測框，刪除其他預測框
"""

import os
from pathlib import Path
from typing import List, Tuple
import shutil


# ============ 配置區 - 請修改以下路徑 ============
CONFIG = {
    'input_dir': r"C:\Users\user\Desktop\team8108\runs\detect\predict\labels",   # 預測結果標註資料夾
    'output_dir': r"C:\Users\user\Desktop\team8108\runs\detect\predict_filtered", # 輸出資料夾
    'backup_original': True,  # 是否備份原始檔案
}
# =================================================


class PredictionPostProcessor:
    """預測結果後處理器"""
    
    def __init__(self, input_dir: str, output_dir: str, backup_original: bool = True):
        """
        初始化後處理器
        
        Args:
            input_dir: 預測結果標註資料夾路徑
            output_dir: 輸出資料夾路徑
            backup_original: 是否備份原始檔案
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.backup_original = backup_original
        
        # 創建輸出資料夾
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.backup_original:
            self.backup_dir = self.output_dir.parent / f"{self.output_dir.name}_backup"
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"初始化完成:")
        print(f"  輸入來源: {self.input_dir}")
        print(f"  輸出位置: {self.output_dir}")
        if self.backup_original:
            print(f"  備份位置: {self.backup_dir}")
        print(f"\n處理規則: 每張圖片只保留信心分數最高的預測框")
    
    def read_yolo_predictions(self, label_path: str) -> List[Tuple[int, float, float, float, float, float]]:
        """
        讀取YOLO預測結果
        
        Args:
            label_path: 標註檔案路徑
            
        Returns:
            預測列表 [(class_id, x_center, y_center, width, height, confidence), ...]
        """
        predictions = []
        
        if not os.path.exists(label_path):
            return predictions
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    # YOLO預測格式: class x y w h confidence
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 如果有信心分數（第6個值）
                        if len(parts) >= 6:
                            confidence = float(parts[5])
                        else:
                            # 如果沒有信心分數，設為1.0（標準標註格式）
                            confidence = 1.0
                        
                        predictions.append((class_id, x_center, y_center, width, height, confidence))
        except Exception as e:
            print(f"  ⚠️ 讀取預測失敗 {label_path}: {str(e)}")
            return []
        
        return predictions
    
    def save_yolo_prediction(self, prediction: Tuple[int, float, float, float, float, float], output_path: str):
        """
        保存單個YOLO預測結果
        
        Args:
            prediction: 預測資料 (class_id, x_center, y_center, width, height, confidence)
            output_path: 輸出檔案路徑
        """
        class_id, x_center, y_center, width, height, confidence = prediction
        
        with open(output_path, 'w') as f:
            # 根據原始格式決定是否輸出信心分數
            # 如果信心分數不是1.0，表示原始檔案有信心分數
            if confidence != 1.0:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
            else:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def filter_predictions(self, predictions: List[Tuple[int, float, float, float, float, float]]) -> Tuple[int, float, float, float, float, float]:
        """
        從預測列表中選擇信心分數最高的
        
        Args:
            predictions: 預測列表
            
        Returns:
            信心分數最高的預測
        """
        if not predictions:
            return None
        
        # 按信心分數排序（降序）
        sorted_predictions = sorted(predictions, key=lambda x: x[5], reverse=True)
        
        # 返回信心分數最高的預測
        return sorted_predictions[0]
    
    def process_single_file(self, label_file: Path) -> Tuple[int, int]:
        """
        處理單個標註檔案
        
        Args:
            label_file: 標註檔案路徑
            
        Returns:
            (原始預測框數量, 過濾後數量)
        """
        # 讀取所有預測
        predictions = self.read_yolo_predictions(str(label_file))
        
        if not predictions:
            # 如果沒有預測，創建空檔案
            output_path = self.output_dir / label_file.name
            open(output_path, 'w').close()
            return 0, 0
        
        original_count = len(predictions)
        
        # 選擇信心分數最高的預測
        best_prediction = self.filter_predictions(predictions)
        
        if best_prediction:
            # 保存結果
            output_path = self.output_dir / label_file.name
            self.save_yolo_prediction(best_prediction, str(output_path))
            filtered_count = 1
        else:
            # 創建空檔案
            output_path = self.output_dir / label_file.name
            open(output_path, 'w').close()
            filtered_count = 0
        
        return original_count, filtered_count
    
    def process_all(self):
        """
        批次處理所有標註檔案
        """
        # 獲取所有txt檔案
        label_files = list(self.input_dir.glob('*.txt'))
        
        if not label_files:
            print("❌ 未找到標註檔案！")
            return
        
        print(f"\n找到 {len(label_files)} 個標註檔案")
        print("\n開始處理...\n")
        
        # 備份原始檔案
        if self.backup_original:
            print(f"正在備份原始檔案...")
            for label_file in label_files:
                shutil.copy(str(label_file), str(self.backup_dir / label_file.name))
            print(f"✓ 已備份 {len(label_files)} 個檔案至 {self.backup_dir}\n")
        
        # 統計
        total_original_boxes = 0
        total_filtered_boxes = 0
        files_with_multiple_boxes = 0
        files_with_no_boxes = 0
        
        for i, label_file in enumerate(label_files, 1):
            original_count, filtered_count = self.process_single_file(label_file)
            
            total_original_boxes += original_count
            total_filtered_boxes += filtered_count
            
            if original_count > 1:
                files_with_multiple_boxes += 1
                removed = original_count - filtered_count
                print(f"[{i}/{len(label_files)}] {label_file.name}: {original_count} 框 -> {filtered_count} 框 (移除 {removed} 框)")
            elif original_count == 1:
                print(f"[{i}/{len(label_files)}] {label_file.name}: 1 框 (無需處理)")
            else:
                files_with_no_boxes += 1
                print(f"[{i}/{len(label_files)}] {label_file.name}: 無預測框")
        
        # 輸出統計
        print("\n" + "="*60)
        print("後處理完成！")
        print(f"總檔案數量: {len(label_files)}")
        print(f"有多個預測框的檔案: {files_with_multiple_boxes}")
        print(f"無預測框的檔案: {files_with_no_boxes}")
        print(f"原始預測框總數: {total_original_boxes}")
        print(f"過濾後預測框總數: {total_filtered_boxes}")
        print(f"移除預測框總數: {total_original_boxes - total_filtered_boxes}")
        print(f"\n輸出位置: {self.output_dir}")
        if self.backup_original:
            print(f"備份位置: {self.backup_dir}")
        print("="*60)


def main():
    """主程式"""
    print("="*60)
    print("YOLO 預測結果後處理工具")
    print("="*60)
    
    print(f"\n當前配置：")
    print(f"  輸入來源: {CONFIG['input_dir']}")
    print(f"  輸出位置: {CONFIG['output_dir']}")
    print(f"  備份原始: {CONFIG['backup_original']}")
    
    # 檢查路徑是否存在
    input_path = Path(CONFIG['input_dir'])
    
    if not input_path.exists():
        print(f"\n❌ 錯誤：輸入資料夾不存在！")
        print(f"   路徑：{CONFIG['input_dir']}")
        print(f"\n請修改程式開頭的 CONFIG['input_dir'] 路徑")
        input("\n按 Enter 鍵結束...")
        return
    
    print(f"\n✓ 路徑檢查通過")
    
    # 顯示處理說明
    print(f"\n處理規則：")
    print(f"  • 每張圖片只保留信心分數最高的預測框")
    print(f"  • 其他預測框將被刪除")
    print(f"  • 支援 YOLO 格式標註（有或無信心分數）")
    
    user_input = input(f"\n是否開始後處理？(y/n): ").strip().lower()
    
    if user_input != 'y':
        print("已取消操作")
        return
    
    # 創建後處理器
    processor = PredictionPostProcessor(
        input_dir=CONFIG['input_dir'],
        output_dir=CONFIG['output_dir'],
        backup_original=CONFIG['backup_original']
    )
    
    # 執行後處理
    processor.process_all()
    
    print(f"\n完成！按 Enter 鍵結束...")
    input()


if __name__ == "__main__":
    main()
