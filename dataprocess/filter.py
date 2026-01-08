"""
YOLO 預測結果連續性過濾工具
只保留連續出現N次的預測結果，用於過濾視頻序列中的誤判
例如：只保留連續30張都有預測的圖片，單獨出現的預測將被刪除
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set
import shutil


# ============ 配置區 - 請修改以下路徑和參數 ============
CONFIG = {
    'input_dir': r"C:\Users\user\Desktop\team8108\runs\detect\predict\labels",   # 預測結果標註資料夾
    'output_dir': r"C:\Users\user\Desktop\team8108\runs\detect\predict_continuous", # 輸出資料夾
    'continuous_threshold': 30,  # 連續出現的閾值（張數）
    'backup_original': True,     # 是否備份原始檔案
}
# =================================================


class ContinuousFilter:
    """連續性過濾器"""
    
    def __init__(self, input_dir: str, output_dir: str, continuous_threshold: int = 30, backup_original: bool = True):
        """
        初始化過濾器
        
        Args:
            input_dir: 預測結果標註資料夾路徑
            output_dir: 輸出資料夾路徑
            continuous_threshold: 連續出現的閾值
            backup_original: 是否備份原始檔案
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.continuous_threshold = continuous_threshold
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
        print(f"  連續閾值: {self.continuous_threshold} 張")
        print(f"\n處理規則: 只保留連續{self.continuous_threshold}張都有預測的圖片")
    
    def extract_frame_number(self, filename: str) -> int:
        """
        從檔名中提取幀號/序號
        
        Args:
            filename: 檔案名稱
            
        Returns:
            幀號（整數），如果無法提取則返回-1
        """
        # 嘗試多種常見的命名模式
        patterns = [
            r'(\d+)\.txt$',           # 純數字.txt
            r'_(\d+)\.txt$',          # xxx_數字.txt
            r'frame_?(\d+)\.txt$',    # frame_數字.txt 或 frame數字.txt
            r'img_?(\d+)\.txt$',      # img_數字.txt 或 img數字.txt
            r'image_?(\d+)\.txt$',    # image_數字.txt 或 image數字.txt
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # 如果所有模式都不匹配，嘗試提取檔名中的最後一組數字
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        
        return -1
    
    def has_predictions(self, label_file: Path) -> bool:
        """
        檢查標註檔案是否有預測結果
        
        Args:
            label_file: 標註檔案路徑
            
        Returns:
            是否有預測結果
        """
        if not label_file.exists():
            return False
        
        try:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                return len(content) > 0
        except:
            return False
    
    def analyze_continuity(self, files_dict: Dict[int, Path]) -> Set[int]:
        """
        分析哪些幀符合連續性要求
        
        Args:
            files_dict: {幀號: 檔案路徑} 的字典
            
        Returns:
            符合連續性要求的幀號集合
        """
        if not files_dict:
            return set()
        
        # 獲取所有有預測的幀號（排序）
        frames_with_predictions = sorted([frame for frame, path in files_dict.items() if self.has_predictions(path)])
        
        if len(frames_with_predictions) < self.continuous_threshold:
            print(f"  ⚠️ 總共只有 {len(frames_with_predictions)} 張有預測，少於閾值 {self.continuous_threshold}")
            return set()
        
        # 找出所有連續片段
        valid_frames = set()
        
        for i, start_frame in enumerate(frames_with_predictions):
            # 檢查從當前幀開始，後續是否有連續N張
            consecutive_count = 1
            current_frame = start_frame
            
            for j in range(i + 1, len(frames_with_predictions)):
                next_frame = frames_with_predictions[j]
                # 檢查是否連續（允許幀號相差1）
                if next_frame == current_frame + 1:
                    consecutive_count += 1
                    current_frame = next_frame
                    
                    # 如果達到閾值，將這個片段的所有幀加入有效集合
                    if consecutive_count >= self.continuous_threshold:
                        for k in range(consecutive_count):
                            valid_frames.add(frames_with_predictions[i + k])
                else:
                    # 不連續了，跳出
                    break
        
        return valid_frames
    
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
        
        # 備份原始檔案
        if self.backup_original:
            print(f"\n正在備份原始檔案...")
            for label_file in label_files:
                shutil.copy(str(label_file), str(self.backup_dir / label_file.name))
            print(f"✓ 已備份 {len(label_files)} 個檔案至 {self.backup_dir}")
        
        print("\n分析檔案序號和預測結果...\n")
        
        # 建立 {幀號: 檔案路徑} 的映射
        files_dict = {}
        files_without_number = []
        
        for label_file in label_files:
            frame_num = self.extract_frame_number(label_file.name)
            if frame_num >= 0:
                files_dict[frame_num] = label_file
            else:
                files_without_number.append(label_file)
                print(f"⚠️ 無法從檔名提取序號: {label_file.name}")
        
        if files_without_number:
            print(f"\n⚠️ 有 {len(files_without_number)} 個檔案無法提取序號，將被跳過\n")
        
        if not files_dict:
            print("❌ 沒有可處理的檔案（無法提取序號）")
            return
        
        print(f"成功解析 {len(files_dict)} 個檔案的序號")
        print(f"幀號範圍: {min(files_dict.keys())} - {max(files_dict.keys())}")
        
        # 統計有預測的檔案
        files_with_predictions = [frame for frame in files_dict.keys() if self.has_predictions(files_dict[frame])]
        print(f"有預測結果的檔案: {len(files_with_predictions)} 個")
        
        # 分析連續性
        print(f"\n正在分析連續性（閾值: {self.continuous_threshold} 張）...\n")
        valid_frames = self.analyze_continuity(files_dict)
        
        print(f"符合連續性要求的幀: {len(valid_frames)} 個")
        
        # 處理檔案
        print(f"\n開始處理檔案...\n")
        
        kept_count = 0
        removed_count = 0
        no_prediction_count = 0
        
        for frame_num in sorted(files_dict.keys()):
            label_file = files_dict[frame_num]
            has_pred = self.has_predictions(label_file)
            
            output_path = self.output_dir / label_file.name
            
            if frame_num in valid_frames:
                # 保留：複製原始檔案
                shutil.copy(str(label_file), str(output_path))
                kept_count += 1
                if has_pred:
                    print(f"✓ 保留: {label_file.name} (幀 {frame_num})")
            else:
                # 不符合連續性：創建空檔案
                if has_pred:
                    open(output_path, 'w').close()
                    removed_count += 1
                    print(f"✗ 移除: {label_file.name} (幀 {frame_num}) - 不符合連續性")
                else:
                    # 原本就沒有預測
                    open(output_path, 'w').close()
                    no_prediction_count += 1
        
        # 輸出統計
        print("\n" + "="*60)
        print("連續性過濾完成！")
        print(f"總檔案數量: {len(files_dict)}")
        print(f"原本有預測的檔案: {len(files_with_predictions)}")
        print(f"符合連續性的檔案: {len(valid_frames)}")
        print(f"保留的預測: {kept_count}")
        print(f"移除的預測: {removed_count}")
        print(f"無預測的檔案: {no_prediction_count}")
        print(f"\n輸出位置: {self.output_dir}")
        if self.backup_original:
            print(f"備份位置: {self.backup_dir}")
        print("="*60)
        
        # 顯示連續片段資訊
        if valid_frames:
            print("\n連續片段資訊:")
            sorted_valid = sorted(valid_frames)
            segments = []
            start = sorted_valid[0]
            end = sorted_valid[0]
            
            for i in range(1, len(sorted_valid)):
                if sorted_valid[i] == end + 1:
                    end = sorted_valid[i]
                else:
                    segments.append((start, end, end - start + 1))
                    start = sorted_valid[i]
                    end = sorted_valid[i]
            segments.append((start, end, end - start + 1))
            
            for i, (start, end, length) in enumerate(segments, 1):
                print(f"  片段 {i}: 幀 {start}-{end} (共 {length} 張)")


def main():
    """主程式"""
    print("="*60)
    print("YOLO 預測結果連續性過濾工具")
    print("="*60)
    
    print(f"\n當前配置：")
    print(f"  輸入來源: {CONFIG['input_dir']}")
    print(f"  輸出位置: {CONFIG['output_dir']}")
    print(f"  連續閾值: {CONFIG['continuous_threshold']} 張")
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
    print(f"  • 只保留連續{CONFIG['continuous_threshold']}張都有預測的圖片")
    print(f"  • 單獨出現或連續少於{CONFIG['continuous_threshold']}張的預測將被刪除")
    print(f"  • 適用於視頻序列幀的誤判過濾")
    print(f"  • 檔名需包含序號（例如: frame_0001.txt, img001.txt, 0001.txt）")
    
    user_input = input(f"\n是否開始過濾？(y/n): ").strip().lower()
    
    if user_input != 'y':
        print("已取消操作")
        return
    
    # 創建過濾器
    filter_processor = ContinuousFilter(
        input_dir=CONFIG['input_dir'],
        output_dir=CONFIG['output_dir'],
        continuous_threshold=CONFIG['continuous_threshold'],
        backup_original=CONFIG['backup_original']
    )
    
    # 執行過濾
    filter_processor.process_all()
    
    print(f"\n完成！按 Enter 鍵結束...")
    input()


if __name__ == "__main__":
    main()
