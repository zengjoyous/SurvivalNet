import pandas as pd
import numpy as np

def load_clinical_data(file_path, sep=None):
    """
    读取 TCGA (.tsv) 或 GEO (.csv/.txt) 的临床数据集
    """
    if sep is None:
        if file_path.endswith('.tsv'):
            sep = '\t'
        else:
            sep = ','
    try:
        df = pd.read_csv(file_path, sep=sep)
        print(f"成功读取数据，原始矩阵形状为: {df.shape}")
        return df
    except Exception as e:
        print(f"读取文件失败，请检查路径: {e}")
        return None

def normalize_survival_data(df, time_col, status_col, id_col=None):
    """
    具体要求 1：正确处理 censoring（删失数据标准映射）
    状态统一映射为: 1 -> 事件发生(死亡/复发), 0 -> 删失(存活/失访)
    """
    processed_df = df.copy()
    
    # 1. 强制转换生存时间为数值型，无法转换的变为空值
    processed_df['duration'] = pd.to_numeric(processed_df[time_col], errors='coerce')
    
    # 2. 核心：删失数据标准化字典 (涵盖 TCGA/GEO 常见表述)
    status_mapping = {
        'Alive': 0, 'alive': 0, 'Censored': 0, 'censored': 0, '0': 0, 0: 0, 'LWT': 0,
        'Dead': 1, 'dead': 1, 'Event': 1, 'event': 1, '1': 1, 1: 1
    }
    
    # 清理状态列的空格并进行映射
    processed_df['event'] = processed_df[status_col].astype(str).str.strip().map(status_mapping)
    
    # 3. 剔除时间和状态有缺失的无效样本
    before_count = len(processed_df)
    processed_df = processed_df.dropna(subset=['duration', 'event'])
    processed_df['event'] = processed_df['event'].astype(int)
    
    print(f"删失数据标准化完成：有效样本 {len(processed_df)} 例 (剔除了 {before_count - len(processed_df)} 例缺失值)")
    
    # 4. 提取输出：ID + 标准生存数据 + 其余临床/组学特征
    keep_cols = ['duration', 'event']
    if id_col and id_col in processed_df.columns:
        keep_cols.insert(0, id_col)
        
    feature_cols = [col for col in processed_df.columns if col not in [time_col, status_col, 'duration', 'event', id_col]]
    
    return processed_df[keep_cols + feature_cols]