import pandas as pd
import numpy as np
import os

def load_clinical_data(file_path, sep=None):
    """
    读取 TCGA (.tsv) 或 GEO (.csv/.txt) 的临床数据集
    """
    if sep is None:
        # 如果是 .tsv 或 .txt 结尾，生信标准默认用制表符 \t 分隔
        if file_path.endswith('.tsv') or file_path.endswith('.txt'):
            sep = '\t'
        else:
            sep = ','
    try:
        # cBioPortal 的临床文件前几行通常有以 # 开头的注释，这里用 comment='#' 优雅跳过
        df = pd.read_csv(file_path, sep=sep, comment='#')
        print(f"成功读取数据，原始矩阵形状为: {df.shape}")
        return df
    except Exception as e:
        print(f"读取文件失败，请检查路径: {e}")
        return None

def normalize_survival_data(df, time_col, status_col, id_col=None):
    """
    具体要求 1：正确处理 censoring（删失数据标准映射）
    """
    processed_df = df.copy()
    
    # 1. 强制转换生存时间为数值型
    processed_df['duration'] = pd.to_numeric(processed_df[time_col], errors='coerce')
    
    # 2. 核心：完美兼容 cBioPortal 下载的真实生存状态标签
    status_mapping = {
        'Alive': 0, 'alive': 0, 'Censored': 0, 'censored': 0, '0': 0, 0: 0, 'LWT': 0, '0:LIVING': 0, 'LIVING': 0,
        'Dead': 1, 'dead': 1, 'Event': 1, 'event': 1, '1': 1, 1: 1, '1:DECEASED': 1, 'DECEASED': 1
    }
    
    processed_df['event'] = processed_df[status_col].astype(str).str.strip().map(status_mapping)
    
    # 3. 剔除无效样本
    before_count = len(processed_df)
    processed_df = processed_df.dropna(subset=['duration', 'event'])
    processed_df['event'] = processed_df['event'].astype(int)
    
    print(f"删失数据标准化完成：有效样本 {len(processed_df)} 例 (剔除了 {before_count - len(processed_df)} 例缺失值)")
    
    keep_cols = ['duration', 'event']
    if id_col and id_col in processed_df.columns:
        keep_cols.insert(0, id_col)
        
    feature_cols = [col for col in processed_df.columns if col not in [time_col, status_col, 'duration', 'event', id_col]]
    
    return processed_df[keep_cols + feature_cols]

def get_builtin_example_data():
    """
    【唐一禾负责：示例数据准备】
    一键获取存放在 examples/example_data/ 下的真实 cBioPortal 临床示例文件
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 💥 这里名字精准对齐你电脑里的真实文件名 data_clinical_patient.txt
    example_file_path = os.path.join(base_dir, "examples", "example_data", "data_clinical_patient.txt")
    
    if not os.path.exists(example_file_path):
        raise FileNotFoundError(
            f"未找到真实的示例数据文件，请确保您已将本地的 data_clinical_patient.txt "
            f"放置在: {example_file_path}"
        )
        
    print(f"\n[SurvivalNet] 成功加载唐一禾准备的真实内置示例数据: {example_file_path}")
    
    # 自动读取
    df_raw = load_clinical_data(example_file_path)
    return df_raw
