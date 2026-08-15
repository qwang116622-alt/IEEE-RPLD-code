import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_pile_feature_columns(target_col):
    """Return the 19 pile-displacement inputs after withholding the target.

    The manuscript uses 47 inputs: 11 JG variables, 17 AQ variables, and the
    other 19 retaining-pile monitoring points.  Keeping the target out of the
    contemporaneous input avoids target leakage in next-step prediction.
    """
    pile_columns = [f'J{i}' for i in range(1, 21)]
    if target_col not in pile_columns:
        raise ValueError(
            f"target_col must be one of {pile_columns}; received {target_col!r}"
        )
    return [column for column in pile_columns if column != target_col]

class PileDisplacementDataset(Dataset):
    """
    基坑支护桩水平位移数据集
    用于生成模型训练和测试所需的数据
    """
    
    def __init__(self, data, seq_length, target_col, augment=False, augment_prob=0.3, original_data=None):
        """
        初始化数据集

        Args:
            data: 包含所有监测点数据的DataFrame
            seq_length: 时间序列长度
            target_col: 目标监测点列名
            augment: 是否进行数据增强
            augment_prob: 数据增强概率
            original_data: 原始未差分的数据（用于差分恢复）
        """
        self.data = data
        self.seq_length = seq_length
        self.target_col = target_col
        self.augment = augment
        self.augment_prob = augment_prob
        self.original_data = original_data
        self.features, self.targets, self.timestamps = self._create_sequences()
    
    def _augment_sequence(self, seq):
        """
        对时间序列进行增强
        
        Args:
            seq: 输入序列 (seq_len, features)
        
        Returns:
            augmented_seq: 增强后的序列
        """
        augmented_seq = seq.copy()
        
        # 随机缩放：0.95-1.05倍
        if np.random.rand() < self.augment_prob:
            scale_factor = np.random.uniform(0.95, 1.05)
            augmented_seq *= scale_factor
        
        # 随机平移：-0.1到0.1之间
        if np.random.rand() < self.augment_prob:
            shift = np.random.uniform(-0.1, 0.1)
            augmented_seq += shift
        
        # 随机噪声：添加高斯噪声
        if np.random.rand() < self.augment_prob:
            noise = np.random.normal(0, 0.05, augmented_seq.shape)
            augmented_seq += noise
        
        return augmented_seq
    
    def _create_sequences(self):
        """
        创建滑动窗口序列
        
        Returns:
            features: 特征序列列表
            targets: 目标值列表
            timestamps: 时间戳列表
        """
        features = []
        targets = []
        timestamps = []
        
        # 分离不同类型的监测点
        jg_cols = [f'JG{i}' for i in range(1, 12)]  # JG1-JG11
        aq_cols = [f'AQ{i}' for i in range(1, 18)]  # AQ1-AQ17
        j_cols = get_pile_feature_columns(self.target_col)  # 19 non-target J points

        required_columns = jg_cols + aq_cols + [f'J{i}' for i in range(1, 21)]
        missing_columns = [column for column in required_columns if column not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Input workbook is missing required columns: {missing_columns}")
        
        # 创建特征矩阵
        jg_data = self.data[jg_cols].values
        aq_data = self.data[aq_cols].values
        j_data = self.data[j_cols].values
        target_data = self.data[self.target_col].values
        
        # 生成滑动窗口
        for i in range(len(self.data) - self.seq_length):
            # 提取特征序列
            jg_seq = jg_data[i:i+self.seq_length]
            aq_seq = aq_data[i:i+self.seq_length]
            j_seq = j_data[i:i+self.seq_length]
            
            # 提取目标值
            target = target_data[i+self.seq_length]
            
            # 提取时间戳
            timestamp = self.data.index[i+self.seq_length]
            
            # 数据增强
            if self.augment:
                jg_seq = self._augment_sequence(jg_seq)
                aq_seq = self._augment_sequence(aq_seq)
                j_seq = self._augment_sequence(j_seq)
            
            # 将特征组合为字典
            feature = {
                'jg': jg_seq,
                'aq': aq_seq,
                'j': j_seq
            }
            
            features.append(feature)
            targets.append(target)
            timestamps.append(timestamp)
        
        return features, targets, timestamps
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        timestamp = self.timestamps[idx]

        # 转换为numpy数组，返回模型需要的所有特征
        # j1_j20 keeps the historical model interface; it contains 19
        # non-target pile features, not the target itself.
        feature_np = {
            'jg1_jg11': np.array(feature['jg'], dtype=np.float32),
            'aq1_aq17': np.array(feature['aq'], dtype=np.float32),
            'j1_j20': np.array(feature['j'], dtype=np.float32)
        }

        # 将Timestamp转换为字符串格式，避免DataLoader的collate错误
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # 如果有原始数据，返回前一个值用于差分恢复
        if self.original_data is not None:
            # 获取前一个值（索引是 seq_length + idx - 1）
            prev_idx = self.seq_length + idx - 1
            if prev_idx >= 0:
                prev_value = self.original_data[self.target_col].iloc[prev_idx]
            else:
                prev_value = 0.0
            return feature_np, np.array(target, dtype=np.float32), timestamp_str, np.array(prev_value, dtype=np.float32)

        return feature_np, np.array(target, dtype=np.float32), timestamp_str

class DataProcessor:
    """
    数据处理器
    用于数据加载、预处理和数据集构建
    """
    
    def __init__(self, config):
        """
        初始化数据处理器

        Args:
            config: 数据配置字典
        """
        self.config = config
        self.data = None
        self.scalers = {}
        self.last_values = {}  # 存储差分前的最后一个值，用于恢复
    
    def load_data(self):
        """
        加载数据

        Returns:
            data: 加载后的DataFrame
        """
        self.data = pd.read_excel(self.config['file_path'], index_col='index')
        self.data.index = pd.to_datetime(self.data.index)

        # 不进行特征筛选，使用所有特征按原始分组输入模型

        return self.data
    
    def normalize_data(self, train_data, test_data):
        """
        对数据进行归一化处理（可选差分）

        Args:
            train_data: 训练集DataFrame
            test_data: 测试集DataFrame

        Returns:
            train_norm: 归一化后的训练集
            test_norm: 归一化后的测试集
        """
        from sklearn.preprocessing import MinMaxScaler

        # 检查是否使用差分
        if self.config.get('use_diff', False):
            diff_order = self.config.get('diff_order', 1)
            print(f"\n使用差分处理：{diff_order}阶差分")

            # 对训练集进行差分
            train_diff = train_data.copy()
            for i in range(diff_order):
                train_diff = train_diff.diff().dropna()

            # 对测试集进行差分（保持时间连续性）
            # 合并训练集和测试集，统一差分，然后再分开
            combined = pd.concat([train_data, test_data])
            for i in range(diff_order):
                combined = combined.diff().dropna()

            # 分离差分后的训练集和测试集
            train_diff = combined.iloc[:len(train_diff)]
            test_diff = combined.iloc[len(train_diff):]

            # 保存差分前的最后一个值（用于后续恢复）
            original_train_data = train_data.copy()
            original_test_data = test_data.copy()
            self.last_values = {col: original_test_data.iloc[0][col] if len(original_test_data) > 0 else original_train_data.iloc[-1][col]
                             for col in original_train_data.columns}

            train_data = train_diff
            test_data = test_diff

            print(f"差分后训练集形状: {train_data.shape}")
            print(f"差分后测试集形状: {test_data.shape}")

        # 创建归一化器
        scaler = MinMaxScaler()

        # 对训练集进行归一化
        train_norm = train_data.copy()
        train_norm.iloc[:, :] = scaler.fit_transform(train_data)

        # 对测试集进行归一化（使用训练集的归一化参数）
        test_norm = test_data.copy()
        test_norm.iloc[:, :] = scaler.transform(test_data)

        # 保存归一化器，用于后续反归一化
        self.scalers['global'] = scaler

        # 保存原始数据用于差分恢复
        if self.config.get('use_diff', False):
            self.original_train_data = train_data
            self.original_test_data = test_data

        return train_norm, test_norm
    
    def split_data(self):
        """
        划分训练集和测试集
        采用时间顺序划分，保留时间序列的连续性
        确保测试集包含足够的历史数据来生成第一个预测样本
        
        Returns:
            train_data: 训练集DataFrame
            test_data: 测试集DataFrame
        """
        if self.data is None:
            self.load_data()
        
        # 按照时间顺序划分，保留时间序列的连续性
        train_size = int(len(self.data) * (1 - self.config['test_size']))
        
        # 为测试集添加训练集的最后seq_length个样本，用于生成测试集的第一个预测样本
        seq_length = self.config['seq_length']
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size-seq_length:]
        
        return train_data, test_data
    
    def create_datasets(self):
        """
        创建训练集和测试集

        Returns:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
        """
        train_data, test_data = self.split_data()

        # 对数据进行归一化处理
        print("\n对数据进行归一化处理...")
        train_data, test_data = self.normalize_data(train_data, test_data)

        # 获取原始数据（用于差分恢复）
        use_diff = self.config.get('use_diff', False)
        if use_diff and hasattr(self, 'original_train_data'):
            original_train = self.original_train_data
            original_test = self.original_test_data
        else:
            original_train = None
            original_test = None

        # 创建数据集 - 训练集禁用数据增强
        train_dataset = PileDisplacementDataset(
            train_data,
            self.config['seq_length'],
            self.config['target_point'],
            augment=False,  # 训练集禁用数据增强
            original_data=original_train
        )

        # 创建数据集 - 测试集禁用数据增强
        test_dataset = PileDisplacementDataset(
            test_data,
            self.config['seq_length'],
            self.config['target_point'],
            augment=False,  # 测试集禁用数据增强
            original_data=original_test
        )

        return train_dataset, test_dataset
    
    def create_dataloaders(self):
        """
        创建DataLoader
        
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        train_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=self.config['shuffle'], 
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            drop_last=False
        )
        
        return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据处理器
    from config import DATA_CONFIG
    
    processor = DataProcessor(DATA_CONFIG)
    train_loader, test_loader = processor.create_dataloaders()
    
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 查看数据形状
    for batch in train_loader:
        features, targets, timestamps = batch
        print(f"JG特征形状: {features['jg'].shape}")
        print(f"AQ特征形状: {features['aq'].shape}")
        print(f"J特征形状: {features['j1_j20'].shape}（已排除目标点）")
        print(f"目标形状: {targets.shape}")
        break

