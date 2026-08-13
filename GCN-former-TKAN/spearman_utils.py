import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class SpearmanCalculator:
    """
    Spearman相关性计算器
    用于计算变量之间的非线性相关性
    Spearman相关性可以捕捉非线性相关性，且scikit-learn内置支持
    """
    
    def __init__(self, alpha=0.6, c=15):
        """
        初始化Spearman计算器
        
        Args:
            alpha: 控制划分网格的灵活性（0 < alpha < 1）
            c: 控制最大网格数量（c > 0）
        """
        self.alpha = alpha
        self.c = c
    
    def calculate_spearman(self, x, y):
        """
        计算两个变量之间的Spearman相关性值
        
        Args:
            x: 第一个变量
            y: 第二个变量
        
        Returns:
            corr: 相关性值（-1到1之间）
        """
        # 计算Spearman相关性
        corr, _ = spearmanr(x, y)
        return abs(corr)  # 取绝对值，范围0-1
    
    def calculate_spearman_matrix(self, data):
        """
        计算数据集中所有变量之间的Spearman矩阵
        
        Args:
            data: 输入数据，形状为(n_samples, n_features)
        
        Returns:
            spearman_matrix: Spearman相关性矩阵
        """
        n_features = data.shape[1]
        mic_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    # 对角线元素设为0（避免自环）
                    mic_matrix[i, j] = 0
                elif i < j:
                    # 计算Spearman值
                    mic_value = self.calculate_spearman(data[:, i], data[:, j])
                    mic_matrix[i, j] = mic_value
                    mic_matrix[j, i] = mic_value  # 利用对称性
        
        return mic_matrix
    
    def create_adjacency_matrix(self, spearman_matrix, threshold=0.6):
        """
        根据Spearman矩阵创建邻接矩阵
        
        Args:
            spearman_matrix: Spearman相关性矩阵
            threshold: 相关性阈值（大于阈值的设为1，否则设为0）
        
        Returns:
            adj_matrix: 邻接矩阵（0-1矩阵）
        """
        adj_matrix = np.where(spearman_matrix > threshold, 1.0, 0.0)
        return adj_matrix

class SpearmanAdjacencyMatrixGenerator:
    """
    基于Spearman相关性的邻接矩阵生成器
    用于生成GCN模型所需的邻接矩阵
    """
    
    def __init__(self, threshold=0.6, alpha=0.6, c=15):
        """
        初始化邻接矩阵生成器
        
        Args:
            threshold: 相关性阈值
            alpha: Spearman计算参数
            c: Spearman计算参数
        """
        self.threshold = threshold
        self.spearman_calculator = SpearmanCalculator(alpha=alpha, c=c)
    
    def generate_adjacency_matrix(self, data, feature_columns):
        """
        生成基于Spearman相关性的邻接矩阵
        
        Args:
            data: 包含所有特征的数据
            feature_columns: 用于计算Spearman的特征列名
        
        Returns:
            adj_matrix: 基于Spearman的邻接矩阵
            spearman_matrix: 原始Spearman相关性矩阵
        """
        # 提取指定特征
        feature_data = data[feature_columns].values
        
        # 计算Spearman矩阵
        spearman_matrix = self.spearman_calculator.calculate_spearman_matrix(feature_data)
        
        # 创建邻接矩阵
        adj_matrix = self.spearman_calculator.create_adjacency_matrix(spearman_matrix, self.threshold)
        
        return adj_matrix, spearman_matrix
    
    def save_matrix_to_file(self, matrix, file_path):
        """
        将矩阵保存到文件
        
        Args:
            matrix: 要保存的矩阵
            file_path: 保存路径
        """
        np.savetxt(file_path, matrix, delimiter=',')
    
    def load_matrix_from_file(self, file_path):
        """
        从文件加载矩阵
        
        Args:
            file_path: 文件路径
        
        Returns:
            matrix: 加载的矩阵
        """
        return np.loadtxt(file_path, delimiter=',')

# 使用示例
if __name__ == "__main__":
    # 示例：计算Spearman相关性并生成邻接矩阵
    from data_processor import DataProcessor
    from config import DATA_CONFIG
    
    # 加载数据
    data_processor = DataProcessor(DATA_CONFIG)
    data = data_processor.load_data()
    
    # 选择J1-J20特征（用于GCN，扩展）
    j1_j20_columns = [f'J{i}' for i in range(1, 21)]
    
    # 生成邻接矩阵
    spearman_generator = SpearmanAdjacencyMatrixGenerator(threshold=0.6)
    adj_matrix, spearman_matrix = spearman_generator.generate_adjacency_matrix(data, j1_j20_columns)
    
    print("Spearman相关性矩阵:")
    print(pd.DataFrame(spearman_matrix, columns=j1_j20_columns, index=j1_j20_columns))
    
    print("\n邻接矩阵（阈值0.6）:")
    print(pd.DataFrame(adj_matrix, columns=j1_j20_columns, index=j1_j20_columns))
    
    # 保存矩阵
    spearman_generator.save_matrix_to_file(adj_matrix, 'mic_adjacency_matrix.csv')
    spearman_generator.save_matrix_to_file(spearman_matrix, 'mic_correlation_matrix.csv')
    print("\n矩阵已保存到文件")

