# 配置文件：基坑支护桩水平位移预测模型
import torch

# 数据配置
DATA_CONFIG = {
    # Keep the monitoring dataset local; this repository does not include data.
    'file_path': 'data/preprocessed_data_no_normalization.xlsx',
    'target_point': 'J19',  # 目标预测点
    'test_size': 0.2,      # 测试集比例
    'seq_length': 7,      # 时间序列长度
    'batch_size': 32,      # 批次大小
    'shuffle': True,       # 训练集是否打乱
    'random_state': 42,    # 随机种子
    'use_diff': False,     # 是否使用差分处理非平稳性（已关闭）
    'diff_order': 1,       # 差分阶数（1:一阶差分, 2:二阶差分）
    'use_rolling_window': False  # 是否使用滚动窗口验证
}

# GCN配置（简化）
GCN_CONFIG = {
    'num_nodes': 20,       # J1-J20节点数（扩展）
    'input_dim': 1,        # 每个节点的输入维度
    'hidden_dim': 32,      # GCN隐藏层维度（简化）        原本32被我调整为64
    'output_dim': 32,      # GCN输出维度（简化）
    'num_layers': 1,       # GCN层数（简化）
    'dropout': 0.3         # Dropout率                 原本0.3改为0.2
}

# TKAN配置（简化）
TKAN_CONFIG = {
    'jg_input_dim': 11,    # JG1-JG11的维度
    'aq_input_dim': 17,    # AQ1-AQ17的维度
    'j_input_dim': 20,     # J1-J20的维度
    'hidden_dim': 64,      # TKAN隐藏层维度（简化）                         原本为32被我调整为64
    'num_layers': 1,       # TKAN层数（简化）
    'output_dim': 32,      # TKAN输出维度（简化）
    'dropout': 0.3         # Dropout率                 原本0.3改为0.2
}

# Transformer配置（简化）
TRANSFORMER_CONFIG = {
    'input_dim': 32,       # GCN输出维度
    'hidden_dim': 48,      # Transformer隐藏层维度（简化）         原本48我改为64
    'num_heads': 2,        # 注意力头数（简化）
    'num_layers': 1,       # Transformer层数（简化）
    'dropout': 0.3         # Dropout率                 原本0.3改为0.2
}

''' BiLSTM配置（新增）
BILSTM_CONFIG = {
    'hidden_dim': 32,      # BiLSTM隐藏层维度
    'num_layers': 1,       # BiLSTM层数
    'output_dim': 32,      # BiLSTM输出维度
    'dropout': 0.2        # Dropout率                 原本0.3改为0.2
}
'''

# 训练配置
TRAIN_CONFIG = {
    'learning_rate': 0.001,    # 学习率
    'weight_decay': 1e-4,       # 权重衰减         原本是1e-4
    'epochs': 100,              # 训练轮数
    'patience': 20,             # 早停 patience                               原本为10被我调整为50
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
}

# 结果保存配置
RESULT_CONFIG = {
    'save_path': 'outputs/prediction_results.xlsx',  # 预测结果保存路径
    'fig_save_path': 'outputs/prediction_figures/'   # 可视化结果保存路径
}

