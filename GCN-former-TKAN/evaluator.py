import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 配置 matplotlib 支持中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    """
    模型评估器
    用于评估模型性能、保存结果和可视化
    """
    
    def __init__(self, result_config):
        """
        初始化评估器
        
        Args:
            result_config: 结果配置字典
        """
        self.result_config = result_config
        # 创建保存目录
        os.makedirs(result_config['fig_save_path'], exist_ok=True)
        self.scaler = None
    
    def set_scaler(self, scaler, feature_columns=None, use_diff=False, last_values=None):
        """
        设置归一化器，用于反归一化

        Args:
            scaler: 训练时使用的归一化器
            feature_columns: 特征列名列表，用于定位目标特征
            use_diff: 是否使用了差分处理
            last_values: 差分前的最后一个值，用于恢复
        """
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.use_diff = use_diff
        self.last_values = last_values
    
    def evaluate(self, model, test_loader, device, inverse_norm=True, save_before_evaluate=False):
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            device: 运行设备
            inverse_norm: 是否对结果进行反归一化
            save_before_evaluate: 是否在计算指标前保存预测结果
        
        Returns:
            metrics: 评估指标字典
            predictions: 预测值列表
            targets: 真实值列表
            timestamps: 时间戳列表
        """
        model.eval()
        predictions = []
        targets = []
        timestamps = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 处理差分数据返回的额外值
                if len(batch) == 4:
                    features, y_true, batch_timestamps, prev_values = batch
                else:
                    features, y_true, batch_timestamps = batch
                    prev_values = None

                # 将数据转移到设备
                features = {
                    k: v.to(device) for k, v in features.items()
                }
                y_true = y_true.to(device)

                # 模型预测
                y_pred = model(features)

                # 保存结果
                predictions.extend(y_pred.cpu().numpy().flatten())
                targets.extend(y_true.cpu().numpy().flatten())
                timestamps.extend(batch_timestamps)

                # 保存前一个值用于差分恢复
                if prev_values is not None:
                    if not hasattr(self, 'prev_values_list'):
                        self.prev_values_list = []
                    self.prev_values_list.extend(prev_values.cpu().numpy().flatten())
        
        # 临时保存归一化后的数据用于计算指标
        norm_targets = targets.copy()
        norm_predictions = predictions.copy()
        
        # 如果需要反归一化，使用保存的归一化器
        if inverse_norm and self.scaler is not None:
            # 注意：归一化器是对整个数据集进行归一化的，但我们只需要反归一化目标特征
            # 我们需要提取目标特征的min_和scale_值
            try:
                # 反归一化公式：x = x_scaled * (max - min) + min
                # 获取目标特征的 min 和 max
                if hasattr(self, 'feature_columns') and self.feature_columns is not None:
                    # 从配置中获取目标特征
                    target_col = self.result_config.get('target_point', 'J3')
                    if target_col in self.feature_columns:
                        target_index = self.feature_columns.index(target_col)
                        min_val = self.scaler.data_min_[target_index]
                        max_val = self.scaler.data_max_[target_index]
                    else:
                        raise ValueError(f"目标列 {target_col} 不在特征列中")
                else:
                    # 如果没有特征列信息，尝试从其他途径获取
                    # 假设目标特征是 J 系列中的某个特征（J3, J4等）
                    # 找到第一个以 'J' 开头但不是 'JG' 或 'AQ' 的列
                    target_index = None
                    for idx, col_name in enumerate(self.scaler.feature_names_in_):
                        if col_name.startswith('J') and not col_name.startswith('JG') and not col_name.startswith('AQ'):
                            target_index = idx
                            break
                    
                    if target_index is None:
                        raise ValueError("无法确定目标特征索引")
                    
                    min_val = self.scaler.data_min_[target_index]
                    max_val = self.scaler.data_max_[target_index]
                
                # 手动反归一化
                targets_denorm = np.array(targets) * (max_val - min_val) + min_val
                predictions_denorm = np.array(predictions) * (max_val - min_val) + min_val

                # 如果使用了差分，需要累加恢复原始值
                if self.use_diff and hasattr(self, 'prev_values_list') and len(self.prev_values_list) > 0:
                    # 使用每个样本的前一个值进行累加
                    prev_values = np.array(self.prev_values_list[:len(predictions)])

                    # 累加恢复：predicted_value = prev_value + diff_prediction
                    targets = targets_denorm + prev_values
                    predictions = predictions_denorm + prev_values

                    print(f"差分恢复完成：累加前一个值")
                else:
                    targets = targets_denorm
                    predictions = predictions_denorm

                print(f"反归一化完成：目标特征索引={target_index}, min={min_val:.4f}, max={max_val:.4f}")
            except Exception as e:
                print(f"反归一化失败: {e}")
                print("将使用归一化后的值")
        
        # 在计算指标前保存预测结果
        if save_before_evaluate:
            # 确保保存目录存在
            os.makedirs('outputs/prediction_results', exist_ok=True)
            # 构建保存路径
            save_path = os.path.join('outputs/prediction_results', 'test_predictions.xlsx')
            # 保存结果
            self.save_results(timestamps, targets, predictions, self.result_config.get('target_point', 'J3'), '测试集', save_path)
            print("预测结果已保存到 outputs/prediction_results/test_predictions.xlsx")
        
        # 计算评估指标（使用反归一化后的数据）
        metrics = self._calculate_metrics(targets, predictions)
        
        return metrics, predictions, targets, timestamps
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            metrics: 评估指标字典
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        #mae = mean_absolute_error(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        #mse = mean_squared_error(y_true, y_pred)
        #rmse = np.sqrt(mse)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics
    
    def save_results(self, timestamps, y_true, y_pred, target_point, result_type='测试集', save_path=None):
        """
        保存预测结果到Excel
        
        Args:
            timestamps: 时间戳列表
            y_true: 真实值列表
            y_pred: 预测值列表
            target_point: 目标监测点
            result_type: 结果类型，如'测试集'或'训练集'
            save_path: 自定义保存路径，如果为None则使用默认路径
        """
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            '监测时间': timestamps,
            '变形值': y_true,
            '预测值': y_pred
        })
        
        # 保存到Excel
        if save_path:
            results_df.to_excel(save_path, index=False)
            print(f"预测结果已保存到: {save_path}")
        else:
            if result_type == '测试集':
                results_df.to_excel(self.result_config['save_path'], index=False)
                print(f"测试集预测结果已保存到: {self.result_config['save_path']}")
            else:
                train_save_path = self.result_config['save_path'].replace('.xlsx', '_train.xlsx')
                results_df.to_excel(train_save_path, index=False)
                print(f"训练集预测结果已保存到: {train_save_path}")
    
    def plot_prediction(self, timestamps, y_true, y_pred, target_point):
        """
        绘制预测值与真实值对比图
        
        Args:
            timestamps: 时间戳列表
            y_true: 真实值列表
            y_pred: 预测值列表
            target_point: 目标监测点
        """
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, y_true, label='真实值', color='blue', linewidth=2)
        plt.plot(timestamps, y_pred, label='预测值', color='red', linestyle='--', linewidth=2)
        plt.xlabel('监测时间')
        plt.ylabel(f'{target_point} 水平位移')
        plt.title(f'{target_point} 水平位移预测结果对比')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.result_config['fig_save_path'], f'{target_point}_prediction.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"预测对比图已保存到: {save_path}")
    
    def plot_loss(self, train_losses, val_losses=None):
        """
        绘制损失值变化曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表（可选）
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失', color='blue')
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='验证损失', color='red')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值 (MSE)')
        plt.title('模型训练损失变化')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.result_config['fig_save_path'], 'loss_curve.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"损失曲线已保存到: {save_path}")
    
    def plot_scatter(self, y_true, y_pred, target_point):
        """
        绘制真实值与预测值的散点图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            target_point: 目标监测点
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color='green')
        # 绘制对角线
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{target_point} 预测值 vs 真实值散点图')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.result_config['fig_save_path'], f'{target_point}_scatter.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"散点图已保存到: {save_path}")
    
    def save_metrics(self, metrics, target_point):
        """
        保存评估指标到文本文件
        
        Args:
            metrics: 评估指标字典
            target_point: 目标监测点
        """
        save_path = os.path.join(self.result_config['fig_save_path'], f'{target_point}_metrics.txt')
        with open(save_path, 'w') as f:
            f.write(f"{target_point} 模型评估指标\n")
            f.write("=" * 30 + "\n")
            for metric_name, metric_value in metrics.items():
                if metric_name == 'MAPE':
                    f.write(f"{metric_name}: {metric_value:.6f}%\n")
                else:
                    f.write(f"{metric_name}: {metric_value:.6f}\n")
        print(f"评估指标已保存到: {save_path}")
    
    def visualize_all(self, timestamps, y_true, y_pred, target_point, train_losses=None, val_losses=None):
        """
        生成所有可视化结果
        
        Args:
            timestamps: 时间戳列表
            y_true: 真实值列表
            y_pred: 预测值列表
            target_point: 目标监测点
            train_losses: 训练损失列表（可选）
            val_losses: 验证损失列表（可选）
        """
        # 绘制预测对比图
        self.plot_prediction(timestamps, y_true, y_pred, target_point)
        
        # 绘制散点图
        self.plot_scatter(y_true, y_pred, target_point)
        
        # 绘制损失曲线
        if train_losses:
            self.plot_loss(train_losses, val_losses)
    
    def print_metrics(self, metrics):
        """
        打印评估指标
        
        Args:
            metrics: 评估指标字典
        """
        print("\n模型评估指标:")
        print("=" * 30)
        for metric_name, metric_value in metrics.items():
            if metric_name == 'MAPE':
                print(f"{metric_name}: {metric_value:.6f}%")
            else:
                print(f"{metric_name}: {metric_value:.6f}")

