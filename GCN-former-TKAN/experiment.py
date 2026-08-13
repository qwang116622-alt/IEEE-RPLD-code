import torch
import pandas as pd
import numpy as np
import os
from config import DATA_CONFIG, GCN_CONFIG, TKAN_CONFIG, TRANSFORMER_CONFIG, TRAIN_CONFIG, RESULT_CONFIG
from data_processor import DataProcessor, PileDisplacementDataset
from models import PileDisplacementModel
from evaluator import ModelEvaluator
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ExperimentManager:
    """
    实验管理器
    用于运行消融实验和对比实验，比较不同模型架构的性能
    """
    
    def __init__(self, configs, result_dir="experiment_results"):
        """
        初始化实验管理器
        
        Args:
            configs: 实验配置列表，每个配置包含模型名称和对应的配置参数
            result_dir: 实验结果保存目录
        """
        self.configs = configs
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        self.results = []
        
    def run_experiment(self, model_name, model_config):
        """
        运行单个实验
        
        Args:
            model_name: 模型名称
            model_config: 模型配置
        """
        print(f"\n{'='*60}")
        print(f"运行实验: {model_name}")
        print(f"{'='*60}")
        
        # 1. 数据准备
        data_processor = DataProcessor(DATA_CONFIG)
        train_data, test_data = data_processor.split_data()
        train_data, test_data = data_processor.normalize_data(train_data, test_data)
        
        # 2. 创建数据集
        train_dataset = PileDisplacementDataset(
            train_data, DATA_CONFIG['seq_length'], DATA_CONFIG['target_point'], augment=False
        )
        test_dataset = PileDisplacementDataset(
            test_data, DATA_CONFIG['seq_length'], DATA_CONFIG['target_point'], augment=False
        )
        
        # 3. 创建DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False, drop_last=False
        )
        
        # 4. 创建模型
        model = PileDisplacementModel(**model_config)
        print(f"模型总参数: {sum(p.numel() for p in model.parameters())}")
        
        # 5. 模型训练
        device = TRAIN_CONFIG['device']
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay']
        )
        criterion = torch.nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
            # 训练
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                features, y_true, _ = batch
                features = {k: v.to(device) for k, v in features.items()}
                y_true = y_true.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                y_pred = model(features)
                loss = criterion(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * y_true.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    features, y_true, _ = batch
                    features = {k: v.to(device) for k, v in features.items()}
                    y_true = y_true.to(device).unsqueeze(1)
                    y_pred = model(features)
                    loss = criterion(y_pred, y_true)
                    val_loss += loss.item() * y_true.size(0)
            
            val_loss = val_loss / len(test_loader.dataset)
            val_losses.append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{self.result_dir}/{model_name}_best_model.pth")
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{TRAIN_CONFIG['epochs']}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"早停！在第 {epoch} 轮停止训练")
                break
        
        # 6. 模型评估
        model.load_state_dict(torch.load(f"{self.result_dir}/{model_name}_best_model.pth"))
        evaluator = ModelEvaluator(RESULT_CONFIG)
        
        if hasattr(data_processor, 'scalers') and 'global' in data_processor.scalers:
            feature_columns = data_processor.scalers['global'].feature_names_in_.tolist()
            use_diff = DATA_CONFIG.get('use_diff', False)
            evaluator.set_scaler(data_processor.scalers['global'], feature_columns, use_diff, getattr(data_processor, 'last_values', None))
        
        # 评估测试集
        test_metrics, test_predictions, test_targets, test_timestamps = evaluator.evaluate(model, test_loader, device)
        
        # 保存结果
        result = {
            'model_name': model_name,
            **test_metrics,
            'best_val_loss': best_val_loss,
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'params': sum(p.numel() for p in model.parameters())
        }
        
        # 保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title(f'{model_name} Loss Curve')
        plt.legend()
        plt.savefig(f"{self.result_dir}/{model_name}_loss_curve.png", dpi=300)
        plt.close()
        
        return result
    
    def run_all_experiments(self):
        """
        运行所有实验
        """
        for config in self.configs:
            model_name = config.pop('model_name')
            result = self.run_experiment(model_name, config)
            self.results.append(result)
        
        # 保存实验结果
        self.save_results()
        
    def save_results(self):
        """
        保存实验结果到CSV文件
        """
        results_df = pd.DataFrame(self.results)
        results_df = results_df[['model_name', 'MAE', 'MSE', 'RMSE', 'R2', 'best_val_loss', 'train_loss', 'val_loss', 'params']]
        results_df.to_csv(f"{self.result_dir}/experiment_results.csv", index=False)
        
        print(f"\n{'='*60}")
        print("实验结果汇总")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
        
        # 可视化比较
        self.visualize_results(results_df)
    
    def visualize_results(self, results_df):
        """
        可视化实验结果
        """
        # 按R2排序
        results_df = results_df.sort_values(by='R2', ascending=False)
        
        # 绘制R2比较图
        plt.figure(figsize=(12, 6))
        plt.bar(results_df['model_name'], results_df['R2'], color='skyblue')
        plt.xlabel('模型架构')
        plt.ylabel('R² 分数')
        plt.title('不同模型架构的R²分数比较')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/r2_comparison.png", dpi=300)
        plt.close()
        
        # 绘制RMSE比较图
        plt.figure(figsize=(12, 6))
        plt.bar(results_df['model_name'], results_df['RMSE'], color='lightgreen')
        plt.xlabel('模型架构')
        plt.ylabel('RMSE')
        plt.title('不同模型架构的RMSE比较')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/rmse_comparison.png", dpi=300)
        plt.close()
        
        # 绘制参数数量比较
        plt.figure(figsize=(12, 6))
        plt.bar(results_df['model_name'], results_df['params'], color='salmon')
        plt.xlabel('模型架构')
        plt.ylabel('参数数量')
        plt.title('不同模型架构的参数数量比较')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/params_comparison.png", dpi=300)
        plt.close()


def main():
    """
    主函数，定义实验配置并运行所有实验
    """
    # 定义实验配置 - 消融实验和对比实验
    experiment_configs = [
        # 1. 基线模型：完整架构 (GCN→Transformer→TKAN)
        {
            'model_name': 'baseline_full',
            'gcn_config': GCN_CONFIG,
            'tkan_config': TKAN_CONFIG,
            'transformer_config': TRANSFORMER_CONFIG,
            'adj_matrix': None
        },
        # 2. 消融实验：无GCN
        {
            'model_name': 'no_gcn',
            'gcn_config': {'num_nodes': 20, 'input_dim': 1, 'hidden_dim': 32, 'output_dim': 32, 'num_layers': 0, 'dropout': 0.3},
            'tkan_config': TKAN_CONFIG,
            'transformer_config': TRANSFORMER_CONFIG,
            'adj_matrix': None
        },
        # 3. 消融实验：无Transformer
        {
            'model_name': 'no_transformer',
            'gcn_config': GCN_CONFIG,
            'tkan_config': TKAN_CONFIG,
            'transformer_config': {'input_dim': 32, 'hidden_dim': 48, 'num_heads': 2, 'num_layers': 0, 'dropout': 0.3},
            'adj_matrix': None
        },
        # 4. 消融实验：不同TKAN层数
        {
            'model_name': 'tkan_2layers',
            'gcn_config': GCN_CONFIG,
            'tkan_config': {**TKAN_CONFIG, 'num_layers': 2},
            'transformer_config': TRANSFORMER_CONFIG,
            'adj_matrix': None
        },
        # 5. 消融实验：不同Transformer层数
        {
            'model_name': 'transformer_2layers',
            'gcn_config': GCN_CONFIG,
            'tkan_config': TKAN_CONFIG,
            'transformer_config': {**TRANSFORMER_CONFIG, 'num_layers': 2},
            'adj_matrix': None
        },
        # 6. 消融实验：不同GCN层数
        {
            'model_name': 'gcn_2layers',
            'gcn_config': {**GCN_CONFIG, 'num_layers': 2},
            'tkan_config': TKAN_CONFIG,
            'transformer_config': TRANSFORMER_CONFIG,
            'adj_matrix': None
        },
    ]
    
    # 初始化实验管理器并运行所有实验
    experiment_manager = ExperimentManager(experiment_configs)
    experiment_manager.run_all_experiments()


if __name__ == "__main__":
    main()

