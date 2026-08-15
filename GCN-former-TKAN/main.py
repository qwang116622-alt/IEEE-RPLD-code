# D:\AiSoftware\Anaconda3\envs\PyTorch_cpu\python.exe main.py --load_model  加载最佳模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import argparse

# 导入自定义模块
from config import DATA_CONFIG, GCN_CONFIG, TKAN_CONFIG, TRANSFORMER_CONFIG, TRAIN_CONFIG, RESULT_CONFIG
from data_processor import DataProcessor, get_pile_feature_columns
from models import PileDisplacementModel
from evaluator import ModelEvaluator

def train(model, train_loader, optimizer, criterion, device):
    """
    训练模型
    
    Args:
        model: 待训练模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 运行设备
    
    Returns:
        train_loss: 训练损失
    """
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        # 处理差分数据返回的额外值
        if len(batch) == 4:
            features, y_true, _, prev_values = batch
        else:
            features, y_true, _ = batch
            prev_values = None

        # 将数据转移到设备
        features = {
            k: v.to(device) for k, v in features.items()
        }
        y_true = y_true.to(device).unsqueeze(1)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 模型预测
        y_pred = model(features)
        
        # 计算损失
        loss = criterion(y_pred, y_true)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新权重
        optimizer.step()
        
        # 累计损失
        train_loss += loss.item() * y_true.size(0)
    
    # 计算平均损失
    train_loss = train_loss / len(train_loader.dataset)
    
    return train_loss

def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    Args:
        model: 待验证模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 运行设备
    
    Returns:
        val_loss: 验证损失
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            # 处理差分数据返回的额外值
            if len(batch) == 4:
                features, y_true, _, prev_values = batch
            else:
                features, y_true, _ = batch
                prev_values = None

            # 将数据转移到设备
            features = {
                k: v.to(device) for k, v in features.items()
            }
            y_true = y_true.to(device).unsqueeze(1)
            
            # 模型预测
            y_pred = model(features)
            
            # 计算损失
            loss = criterion(y_pred, y_true)
            
            # 累计损失
            val_loss += loss.item() * y_true.size(0)
    
    # 计算平均损失
    val_loss = val_loss / len(val_loader.dataset)
    
    return val_loss

def main():
    """
    主函数
    实现模型的完整训练和预测流程
    """
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='基坑支护桩水平位移预测模型')
    parser.add_argument('--load_model', action='store_true',
                        help='直接加载 outputs/best_model.pth 进行预测，跳过训练步骤')
    args = parser.parse_args()
    
    print("=" * 60)
    print("基坑支护桩水平位移预测模型")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n1. 加载配置...")
    print(f"目标预测点: {DATA_CONFIG['target_point']}")
    print(f"时间序列长度: {DATA_CONFIG['seq_length']}")
    print(f"批次大小: {DATA_CONFIG['batch_size']}")
    print(f"训练轮数: {TRAIN_CONFIG['epochs']}")
    print(f"运行设备: {TRAIN_CONFIG['device']}")
    print(f"加载模型: {args.load_model}")
    
    # 2. 数据准备
    print("\n2. 数据准备...")
    data_processor = DataProcessor(DATA_CONFIG)
    train_loader, test_loader = data_processor.create_dataloaders()
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"训练集批次: {len(train_loader)}")
    print(f"测试集批次: {len(test_loader)}")
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    
    # 尝试加载预计算的邻接矩阵
    import numpy as np
    import os
    adj_matrix_path = 'data/mic_adjacency_matrix.csv'
    adj_matrix = None
    
    if os.path.exists(adj_matrix_path):
        print(f"加载预计算的邻接矩阵: {adj_matrix_path}")
        adj_matrix = np.loadtxt(adj_matrix_path, delimiter=',')
        print(f"邻接矩阵形状: {adj_matrix.shape}")
    else:
        print("未找到预计算邻接矩阵，将从本地数据计算 Spearman 邻接矩阵")
        from spearman_utils import SpearmanAdjacencyMatrixGenerator
        data = data_processor.data
        j_columns = get_pile_feature_columns(DATA_CONFIG['target_point'])
        generator = SpearmanAdjacencyMatrixGenerator(threshold=0.6)
        adj_matrix, _ = generator.generate_adjacency_matrix(data, j_columns)
    
    # 创建模型，直接传递邻接矩阵
    model = PileDisplacementModel(GCN_CONFIG, TKAN_CONFIG, TRANSFORMER_CONFIG, adj_matrix=adj_matrix)
    print(f"模型总参数: {sum(p.numel() for p in model.parameters())}")
    
    # 将模型转移到设备
    device = TRAIN_CONFIG['device']
    model.to(device)
    
    # 初始化训练损失记录
    train_losses = []
    
    # 4. 模型训练（如果未指定加载模型）
    if not args.load_model:
        print("\n4. 模型训练...")
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG['learning_rate'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )
        
        criterion = nn.MSELoss()
        
        # 学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=TRAIN_CONFIG['scheduler_factor'],
            patience=TRAIN_CONFIG['scheduler_patience'],
        )
        
        # 早停机制
        best_train_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, TRAIN_CONFIG['epochs'] + 1):
            # 训练
            train_loss = train(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            
            # The manuscript monitors training MSE for scheduling and early
            # stopping. The chronological test set remains untouched until
            # final evaluation.
            scheduler.step(train_loss)
            
            # 早停检查
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(RESULT_CONFIG['model_path']), exist_ok=True)
                torch.save(model.state_dict(), RESULT_CONFIG['model_path'])
            else:
                patience_counter += 1
            
            # 打印训练进度
            print(f"Epoch [{epoch}/{TRAIN_CONFIG['epochs']}], "
                  f"Train Loss: {train_loss:.6f}, "
                   f"Best Train Loss: {best_train_loss:.6f}")
            
            # 早停
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f"早停！在第 {epoch} 轮停止训练")
                break
        
        end_time = time.time()
        print(f"\n训练完成！耗时: {end_time - start_time:.2f} 秒")
    else:
        print("\n4. 跳过训练，直接加载模型...")
    
    # 5. 加载最佳模型
    print("\n5. 加载最佳模型...")
    model.load_state_dict(torch.load(RESULT_CONFIG['model_path'], weights_only=True))
    
    # 7. 模型评估
    print("\n6. 模型评估...")
    evaluator = ModelEvaluator(RESULT_CONFIG)
    
    # 设置归一化器，用于反归一化
    if hasattr(data_processor, 'scalers') and 'global' in data_processor.scalers:
        # 获取特征列名列表
        feature_columns = data_processor.scalers['global'].feature_names_in_.tolist()
        use_diff = DATA_CONFIG.get('use_diff', False)
        last_values = data_processor.last_values if use_diff else None
        evaluator.set_scaler(data_processor.scalers['global'], feature_columns, use_diff, last_values)
        print("已设置归一化器，将进行反归一化操作")
        print(f"目标预测点: {DATA_CONFIG['target_point']}")
        if use_diff:
            print(f"差分模式: {DATA_CONFIG.get('diff_order', 1)}阶差分")
    
    # 7.1 评估训练集（按原始顺序）
    print("\n6.1 评估训练集...")
    # 临时保存原始shuffle配置
    original_shuffle = DATA_CONFIG['shuffle']
    # 设置为不打乱顺序
    DATA_CONFIG['shuffle'] = False
    # 重新创建dataloader，此时训练集将按原始顺序输出
    train_loader_original, test_loader = data_processor.create_dataloaders()
    train_metrics, train_predictions, train_targets, train_timestamps = evaluator.evaluate(model, train_loader_original, device)
    evaluator.print_metrics(train_metrics)
    # 恢复原始shuffle配置
    DATA_CONFIG['shuffle'] = original_shuffle
    
    # 7.2 评估测试集
    print("\n6.2 评估测试集...")
    # 使用已经创建好的测试集loader，它本来就是按顺序的
    test_metrics, test_predictions, test_targets, test_timestamps = evaluator.evaluate(model, test_loader, device, save_before_evaluate=True)
    evaluator.print_metrics(test_metrics)
    
    # 8. 保存结果
    print("\n7. 保存结果...")
    
    # 8.1 保存训练集结果
    evaluator.save_results(train_timestamps, train_targets, train_predictions, DATA_CONFIG['target_point'], result_type='训练集')
    
    # 8.2 保存测试集结果
    evaluator.save_results(test_timestamps, test_targets, test_predictions, DATA_CONFIG['target_point'], result_type='测试集')
    
    # 9. 可视化
    print("\n8. 生成可视化结果...")
    
    # 9.1 生成测试集可视化结果
    evaluator.visualize_all(test_timestamps, test_targets, test_predictions, DATA_CONFIG['target_point'], train_losses, None)
    
    # 10. 保存评估指标
    print("\n9. 保存评估指标...")
    evaluator.save_metrics(test_metrics, DATA_CONFIG['target_point'])  # 保存测试集指标
    
    print("\n" + "=" * 60)
    print("项目完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

