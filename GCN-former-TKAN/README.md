# GCN-former-TKAN

本仓库实现论文中的深基坑围护桩水平位移一步预测流程：以监测点间的空间关联为图结构，结合 GCN、Transformer 和 TKAN 预测指定桩点的下一时刻水平位移。

> 仓库只包含代码，不包含监测数据、坐标、训练权重、预测表或图件。该工具用于科研复现和辅助分析，不能替代现场巡检、设计复核或工程预警决策。

## 适用对象与任务

适用于拥有按时间排序的基坑监测数据的岩土工程研究人员、监测工程师和数据分析人员。每次预测使用连续 7 个历史时刻的数据，输出一个目标桩点在下一个时刻的位移预测值。

论文对应的输入由 47 个变量组成：11 个 `JG` 特征、17 个 `AQ` 特征，以及除目标桩点外的 19 个围护桩位移特征。代码会自动把 `target_point` 从 `J1`–`J20` 的图节点和输入特征中排除，避免把待预测点作为同期输入而造成数据泄漏。

## 论文对应的默认网络与超参数

| 模块 | 当前默认实现 |
| --- | --- |
| 图构建 | 19 个非目标桩点；Spearman 相关系数阈值 `0.6`；加自环并作对称归一化 |
| GCN | 2 层图卷积，每层 32 维 |
| Transformer | 2 个编码器层、2 个注意力头、64 维隐藏表示、256 维前馈层；输入层归一化和位置编码 |
| TKAN | 2 个时序 KAN 层；每层 3 个 KAN 单元；128 维隐藏状态；三阶 B 样条、网格大小 5；带输入/遗忘/输出门的时间记忆 |
| 预测头 | 最后一时刻 TKAN 表示经线性层输出一个标量 |
| 优化器与损失 | Adam、MSE |
| 初始学习率 | `1e-4` |
| 调度器 | `ReduceLROnPlateau`，因子 `0.5`、耐心值 10 |
| 批量大小 / 最大轮数 | 32 / 150 |
| 早停 | 训练 MSE 连续 15 轮未改善时停止 |
| 时间窗口 | 7 个历史时刻 |

配置集中在 `config.py`。如果为新数据集改动输入列、特征数或模型宽度，应同时检查 `DATA_CONFIG`、`GCN_CONFIG`、`TKAN_CONFIG` 与 `TRANSFORMER_CONFIG` 是否保持一致。

## 数据准备（仅在本地）

在项目目录创建 `data` 文件夹，并把私有 Excel 文件保存为 `data/preprocessed_data_no_normalization.xlsx`，或在 `config.py` 修改 `file_path`。

工作簿要求：

- 每行代表一个监测时刻，按从早到晚排序；
- 有可转换为日期时间的 `index` 列；
- 包含数值列 `JG1`–`JG11`、`AQ1`–`AQ17`、`J1`–`J20`；
- 用作 `target_point` 的列必须属于 `J1`–`J20`；
- 所用列不应含缺失值。

默认采用按时间顺序的 80%/20% 训练—测试划分。归一化器仅用训练期拟合；测试期只在训练结束后用于最终评估。不要把数据、权重或生成结果提交到 GitHub。

目录示例：

```text
GCN-former-TKAN/
├── data/
│   └── preprocessed_data_no_normalization.xlsx  # 私有数据，不上传
├── config.py
├── main.py
├── models.py
└── requirements.txt
```

## 安装与运行

推荐使用 Python 3.10+。本代码使用 Python 3.10.15、PyTorch 2.4.1 进行过结构与随机数据前向/反向验证。

```powershell
cd GCN-former-TKAN
D:\AiSoftware\Anaconda3\envs\PyTorch_cpu\python.exe -m pip install -r requirements.txt
D:\AiSoftware\Anaconda3\envs\PyTorch_cpu\python.exe main.py
```

在 CPU 环境中可直接运行；如需 CUDA，请安装与本机 CUDA 匹配的 PyTorch，并在 `config.py` 中确认设备设置。

要仅评估已经训练好的本地权重，请将权重置于 `outputs/best_model.pth` 后运行：

```powershell
D:\AiSoftware\Anaconda3\envs\PyTorch_cpu\python.exe main.py --load_model
```

## 常用配置

在 `config.py` 调整：

```python
DATA_CONFIG['target_point'] = 'J19'  # 可改为 J1–J20 中任一点
DATA_CONFIG['seq_length'] = 7
DATA_CONFIG['test_size'] = 0.2
```

改变目标点后，代码自动把该点从 19 个桩位移输入和 Spearman 图中排除。若使用预计算邻接矩阵 `data/mic_adjacency_matrix.csv`，它必须是与这 19 个非目标桩点顺序一致的 `19 × 19` 矩阵；否则删除该本地文件，程序会从工作簿计算邻接矩阵。

## 输出与解释

训练得到的权重保存在 `outputs/best_model.pth`。预测表、评价指标和图件也只会保存在本地 `outputs/` 下。输出指标包括 MAE、MAPE、RMSE 和 R²。

指标应仅针对时间上更晚、未参与训练和参数选择的测试期解释。即使误差很低，也应结合传感器质量、施工工况、地质条件和工程安全限值进行独立复核。

## 文件说明

- `models.py`：论文配置的 GCN、位置编码 Transformer 与门控时序 KAN。
- `data_processor.py`：时间切分、仅训练集拟合的归一化、47 维输入序列构建。
- `spearman_utils.py`：Spearman 图邻接矩阵生成。
- `main.py`：训练、学习率调度、早停与最终测试评估。
- `evaluator.py`：反归一化、指标、预测表和图件。
- `experiment.py`：原始实验管理脚本；正式论文配置请以 `main.py` 和 `config.py` 为准。

## 故障排查

- **找不到数据文件**：检查 `DATA_CONFIG['file_path']` 和本地文件名。
- **缺少列**：补齐 `JG1`–`JG11`、`AQ1`–`AQ17`、`J1`–`J20`，且严格使用上述列名。
- **邻接矩阵尺寸错误**：使用 `19 × 19` 的矩阵，且顺序与“除目标点外的 J 列”一致，或让程序自动计算。
- **样本过少**：每个时间段至少需要多于 `seq_length` 的记录；实际训练建议远多于此数量。
- **CUDA 报错**：在 `config.py` 把设备改为 `cpu`，或安装匹配版本的 CUDA PyTorch。

