import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file_path = 'preprocessed_data_no_normalization.xlsx'
data = pd.read_excel(file_path, index_col='index')
data.index = pd.to_datetime(data.index)

# 分析目标点J3的分布情况
target_col = 'J3'
target_data = data[target_col]

print(f"目标点{target_col}的统计信息：")
print(target_data.describe())

# 绘制标签分布图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(target_data, bins=20, edgecolor='black')
plt.title(f'{target_col} 分布图')
plt.xlabel('位移值')
plt.ylabel('频数')

# 绘制时间序列图
plt.subplot(1, 2, 2)
plt.plot(data.index, target_data)
plt.title(f'{target_col} 时间序列')
plt.xlabel('时间')
plt.ylabel('位移值')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('target_distribution.png')
print("\n标签分布图已保存到target_distribution.png")

# 检查数据是否有缺失值
print(f"\n数据缺失值情况：")
print(data.isnull().sum().head(10))

# 检查数据类型
print(f"\n数据类型：")
print(data.dtypes.head(10))

