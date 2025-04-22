import os
import pickle
import numpy as np

market_name = 'NASDAQ'
dataset_path = 'dataset/' + market_name

# 加载eod_data
with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
    eod_data = pickle.load(f)

# 打印eod_data的形状
print(f"eod_data.shape: {eod_data.shape}")

# 检查eod_data中是否存在大于1的值
has_values_greater_than_1 = np.any(eod_data > 1)
max_value = np.max(eod_data)

print(f"eod_data中是否存在大于1的值: {has_values_greater_than_1}")
print(f"eod_data中的最大值: {max_value}")

# 如果存在大于1的值，打印这些值的数量和位置
if has_values_greater_than_1:
    greater_than_1_count = np.sum(eod_data > 1)
    greater_than_1_indices = np.where(eod_data > 1)
    print(f"大于1的值的数量: {greater_than_1_count}")
    print(f"大于1的值的位置示例(前10个): {list(zip(*[i[:10] for i in greater_than_1_indices]))}") 