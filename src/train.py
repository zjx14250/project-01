import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from models import get_loss, get_model
import pickle
import yaml
import uuid
from datetime import datetime
import shutil
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

# 1. 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
parser.add_argument('--override', type=str, default=None, help='覆盖配置项，格式如 key1=val1,key2=val2')
parser.add_argument('--exp_id', type=str, default=None, help='实验ID（可选，若不指定则自动生成）')
args = parser.parse_args()

# 2. 读取配置
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# 3. 解析并应用 override
def apply_override(config, override_str):
    if not override_str:
        return config
    for item in override_str.split(','):
        key, val = item.split('=')
        # 支持嵌套key，如 model.backbone
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        # 自动类型转换
        try:
            val = eval(val)
        except:
            pass
        d[keys[-1]] = val
    return config

config = apply_override(config, args.override)

# 取参数
model_name = config['model']
data_path = config['data_path']
market_name = config['market_name']
relation_name = config['relation_name']
stock_num = config['stock_num']
lookback_length = config['lookback_length']
epochs = config['epochs']
valid_index = config['valid_index']
test_index = config['test_index']
fea_num = config['fea_num']
market_num = config['market_num']
steps = config['steps']
learning_rate = config['learning_rate']
alpha = config['alpha']
scale_factor = config['scale_factor']
activation = config['activation']
patience = config['patience']

dataset_path = '../dataset/' + market_name
if market_name == "SP500":
    data = np.load('../dataset/SP500/SP500.npy')
    data = data[:, 915:, :]
    price_data = data[:, :, -1]
    mask_data = np.ones((data.shape[0], data.shape[1]))
    eod_data = data
    gt_data = np.zeros((data.shape[0], data.shape[1]))
    for ticket in range(0, data.shape[0]):
        for row in range(1, data.shape[1]):
            gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                   data[ticket][row - steps][-1]
    stock_num = data.shape[0]
    print(f"调整stock_num为SP500实际股票数量: {stock_num}")
    config['stock_num'] = stock_num
else:
    with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
        eod_data = pickle.load(f)
    with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
        mask_data = pickle.load(f)
    with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
        gt_data = pickle.load(f)
    with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
        price_data = pickle.load(f)

trade_dates = mask_data.shape[1]
model = get_model(
    model_name,
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

# 生成实验ID和时间戳
if args.exp_id is not None:
    exp_id = args.exp_id
else:
    exp_id = str(uuid.uuid4())[:8]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
exp_dir = os.path.join('experiments', f"{exp_id}_{timestamp}")
os.makedirs(exp_dir, exist_ok=True)

# 复制配置文件到实验文件夹
with open(os.path.join(exp_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True)

# 日志文件路径
log_path = os.path.join(exp_dir, 'train.log')

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)

# 打印最终配置到日志
print("===== 最终训练配置 =====")
print(yaml.dump(config, allow_unicode=True))
print("======================")

def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(

                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = lookback_length
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

# 早停机制参数
patience = 5  # 容忍epoch数
no_improve_count = 0  # 未提升计数

for epoch in range(epochs):
    print("epoch{}##########################################################".format(epoch + 1))
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    for j in range(valid_index - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss = cur_loss
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / (valid_index - lookback_length - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - lookback_length - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - lookback_length - steps + 1)
    print('Train : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(val_loss, val_reg_loss, val_rank_loss))

    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{:.2e}  =  {:.2e} + alpha*{:.2e}'.format(test_loss, test_reg_loss, test_rank_loss))

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf
        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
        no_improve_count = 0  # 有提升，重置计数
    else:
        no_improve_count += 1  # 无提升，计数+1

    print('Valid performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(val_perf['mse'], val_perf['IC'],
                                                     val_perf['RIC'], val_perf['prec_10'], val_perf['sharpe5']))
    print('Test performance:\n', 'mse:{:.2e}, IC:{:.2e}, RIC:{:.2e}, prec@10:{:.2e}, SR:{:.2e}'.format(test_perf['mse'], test_perf['IC'],
                                                                            test_perf['RIC'], test_perf['prec_10'], test_perf['sharpe5']), '\n\n')

    # 早停判断
    if no_improve_count >= patience:
        print(f"验证集loss已连续{patience}个epoch未提升，提前终止训练。")
        print(f"最佳验证集loss: {best_valid_loss:.4e}")
        print(f"最佳验证集性能: {best_valid_perf}")
        print(f"最佳测试集性能: {best_test_perf}")
        break
# 可视化
# 1. 加载最佳模型权重
model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
model.eval()

# 2. 用测试集跑一遍，收集预测和真实值
all_preds = []
all_gts = []
all_masks = []

with torch.no_grad():
    for cur_offset in range(test_index - lookback_length - steps + 1, trade_dates - lookback_length - steps + 1):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(cur_offset)
        )
        prediction = model(data_batch)
        # 这里假设 prediction 和 gt_batch shape 都是 [stock_num, 1]
        all_preds.append(prediction.cpu().numpy())
        all_gts.append(gt_batch.cpu().numpy())
        all_masks.append(mask_batch.cpu().numpy())

# 3. 拼接为时间序列
all_preds = np.concatenate(all_preds, axis=1)  # shape: [stock_num, 时间]
all_gts = np.concatenate(all_gts, axis=1)
all_masks = np.concatenate(all_masks, axis=1)

# 4. 画图（前8个股票，4行2列子图）
plt.figure(figsize=(16, 20))
for i in range(8):
    plt.subplot(4, 2, i+1)
    plt.plot(all_preds[i], label='Predicted')
    plt.plot(all_gts[i], label='Ground Truth')
    plt.title(f'Stock {i} Prediction vs Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
plt.savefig(os.path.join(exp_dir, 'test_timeseries_8stocks.png'))
plt.close()

# 5. 保存前8个股票的序列数据到CSV
time_steps = all_preds.shape[1]
time_indices = np.arange(time_steps)

# 创建一个DataFrame存储所有数据
csv_data = {'Time': time_indices}

# 添加每个股票的预测和真实值列
for i in range(8):
    csv_data[f'Stock {i} Ground Truth'] = all_gts[i]
    csv_data[f'Stock {i} Prediction'] = all_preds[i]

# 创建DataFrame并保存
series_df = pd.DataFrame(csv_data)
series_df.to_csv(os.path.join(exp_dir, 'stock_series.csv'), index=False)
print(f"股票时间序列数据已保存到 {os.path.join(exp_dir, 'stock_series.csv')}")
