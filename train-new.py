"""
STGCN-TCN 训练脚本 - 加权损失版本
特点：
1. 多步预测加权损失（近期预测权重更高）
2. 节点流量加权损失（流量大的节点权重更高）
3. 支持分节点标准化数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import sys

# 导入模型
from STGCN_new import STGCN_TCN


class WeightedMSELoss(nn.Module):
    """
    加权MSE损失
    支持：
    1. 时间步加权：近期预测权重更高
    2. 节点流量加权：流量大的节点权重更高
    """
    def __init__(self, pred_len=12, node_weights=None, time_decay='linear'):
        super(WeightedMSELoss, self).__init__()
        self.pred_len = pred_len
        self.node_weights = node_weights
        self.time_decay = time_decay

        # 计算时间步权重
        if time_decay == 'linear':
            # 线性衰减：近期权重高，远期权重低
            self.time_weights = torch.linspace(1.0, 0.5, pred_len)
        elif time_decay == 'exp':
            # 指数衰减
            self.time_weights = torch.exp(-torch.linspace(0, 2, pred_len))
        elif time_decay == 'uniform':
            # 均匀权重
            self.time_weights = torch.ones(pred_len)
        else:
            self.time_weights = torch.ones(pred_len)

    def forward(self, pred, true):
        """
        pred, true: [batch, pred_len, num_nodes]
        """
        batch_size, pred_len, num_nodes = pred.shape

        # 基础MSE
        mse = (pred - true) ** 2  # [batch, pred_len, num_nodes]

        # 时间步加权
        time_weights = self.time_weights.to(pred.device)
        time_weights = time_weights.view(1, pred_len, 1)  # [1, pred_len, 1]
        mse = mse * time_weights

        # 节点加权
        if self.node_weights is not None:
            node_weights = self.node_weights.to(pred.device)
            node_weights = node_weights.view(1, 1, num_nodes)  # [1, 1, num_nodes]
            mse = mse * node_weights

        return mse.mean()


class MultiStepWeightedLoss(nn.Module):
    """
    多步预测加权损失
    对每个预测步分别计算损失并加权求和
    """
    def __init__(self, pred_len=12, step_weights=None):
        super(MultiStepWeightedLoss, self).__init__()
        self.pred_len = pred_len

        if step_weights is None:
            # 默认权重：近期预测更重要
            # 权重从1.0递减到0.5
            self.step_weights = torch.linspace(1.0, 0.5, pred_len)
        else:
            self.step_weights = torch.tensor(step_weights)

    def forward(self, pred, true):
        """
        pred, true: [batch, pred_len, num_nodes]
        """
        batch_size, pred_len, num_nodes = pred.shape

        # 对每个预测步计算损失
        step_losses = []
        for t in range(pred_len):
            step_loss = F.mse_loss(pred[:, t, :], true[:, t, :])
            step_losses.append(step_loss)

        # 加权求和
        step_weights = self.step_weights.to(pred.device)
        total_loss = sum(w * l for w, l in zip(step_weights, step_losses))

        return total_loss / step_weights.sum()


class HuberWeightedLoss(nn.Module):
    """
    Huber损失 + 加权
    对大误差有更好的鲁棒性
    """
    def __init__(self, delta=1.0, pred_len=12, time_decay='linear'):
        super(HuberWeightedLoss, self).__init__()
        self.delta = delta
        self.pred_len = pred_len

        # 时间步权重
        if time_decay == 'linear':
            self.time_weights = torch.linspace(1.0, 0.5, pred_len)
        else:
            self.time_weights = torch.ones(pred_len)

    def forward(self, pred, true):
        """
        pred, true: [batch, pred_len, num_nodes]
        """
        error = pred - true
        abs_error = torch.abs(error)

        # Huber损失
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear

        # 时间步加权
        time_weights = self.time_weights.to(pred.device)
        time_weights = time_weights.view(1, self.pred_len, 1)
        loss = loss * time_weights

        return loss.mean()


def calculate_node_weights(y_train, method='std'):
    """
    计算节点权重
    method:
        'std': 标准差越大权重越高（波动大的节点更重要）
        'mean': 均值越大权重越高（流量大的节点更重要）
        'uniform': 均匀权重
    """
    # y_train: [samples, pred_len, nodes]
    if method == 'std':
        # 计算每个节点的标准差
        node_std = np.std(y_train, axis=(0, 1))  # [nodes]
        weights = node_std / node_std.mean()
    elif method == 'mean':
        # 计算每个节点的均值
        node_mean = np.mean(y_train, axis=(0, 1))  # [nodes]
        weights = node_mean / node_mean.mean()
    else:
        weights = np.ones(y_train.shape[2])

    # 归一化到 [0.5, 1.5] 范围
    weights = 0.5 + (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    return torch.FloatTensor(weights)


def calculate_metrics(pred, true, scaler):
    """计算评估指标（支持分节点标准化）"""
    if isinstance(scaler, tuple) and len(scaler) == 2:
        mean, std = scaler
        if isinstance(mean, np.ndarray) and mean.ndim == 1:
            # 分节点标准化
            pred_real = pred * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
            true_real = true * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
        else:
            # 全局标准化
            pred_real = pred * std + mean
            true_real = true * std + mean
    else:
        pred_real, true_real = pred, true

    # 展平
    pred_flat = pred_real.flatten()
    true_flat = true_real.flatten()

    # MAE
    mae = np.mean(np.abs(pred_flat - true_flat))

    # RMSE
    rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))

    # MAPE
    threshold = 10.0
    mask = true_flat > threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100
    else:
        mape = 0.0

    # R²
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return mae, rmse, mape, r2


def load_data(data_path, batch_size=64):
    """加载数据（支持分节点标准化）"""
    data = np.load(data_path)

    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.FloatTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'])
    adj_matrix = torch.FloatTensor(data['adj_matrix'])

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 支持分节点标准化
    if 'node_mean' in data.files:
        scaler = (data['node_mean'], data['node_std'])
        print("[OK] 使用分节点标准化参数")
    else:
        scaler = (data['mean'].item(), data['std'].item())
        print("[OK] 使用全局标准化参数")

    print(f"数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  邻接矩阵: {adj_matrix.shape}")

    return train_loader, val_loader, test_loader, adj_matrix, scaler, y_train.numpy()


def train_epoch(model, train_loader, adj, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X, adj)
        loss = criterion(output, batch_y)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader, adj, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X, adj)
            loss = criterion(output, batch_y)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def predict_and_plot(model, test_loader, adj, scaler, device, save_path='stgcn_prediction-new.png'):
    """预测并绘图（支持分节点标准化）"""
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X, adj)

            predictions.append(output.cpu().numpy())
            true_values.append(batch_y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    # 反标准化（支持分节点标准化）
    mean, std = scaler
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        # 分节点标准化
        predictions_real = predictions * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
        true_values_real = true_values * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    else:
        # 全局标准化
        predictions_real = predictions * std + mean
        true_values_real = true_values * std + mean

    # 选择节点0可视化
    node_idx = 0
    pred_flat = predictions_real[:, 0, node_idx]
    true_flat = true_values_real[:, 0, node_idx]

    # 移动平均
    window_size = 12
    moving_avg = np.convolve(pred_flat, np.ones(window_size)/window_size, mode='valid')

    # 绘图
    plt.figure(figsize=(14, 6))
    plot_len = min(1200, len(true_flat))

    plt.plot(true_flat[:plot_len], label='True Value', color='blue', linewidth=1.5, alpha=0.7)
    plt.plot(pred_flat[:plot_len], label='Prediction', color='green', linewidth=1,
             linestyle='--', alpha=0.7)

    if len(moving_avg) > 0:
        ma_len = min(plot_len, len(moving_avg))
        plt.plot(range(window_size-1, window_size-1+ma_len), moving_avg[:ma_len],
                label='Moving Average', color='orange', linewidth=1.5, alpha=0.8)

    plt.xlabel('Hour Timesteps', fontsize=12)
    plt.ylabel('Output Value', fontsize=12)
    plt.title('STGCN-TCN: Prediction vs. True Value', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n预测图已保存到: {save_path}")
    plt.close()

    return predictions, true_values


def main():
    # ========== 配置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    data_path = "processed_data-new.npz"  # 使用新的分节点标准化数据
    batch_size = 64
    num_nodes = 307
    hidden_channels = 64
    num_layers = 3
    pred_len = 12
    kernel_size = 3
    num_epochs = 200
    learning_rate = 0.001
    patience = 15

    # 模型配置
    use_tcn = False  # True: 使用TCN模块（较慢但效果可能更好）, False: 使用普通卷积（更快）

    # 加权损失配置
    loss_type = 'weighted_mse'  # 可选: 'weighted_mse', 'multi_step', 'huber', 'mse'
    time_decay = 'linear'       # 可选: 'linear', 'exp', 'uniform'
    node_weight_method = 'std'  # 可选: 'std', 'mean', 'uniform'

    print("\n" + "="*50)
    print("STGCN 模型训练 (加权损失版本)")
    print("="*50)
    print(f"使用TCN: {use_tcn}")
    print(f"损失函数类型: {loss_type}")
    print(f"时间衰减方式: {time_decay}")
    print(f"节点权重方式: {node_weight_method}")

    # ========== 加载数据 ==========
    train_loader, val_loader, test_loader, adj_matrix, scaler, y_train = load_data(data_path, batch_size)
    adj = adj_matrix.to(device)

    # 计算节点权重
    node_weights = calculate_node_weights(y_train, method=node_weight_method)
    print(f"节点权重范围: [{node_weights.min():.3f}, {node_weights.max():.3f}]")

    # ========== 创建模型 ==========
    model = STGCN_TCN(
        num_nodes=num_nodes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        pred_len=pred_len,
        kernel_size=kernel_size,
        use_tcn=use_tcn
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # ========== 选择损失函数 ==========
    if loss_type == 'weighted_mse':
        criterion = WeightedMSELoss(pred_len=pred_len, node_weights=node_weights, time_decay=time_decay)
        print("[OK] 使用加权MSE损失")
    elif loss_type == 'multi_step':
        criterion = MultiStepWeightedLoss(pred_len=pred_len)
        print("[OK] 使用多步加权损失")
    elif loss_type == 'huber':
        criterion = HuberWeightedLoss(delta=1.0, pred_len=pred_len, time_decay=time_decay)
        print("[OK] 使用Huber加权损失")
    else:
        criterion = nn.MSELoss()
        print("[OK] 使用普通MSE损失")

    # ========== 训练 ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, adj, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, adj, criterion, device)

        # 学习率调度
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_stgcn_model-new.pth')
            if (epoch + 1) % 5 == 0:
                print(f"  → 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停! 验证损失{patience}个epoch未改善")
                break

    # ========== 测试评估 ==========
    print("\n" + "="*50)
    print("测试集评估")
    print("="*50)

    model.load_state_dict(torch.load('best_stgcn_model-new.pth', weights_only=True))

    # 预测并绘图
    predictions, true_values = predict_and_plot(model, test_loader, adj, scaler, device)

    mae, rmse, mape, r2 = calculate_metrics(predictions, true_values, scaler)

    print(f"\n测试集指标 (STGCN-TCN + 加权损失):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve (STGCN-TCN + Weighted Loss)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curve-new.png', dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存到: training_curve-new.png")
    plt.close()

    # 保存结果
    np.savez('stgcn_results-new.npz',
             predictions=predictions,
             true_values=true_values,
             train_losses=train_losses,
             val_losses=val_losses,
             mae=mae, rmse=rmse, mape=mape, r2=r2)

    print("\n训练完成!")
    print("保存文件:")
    print("  - best_stgcn_model-new.pth (模型权重)")
    print("  - stgcn_prediction-new.png (预测图)")
    print("  - training_curve-new.png (训练曲线)")
    print("  - stgcn_results-new.npz (所有结果)")

    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
