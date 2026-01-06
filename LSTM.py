import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


class SimpleLSTM(nn.Module):
    """
    简单的LSTM交通流量预测模型
    """
    def __init__(self, num_nodes, hidden_dim=64, num_layers=2, dropout=0.1):
        """
        参数:
            num_nodes: 节点数量 (307)
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(SimpleLSTM, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=num_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_nodes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        x: [batch_size, seq_len, num_nodes]
        return: [batch_size, pred_len, num_nodes]
        """
        batch_size, seq_len, num_nodes = x.shape
        
        # LSTM: [batch_size, seq_len, num_nodes] -> [batch_size, seq_len, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用所有时间步的输出进行预测（取最后pred_len步）
        # 这里简化为使用最后的hidden state来预测所有未来步
        outputs = []
        hidden = (h_n, c_n)
        
        # 使用最后一个时间步作为初始输入
        decoder_input = x[:, -1:, :]  # [batch_size, 1, num_nodes]
        
        # 预测12步
        for t in range(12):
            lstm_out, hidden = self.lstm(decoder_input, hidden)
            lstm_out = self.dropout(lstm_out)
            pred = self.fc(lstm_out)  # [batch_size, 1, num_nodes]
            outputs.append(pred)
            decoder_input = pred  # 使用预测值作为下一步输入
        
        # 拼接所有预测
        output = torch.cat(outputs, dim=1)  # [batch_size, 12, num_nodes]
        
        return output


def calculate_metrics(pred, true, scaler):
    """
    计算评估指标（在原始尺度上）
    pred, true: [num_samples, pred_len, num_nodes] 标准化后的数据
    scaler: (mean, std)
    """
    mean, std = scaler
    
    # 反标准化
    pred_real = pred * std + mean
    true_real = true * std + mean
    
    # 展平为一维
    pred_flat = pred_real.flatten()
    true_flat = true_real.flatten()
    
    # MAE
    mae = np.mean(np.abs(pred_flat - true_flat))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
    
    # MAPE (避免除零，设置阈值)
    # 只计算真实值大于阈值的位置（避免除以很小的数）
    threshold = 10.0  # 流量小于10的不计入MAPE
    mask = true_flat > threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100
    else:
        mape = 0.0
    
    # R² (决定系数)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mae, rmse, mape, r2


def load_data(data_path, batch_size=64):
    """
    加载预处理后的数据
    """
    data = np.load(data_path)
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.FloatTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'])
    
    # 创建数据集
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 标准化参数
    scaler = (data['mean'].item(), data['std'].item())
    
    print(f"数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader, scaler


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_X)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def predict_and_plot(model, test_loader, scaler, device, save_path='lstm_prediction.png'):
    """
    预测并绘制结果（类似你提供的图）
    """
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            
            predictions.append(output.cpu().numpy())
            true_values.append(batch_y.numpy())
    
    # 拼接所有批次
    predictions = np.concatenate(predictions, axis=0)  # [num_samples, 12, 307]
    true_values = np.concatenate(true_values, axis=0)
    
    # 反标准化
    mean, std = scaler
    predictions_real = predictions * std + mean
    true_values_real = true_values * std + mean
    
    # 选择一个节点进行可视化（节点0）
    node_idx = 0
    
    # 展平时间维度：将所有样本的预测连接起来
    # 只取每个样本的第一个预测步，避免重叠
    pred_flat = predictions_real[:, 0, node_idx]  # 取第1步预测
    true_flat = true_values_real[:, 0, node_idx]
    
    # 计算移动平均（平滑）
    window_size = 12
    moving_avg = np.convolve(pred_flat, np.ones(window_size)/window_size, mode='valid')
    
    # 绘图
    plt.figure(figsize=(14, 6))
    
    # 只显示前1200个时间步
    plot_len = min(1200, len(true_flat))
    
    plt.plot(true_flat[:plot_len], label='True Value', color='blue', linewidth=1.5, alpha=0.7)
    plt.plot(pred_flat[:plot_len], label='Prediction', color='green', linewidth=1, 
             linestyle='--', alpha=0.7)
    
    # 移动平均
    if len(moving_avg) > 0:
        ma_len = min(plot_len, len(moving_avg))
        plt.plot(range(window_size-1, window_size-1+ma_len), moving_avg[:ma_len], 
                label='Moving Average', color='orange', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Hour Timesteps', fontsize=12)
    plt.ylabel('Output Value', fontsize=12)
    plt.title('Prediction vs. True Value', fontsize=14)
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
    
    data_path = "processed_data.npz"
    batch_size = 64
    num_nodes = 307
    hidden_dim = 64
    num_layers = 2
    dropout = 0.1
    num_epochs = 100
    learning_rate = 0.001
    patience = 10
    
    print("\n" + "="*50)
    print("简单LSTM模型训练")
    print("="*50)
    
    # ========== 加载数据 ==========
    train_loader, val_loader, test_loader, scaler = load_data(data_path, batch_size)
    
    # ========== 创建模型 ==========
    model = SimpleLSTM(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    # ========== 训练 ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model.pth')
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
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_lstm_model.pth', weights_only=True))
    
    # 获取所有预测结果
    model.eval()
    all_predictions = []
    all_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            all_predictions.append(output.cpu().numpy())
            all_true.append(batch_y.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    true_values = np.concatenate(all_true, axis=0)
    
    # 计算指标
    mae, rmse, mape, r2 = calculate_metrics(predictions, true_values, scaler)
    
    print(f"\n测试集指标 (LSTM):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    # 绘制预测图
    predict_and_plot(model, test_loader, scaler, device)
    
    # 保存结果
    np.savez('lstm_results.npz',
             predictions=predictions,
             true_values=true_values,
             train_losses=train_losses,
             val_losses=val_losses,
             mae=mae, rmse=rmse, mape=mape, r2=r2)
    
    print("\n训练完成!")
    print("保存文件:")
    print("  - best_lstm_model.pth (模型权重)")
    print("  - lstm_prediction.png (预测图)")
    print("  - lstm_results.npz (所有结果)")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()