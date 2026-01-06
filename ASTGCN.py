import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


class SpatialAttention(nn.Module):
    """
    空间注意力机制
    学习节点之间的动态权重
    """
    def __init__(self, num_nodes, num_features):
        super(SpatialAttention, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_features, num_features))
        self.W2 = nn.Parameter(torch.FloatTensor(num_features, num_features))
        self.W3 = nn.Parameter(torch.FloatTensor(num_features))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.W3.unsqueeze(0))
        nn.init.xavier_uniform_(self.bs)
        nn.init.xavier_uniform_(self.Vs)
    
    def forward(self, x):
        """
        x: [batch_size, num_nodes, num_features]
        return: [batch_size, num_nodes, num_nodes]
        """
        # x: [B, N, F]
        lhs = torch.matmul(x, self.W1)  # [B, N, F]
        rhs = torch.matmul(x, self.W2)  # [B, N, F]
        
        # [B, N, F] @ [B, F, N] -> [B, N, N]
        product = torch.matmul(lhs, rhs.transpose(-2, -1))
        
        # 添加偏置
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        
        # Softmax归一化
        S = F.softmax(S, dim=-1)
        
        return S


class TemporalAttention(nn.Module):
    """
    时间注意力机制
    学习不同时间步的重要性
    """
    def __init__(self, num_nodes, num_features, num_timesteps):
        super(TemporalAttention, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(num_features, num_nodes))
        self.U3 = nn.Parameter(torch.FloatTensor(num_features))
        self.be = nn.Parameter(torch.FloatTensor(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_timesteps, num_timesteps))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U1.unsqueeze(0))
        nn.init.xavier_uniform_(self.U2)
        nn.init.xavier_uniform_(self.U3.unsqueeze(0))
        nn.init.xavier_uniform_(self.be)
        nn.init.xavier_uniform_(self.Ve)
    
    def forward(self, x):
        """
        x: [batch_size, num_timesteps, num_nodes, num_features]
        return: [batch_size, num_timesteps, num_timesteps]
        """
        # x: [B, T, N, F]
        _, num_timesteps, num_nodes, num_features = x.shape
        
        # 简化版时间注意力
        # [B, T, N, F] -> [B, T, N]
        lhs = torch.matmul(x, self.U3)  # [B, T, N]
        
        # [B, T, N] @ [N] -> [B, T]
        lhs = torch.matmul(lhs, self.U1)  # [B, T]
        
        # [B, T, 1] @ [B, 1, T] -> [B, T, T]
        E = torch.matmul(lhs.unsqueeze(-1), lhs.unsqueeze(1))
        
        E = torch.matmul(self.Ve, torch.sigmoid(E + self.be))
        
        # Softmax归一化
        E = F.softmax(E, dim=-1)
        
        return E


class ASTGCNBlock(nn.Module):
    """
    注意力时空图卷积块
    结构: 时间注意力 + 空间注意力 + 时空卷积
    """
    def __init__(self, in_channels, out_channels, num_nodes, num_timesteps, kernel_size=3):
        super(ASTGCNBlock, self).__init__()
        
        self.spatial_attention = SpatialAttention(num_nodes, in_channels)
        self.temporal_attention = TemporalAttention(num_nodes, in_channels, num_timesteps)
        
        # 时间卷积
        self.temporal_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)
        )
        
        # 空间卷积
        self.spatial_conv = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=(1, 1)
        )
        
        # 批归一化
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                             if in_channels != out_channels else None
    
    def forward(self, x, adj):
        """
        x: [batch_size, in_channels, num_nodes, num_timesteps]
        adj: [num_nodes, num_nodes]
        """
        batch_size, in_channels, num_nodes, num_timesteps = x.shape
        
        # 时间注意力
        # 需要重塑: [B, C, N, T] -> [B, T, N, C]
        x_t = x.permute(0, 3, 2, 1)
        temporal_att = self.temporal_attention(x_t)  # [B, T, T]
        
        # 应用时间注意力: [B, T, T] @ [B, T, N*C] -> [B, T, N*C]
        x_t_flat = x_t.reshape(batch_size, num_timesteps, -1)
        x_t_att = torch.matmul(temporal_att, x_t_flat)
        x_t_att = x_t_att.reshape(batch_size, num_timesteps, num_nodes, in_channels)
        x = x_t_att.permute(0, 3, 2, 1)  # [B, C, N, T]
        
        # 空间注意力
        # 需要重塑: [B, C, N, T] -> [B*T, N, C]
        x_s = x.permute(0, 3, 2, 1).reshape(-1, num_nodes, in_channels)
        spatial_att = self.spatial_attention(x_s)  # [B*T, N, N]
        
        # 应用空间注意力: [B*T, N, N] @ [B*T, N, C] -> [B*T, N, C]
        x_s_att = torch.matmul(spatial_att, x_s)
        x_s_att = x_s_att.reshape(batch_size, num_timesteps, num_nodes, in_channels)
        x = x_s_att.permute(0, 3, 2, 1)  # [B, C, N, T]
        
        residual = x
        
        # 时间卷积
        x = self.temporal_conv(x)
        x = F.relu(x)
        
        # 图卷积（使用邻接矩阵）
        # [B, C, N, T] -> [B, T, N, C]
        x = x.permute(0, 3, 2, 1)
        x = torch.matmul(adj, x)  # 空间聚合
        x = x.permute(0, 3, 2, 1)  # [B, C, N, T]
        
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        
        # 残差连接
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        x = F.relu(x + residual)
        
        return x


class ASTGCN(nn.Module):
    """
    注意力时空图卷积网络
    Attention-based Spatio-Temporal Graph Convolutional Network
    """
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, 
                 num_layers=2, num_timesteps=12, pred_len=12, kernel_size=3):
        """
        参数:
            num_nodes: 节点数量 (307)
            in_channels: 输入特征维度 (1)
            hidden_channels: 隐藏层维度
            num_layers: ASTGCN块的数量
            num_timesteps: 输入时间步长
            pred_len: 预测步长
            kernel_size: 卷积核大小
        """
        super(ASTGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # 起始卷积
        self.start_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # ASTGCN块
        self.astgcn_blocks = nn.ModuleList([
            ASTGCNBlock(hidden_channels, hidden_channels, num_nodes, num_timesteps, kernel_size)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.end_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.end_conv2 = nn.Conv2d(hidden_channels // 2, pred_len, kernel_size=1)
    
    def forward(self, x, adj):
        """
        x: [batch_size, seq_len, num_nodes]
        adj: [num_nodes, num_nodes]
        return: [batch_size, pred_len, num_nodes]
        """
        batch_size, seq_len, num_nodes = x.shape
        
        # 重塑为 [batch_size, in_channels, num_nodes, seq_len]
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, N, T]
        
        # 起始卷积
        x = self.start_conv(x)  # [B, hidden_channels, N, T]
        
        # ASTGCN块
        for block in self.astgcn_blocks:
            x = block(x, adj)
        
        # 输出层
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)  # [B, pred_len, N, T]
        
        # 时间维度平均
        x = x.mean(dim=-1)  # [B, pred_len, N]
        
        return x


def calculate_metrics(pred, true, scaler):
    """计算评估指标"""
    mean, std = scaler
    
    pred_real = pred * std + mean
    true_real = true * std + mean
    
    pred_flat = pred_real.flatten()
    true_flat = true_real.flatten()
    
    mae = np.mean(np.abs(pred_flat - true_flat))
    rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
    
    threshold = 10.0
    mask = true_flat > threshold
    if mask.sum() > 0:
        mape = np.mean(np.abs((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100
    else:
        mape = 0.0
    
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mae, rmse, mape, r2


def load_data(data_path, batch_size=64):
    """加载数据"""
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
    
    scaler = (data['mean'].item(), data['std'].item())
    
    print(f"数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader, adj_matrix, scaler


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


def predict_and_plot(model, test_loader, adj, scaler, device, save_path='astgcn_prediction.png'):
    """预测并绘图"""
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
    
    mean, std = scaler
    predictions_real = predictions * std + mean
    true_values_real = true_values * std + mean
    
    node_idx = 0
    pred_flat = predictions_real[:, 0, node_idx]
    true_flat = true_values_real[:, 0, node_idx]
    
    window_size = 12
    moving_avg = np.convolve(pred_flat, np.ones(window_size)/window_size, mode='valid')
    
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
    plt.title('ASTGCN: Prediction vs. True Value', fontsize=14)
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
    batch_size = 32  # ASTGCN参数较多，减小batch_size
    num_nodes = 307
    hidden_channels = 64
    num_layers = 2  # 注意力机制较复杂，减少层数
    num_timesteps = 12
    pred_len = 12
    kernel_size = 3
    num_epochs = 100 #轮数
    learning_rate = 0.001
    patience = 15
    
    print("\n" + "="*50)
    print("ASTGCN模型训练")
    print("="*50)
    
    # ========== 加载数据 ==========
    train_loader, val_loader, test_loader, adj_matrix, scaler = load_data(data_path, batch_size)
    adj = adj_matrix.to(device)
    
    # ========== 创建模型 ==========
    model = ASTGCN(
        num_nodes=num_nodes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        pred_len=pred_len,
        kernel_size=kernel_size
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    # ========== 训练 ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, adj, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, adj, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_astgcn_model.pth')
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
    
    model.load_state_dict(torch.load('best_astgcn_model.pth', weights_only=True))
    
    model.eval()
    all_predictions = []
    all_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X, adj)
            all_predictions.append(output.cpu().numpy())
            all_true.append(batch_y.numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    true_values = np.concatenate(all_true, axis=0)
    
    mae, rmse, mape, r2 = calculate_metrics(predictions, true_values, scaler)
    
    print(f"\n测试集指标 (ASTGCN):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    # 绘图
    predict_and_plot(model, test_loader, adj, scaler, device)
    
    # 保存结果
    np.savez('astgcn_results.npz',
             predictions=predictions,
             true_values=true_values,
             train_losses=train_losses,
             val_losses=val_losses,
             mae=mae, rmse=rmse, mape=mape, r2=r2)
    
    print("\n训练完成!")
    print("保存文件:")
    print("  - best_astgcn_model.pth (模型权重)")
    print("  - astgcn_prediction.png (预测图)")
    print("  - astgcn_results.npz (所有结果)")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()