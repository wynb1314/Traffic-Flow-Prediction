"""
PEMS04 数据预处理 - 分节点标准化版本
特点：每个节点独立计算均值和标准差进行标准化
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp


def build_correlation_adj(flow_data, top_k=3, threshold=0.5):
    """
    基于流量相关性构建邻接矩阵
    flow_data: (T, N) 时间序列流量数据
    top_k: 每个节点保留相关性最高的k个邻居
    threshold: 相关性阈值（0-1之间）
    """
    N = flow_data.shape[1]
    print(f"正在计算 {N} 个节点的相关性矩阵...")

    # 计算皮尔逊相关系数矩阵
    corr_matrix = np.corrcoef(flow_data.T)  # (N, N)

    # 处理NaN值
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 将相关系数转为0-1之间（取绝对值）
    corr_matrix = np.abs(corr_matrix)

    # 构建邻接矩阵：Top-K策略
    adj_matrix = np.zeros((N, N))
    for i in range(N):
        corr_i = corr_matrix[i].copy()
        corr_i[i] = 0  # 排除自己

        valid_indices = np.where(corr_i >= threshold)[0]

        if len(valid_indices) > 0:
            if len(valid_indices) > top_k:
                top_indices = valid_indices[np.argsort(corr_i[valid_indices])[-top_k:]]
            else:
                top_indices = valid_indices
            adj_matrix[i, top_indices] = 1
        else:
            top_indices = np.argsort(corr_i)[-top_k:]
            adj_matrix[i, top_indices] = 1

    # 对称化（无向图）
    adj_matrix = (adj_matrix + adj_matrix.T > 0).astype(float)

    return adj_matrix


def load_pems_data(data_path):
    """加载PEMS数据"""
    data = np.load(data_path)
    print(f"数据集包含的键: {data.files}")

    # 流量数据 (T, N, F)，取第一个特征（流量）
    flow_data = data['data'][:, :, 0]  # (T, N)

    # 尝试加载邻接矩阵
    if 'adj' in data.files:
        adj_matrix = data['adj']
        print("[OK] 使用数据集提供的邻接矩阵")
    elif 'adj_mx' in data.files:
        adj_matrix = data['adj_mx']
        print("[OK] 使用数据集提供的邻接矩阵 (adj_mx)")
    else:
        print("[OK] 未找到预定义邻接矩阵，基于流量相关性构建...")
        adj_matrix = build_correlation_adj(flow_data, top_k=3, threshold=0.5)
        print("[OK] 邻接矩阵构建完成（基于流量相关性）")

    print(f"流量数据形状: {flow_data.shape}")
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print(f"邻接矩阵边数: {int(np.sum(adj_matrix))}")
    print(f"平均度数: {np.sum(adj_matrix) / adj_matrix.shape[0]:.2f}")

    return flow_data, adj_matrix


def normalize_adj(adj):
    """对称归一化邻接矩阵: D^(-1/2) * (A + I) * D^(-1/2)"""
    adj = adj + np.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized.toarray()


def create_dataset(data, seq_len, pred_len):
    """
    创建滑动窗口数据集
    data: (T, N)
    返回: X (num_samples, seq_len, N), y (num_samples, pred_len, N)
    """
    T, N = data.shape
    num_samples = T - seq_len - pred_len + 1

    X = np.zeros((num_samples, seq_len, N))
    y = np.zeros((num_samples, pred_len, N))

    for i in range(num_samples):
        X[i] = data[i:i+seq_len]
        y[i] = data[i+seq_len:i+seq_len+pred_len]

    return X, y


def node_level_normalize(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    分节点标准化：每个节点独立计算均值和标准差

    返回:
        - 标准化后的数据
        - node_mean: (N,) 每个节点的均值
        - node_std: (N,) 每个节点的标准差
    """
    num_nodes = X_train.shape[2]

    # 只使用训练集计算每个节点的统计量
    # X_train: (samples, seq_len, nodes)
    # 对每个节点，计算所有样本和时间步的均值和标准差

    node_mean = np.zeros(num_nodes)
    node_std = np.zeros(num_nodes)

    for i in range(num_nodes):
        # 获取节点i的所有训练数据
        node_data = X_train[:, :, i].flatten()  # (samples * seq_len,)
        node_mean[i] = np.mean(node_data)
        node_std[i] = np.std(node_data)

        # 避免除零
        if node_std[i] < 1e-8:
            node_std[i] = 1.0

    # 标准化函数
    def normalize(X, mean, std):
        # X: (samples, seq_len, nodes)
        # mean, std: (nodes,)
        return (X - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]

    def normalize_y(y, mean, std):
        # y: (samples, pred_len, nodes)
        return (y - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]

    X_train_norm = normalize(X_train, node_mean, node_std)
    y_train_norm = normalize_y(y_train, node_mean, node_std)
    X_val_norm = normalize(X_val, node_mean, node_std)
    y_val_norm = normalize_y(y_val, node_mean, node_std)
    X_test_norm = normalize(X_test, node_mean, node_std)
    y_test_norm = normalize_y(y_test, node_mean, node_std)

    return X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm, node_mean, node_std


class TrafficDataset(Dataset):
    """交通数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    # ========== 配置参数 ==========
    data_path = "data/PEMS04/PEMS04.npz"
    seq_len = 12      # 历史1小时（5分钟*12）
    pred_len = 12     # 预测1小时
    train_ratio = 0.7
    val_ratio = 0.1

    print("="*50)
    print("PEMS04数据预处理 - 分节点标准化版本")
    print("="*50)

    # ========== 1. 加载数据 ==========
    flow_data, adj_matrix = load_pems_data(data_path)

    # ========== 2. 创建滑动窗口 ==========
    X, y = create_dataset(flow_data, seq_len, pred_len)
    print(f"\n滑动窗口数据集: X={X.shape}, y={y.shape}")

    # ========== 3. 划分数据集（按时间顺序）==========
    num_samples = len(X)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # ========== 4. 分节点标准化 ==========
    print(f"\n正在进行分节点标准化...")
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm, node_mean, node_std = \
        node_level_normalize(X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"\n分节点标准化参数:")
    print(f"  节点均值范围: [{node_mean.min():.2f}, {node_mean.max():.2f}]")
    print(f"  节点标准差范围: [{node_std.min():.2f}, {node_std.max():.2f}]")
    print(f"  平均均值: {node_mean.mean():.4f}")
    print(f"  平均标准差: {node_std.mean():.4f}")

    # ========== 5. 归一化邻接矩阵 ==========
    adj_norm = normalize_adj(adj_matrix)
    print(f"\n邻接矩阵归一化完成")

    # ========== 6. 创建PyTorch数据集 ==========
    train_dataset = TrafficDataset(X_train_norm, y_train_norm)
    val_dataset = TrafficDataset(X_val_norm, y_val_norm)
    test_dataset = TrafficDataset(X_test_norm, y_test_norm)

    # ========== 7. 保存处理后的数据 ==========
    np.savez("processed_data-new.npz",
             X_train=X_train_norm, y_train=y_train_norm,
             X_val=X_val_norm, y_val=y_val_norm,
             X_test=X_test_norm, y_test=y_test_norm,
             adj_matrix=adj_norm,
             node_mean=node_mean,   # 分节点均值
             node_std=node_std,     # 分节点标准差
             # 同时保存全局统计量（用于兼容旧代码）
             global_mean=np.mean(node_mean),
             global_std=np.mean(node_std),
             seq_len=seq_len,
             pred_len=pred_len)

    print(f"\n数据已保存到 processed_data-new.npz")

    # ========== 8. 返回结果 ==========
    print("\n" + "="*50)
    print("数据预处理完成!")
    print("="*50)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'adj_matrix': adj_norm,
        'scaler': (node_mean, node_std)  # 返回分节点标准化参数
    }


if __name__ == "__main__":
    data = main()

    # ========== 数据验证 ==========
    print("\n数据验证:")
    print(f"训练集样本数: {len(data['train_dataset'])}")
    print(f"验证集样本数: {len(data['val_dataset'])}")
    print(f"测试集样本数: {len(data['test_dataset'])}")

    sample_X, sample_y = data['train_dataset'][0]
    print(f"\n样本形状:")
    print(f"  输入 X: {sample_X.shape}")
    print(f"  输出 y: {sample_y.shape}")
    print(f"  邻接矩阵: {data['adj_matrix'].shape}")

    node_mean, node_std = data['scaler']
    print(f"\n分节点标准化参数形状:")
    print(f"  节点均值: {node_mean.shape}")
    print(f"  节点标准差: {node_std.shape}")

    print("\n使用说明:")
    print("- 分节点标准化：每个节点独立计算均值和标准差")
    print("- 反标准化公式: y_real = y_norm * node_std[node_id] + node_mean[node_id]")
    print("- 适用于流量差异较大的节点场景")
