"""
STGCN 模型 - TCN升级版本（高效实现）
特点：使用共享权重的TCN，避免对每个节点独立计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalConvLayer(nn.Module):
    """
    时间卷积层（高效版本）
    使用1D卷积在时间维度上操作，所有节点共享权重
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)
        )

    def forward(self, x):
        # x: [batch, channels, nodes, time]
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    TCN块 - 高效版本
    使用扩张卷积 + 残差连接
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super(TCNBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size - 1) * dilation),
            dilation=(1, dilation)
        )
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size - 1) * dilation),
            dilation=(1, dilation)
        )

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, channels, nodes, time]
        residual = x

        # 第一个扩张卷积
        out = self.conv1(x)
        # 移除右侧填充（因果卷积）
        if self.conv1.padding[1] > 0:
            out = out[:, :, :, :-self.conv1.padding[1]]
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # 第二个扩张卷积
        out = self.conv2(out)
        if self.conv2.padding[1] > 0:
            out = out[:, :, :, :-self.conv2.padding[1]]
        out = self.bn2(out)
        out = self.dropout(out)

        # 残差连接
        out = F.relu(out + residual)
        return out


class TCNModule(nn.Module):
    """
    TCN模块 - 堆叠多个扩张卷积块
    扩张率: 1, 2, 4, 8... 扩大感受野
    """
    def __init__(self, channels, num_layers=4, kernel_size=3, dropout=0.1):
        super(TCNModule, self).__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(channels, kernel_size, dilation, dropout))

        self.tcn = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.tcn:
            x = layer(x)
        return x


class SpatialGraphConvLayer(nn.Module):
    """
    空间图卷积层
    """
    def __init__(self, in_channels, out_channels, num_nodes):
        super(SpatialGraphConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, adj):
        # x: [batch, in_channels, num_nodes, time]
        # adj: [num_nodes, num_nodes]
        batch_size, in_channels, num_nodes, time_steps = x.shape

        # 重塑: [batch, time, nodes, in_channels]
        x = x.permute(0, 3, 2, 1)

        # 图卷积: A * X * Theta
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.theta)

        # 恢复形状: [batch, out_channels, nodes, time]
        x = x.permute(0, 3, 2, 1)
        return x


class STConvBlock(nn.Module):
    """
    时空卷积块 - TCN升级版
    结构: TCN时间卷积 -> 图卷积 -> TCN时间卷积
    """
    def __init__(self, in_channels, out_channels, num_nodes, kernel_size=3, use_tcn=True):
        super(STConvBlock, self).__init__()

        self.use_tcn = use_tcn

        # 第一个时间卷积
        if use_tcn:
            self.temporal_conv1 = TemporalConvLayer(in_channels, out_channels, kernel_size)
            self.tcn1 = TCNModule(out_channels, num_layers=2, kernel_size=kernel_size)
        else:
            self.temporal_conv1 = TemporalConvLayer(in_channels, out_channels, kernel_size)

        # 空间图卷积
        self.graph_conv = SpatialGraphConvLayer(out_channels, out_channels, num_nodes)

        # 第二个时间卷积
        if use_tcn:
            self.temporal_conv2 = TemporalConvLayer(out_channels, out_channels, kernel_size)
            self.tcn2 = TCNModule(out_channels, num_layers=2, kernel_size=kernel_size)
        else:
            self.temporal_conv2 = TemporalConvLayer(out_channels, out_channels, kernel_size)

        # 批归一化
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                             if in_channels != out_channels else None

    def forward(self, x, adj):
        residual = x

        # 时间卷积1 + TCN
        x = self.temporal_conv1(x)
        if self.use_tcn:
            x = self.tcn1(x)
        x = F.relu(x)

        # 图卷积
        x = self.graph_conv(x, adj)
        x = F.relu(x)

        # 时间卷积2 + TCN
        x = self.temporal_conv2(x)
        if self.use_tcn:
            x = self.tcn2(x)
        x = self.batch_norm(x)

        # 残差连接
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        x = F.relu(x + residual)
        return x


class STGCN_TCN(nn.Module):
    """
    STGCN-TCN: 时空图卷积网络 + TCN时间模块
    """
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64,
                 num_layers=3, pred_len=12, kernel_size=3, use_tcn=True):
        super(STGCN_TCN, self).__init__()

        self.num_nodes = num_nodes
        self.pred_len = pred_len

        # 第一层：输入适配
        self.start_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # ST-Conv块
        self.st_blocks = nn.ModuleList([
            STConvBlock(hidden_channels, hidden_channels, num_nodes, kernel_size, use_tcn)
            for _ in range(num_layers)
        ])

        # 输出层
        self.end_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.end_conv2 = nn.Conv2d(hidden_channels // 2, pred_len, kernel_size=1)

    def forward(self, x, adj):
        # x: [batch_size, seq_len, num_nodes]
        batch_size, seq_len, num_nodes = x.shape

        # 重塑为 [batch_size, in_channels, num_nodes, seq_len]
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)

        # 起始卷积
        x = self.start_conv(x)

        # ST-Conv块
        for block in self.st_blocks:
            x = block(x, adj)

        # 输出层
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)

        # 时间维度求平均
        x = x.mean(dim=-1)

        return x


# 兼容旧接口
class STGCN(STGCN_TCN):
    """兼容旧接口的别名"""
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64,
                 num_layers=3, pred_len=12, kernel_size=3):
        super(STGCN, self).__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            pred_len=pred_len,
            kernel_size=kernel_size,
            use_tcn=True
        )


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    batch_size = 64
    seq_len = 12
    num_nodes = 307
    pred_len = 12

    # 创建模型
    model = STGCN_TCN(
        num_nodes=num_nodes,
        hidden_channels=64,
        num_layers=3,
        pred_len=pred_len,
        kernel_size=3,
        use_tcn=True
    ).to(device)

    # 测试输入
    x = torch.randn(batch_size, seq_len, num_nodes).to(device)
    adj = torch.rand(num_nodes, num_nodes).to(device)
    adj = (adj + adj.T) / 2
    adj = (adj > 0.5).float()

    # 测试速度
    import time
    model.eval()
    with torch.no_grad():
        # 预热
        _ = model(x, adj)

        # 计时
        start = time.time()
        for _ in range(10):
            _ = model(x, adj)
        end = time.time()

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {model(x, adj).shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"平均推理时间: {(end-start)/10*1000:.2f}ms")
