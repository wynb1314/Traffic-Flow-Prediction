# -*- coding: utf-8 -*-
"""
PEMS04 STGCN流量预测仪表盘 - 简化版
streamlit run traffic_dashboard.py
"""

# 1. 页面配置
import streamlit as st
st.set_page_config(
    page_title="STGCN流量预测",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 3. STGCN模型定义
class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
    def forward(self, x): return self.conv(x)

class SpatialGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(SpatialGraphConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.num_nodes = num_nodes
        self.reset_parameters()
    def reset_parameters(self): nn.init.xavier_uniform_(self.theta)
    def forward(self, x, adj):
        batch_size, in_channels, num_nodes, time_steps = x.shape
        x = x.permute(0, 3, 2, 1)
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.theta)
        x = x.permute(0, 3, 2, 1)
        return x

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, kernel_size=3):
        super(STConvBlock, self).__init__()
        self.temporal_conv1 = TemporalConvLayer(in_channels, out_channels, kernel_size)
        self.graph_conv = SpatialGraphConvLayer(out_channels, out_channels, num_nodes)
        self.temporal_conv2 = TemporalConvLayer(out_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
    def forward(self, x, adj):
        residual = x
        x = self.temporal_conv1(x); x = F.relu(x)
        x = self.graph_conv(x, adj); x = F.relu(x)
        x = self.temporal_conv2(x); x = self.batch_norm(x)
        if self.residual_conv is not None: residual = self.residual_conv(residual)
        x = F.relu(x + residual)
        return x

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, num_layers=3, pred_len=12, kernel_size=3):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes; self.pred_len = pred_len
        self.start_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.st_blocks = nn.ModuleList([STConvBlock(hidden_channels, hidden_channels, num_nodes, kernel_size) for _ in range(num_layers)])
        self.end_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.end_conv2 = nn.Conv2d(hidden_channels // 2, pred_len, kernel_size=1)
    def forward(self, x, adj):
        batch_size, seq_len, num_nodes = x.shape
        x = x.unsqueeze(1); x = x.permute(0, 1, 3, 2)
        x = self.start_conv(x)
        for block in self.st_blocks: x = block(x, adj)
        x = F.relu(self.end_conv1(x)); x = self.end_conv2(x)
        x = x.mean(dim=-1)
        return x

# 4. 加载函数
@st.cache_resource
def load_model_and_data():
    """加载模型和测试数据"""
    try:
        # 路径配置
        data_path = "processed_data.npz"
        model_path = "best_stgcn_model.pth"
        
        if not os.path.exists(data_path):
            st.error(f"数据文件不存在: {data_path}")
            return None, None, None, None
        
        if not os.path.exists(model_path):
            st.error(f"模型文件不存在: {model_path}")
            return None, None, None, None
        
        # 加载数据
        data = np.load(data_path)
        
        # 提取测试集
        X_test = data['X_test']  # [batch, seq_len=12, nodes=307]
        y_test = data['y_test']  # [batch, pred_len=12, nodes=307]
        adj_matrix = data['adj_matrix']
        mean = data['mean'].item()
        std = data['std'].item()
        
        st.success(f"✅ 数据加载成功: X_test形状={X_test.shape}, y_test形状={y_test.shape}")
        st.info(f"📊 标准化参数: mean={mean:.4f}, std={std:.4f}")
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = STGCN(num_nodes=307, in_channels=1, hidden_channels=64, num_layers=3, pred_len=12, kernel_size=3)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        st.success(f"✅ 模型加载成功 (设备: {device})")
        
        return X_test, y_test, adj_matrix, (mean, std), model, device
        
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        return None, None, None, None, None, None

# 5. 主界面
st.title("🚦 STGCN交通流量预测系统")

# 初始化按钮
if st.button("🚀 初始化系统", type="primary"):
    with st.spinner("正在加载模型和数据..."):
        X_test, y_test, adj_matrix, scaler, model, device = load_model_and_data()
        
        if X_test is not None:
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test  # 真实值
            st.session_state['adj_matrix'] = adj_matrix
            st.session_state['scaler'] = scaler
            st.session_state['model'] = model
            st.session_state['device'] = device
            
            # 立即进行预测
            with st.spinner("正在进行预测..."):
                try:
                    adj_tensor = torch.FloatTensor(adj_matrix).to(device)
                    batch_size = 32
                    all_predictions = []
                    
                    # 分批预测
                    for i in range(0, len(X_test), batch_size):
                        batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
                        with torch.no_grad():
                            batch_pred = model(batch_X, adj_tensor)
                            all_predictions.append(batch_pred.cpu().numpy())
                    
                    predictions = np.concatenate(all_predictions, axis=0)  # [batch, 12, 307]
                    
                    # 保存预测结果
                    st.session_state['predictions'] = predictions
                    st.success(f"✅ 预测完成！预测结果形状: {predictions.shape}")
                    
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")

# 检查是否已初始化
if 'predictions' not in st.session_state:
    st.info("👆 请点击'初始化系统'按钮开始")
    st.stop()

# 获取数据
X_test = st.session_state['X_test']
y_test = st.session_state['y_test']  # 真实值（标准化后）
predictions = st.session_state['predictions']  # 预测值（标准化后）
scaler = st.session_state['scaler']
mean, std = scaler

# 反标准化
y_test_real = y_test * std + mean
predictions_real = predictions * std + mean

st.success(f"✅ 系统已就绪 | 测试样本数: {len(X_test)} | 预测步长: {predictions.shape[1]} | 节点数: {predictions.shape[2]}")

# 6. 侧边栏配置
st.sidebar.header("⚙️ 可视化配置")

# 选择样本
sample_idx = st.sidebar.slider(
    "选择测试样本",
    min_value=0,
    max_value=len(X_test)-1,
    value=0,
    help="选择测试集中的第几个样本"
)

# 选择节点
num_nodes = predictions.shape[2]
selected_node = st.sidebar.selectbox(
    "选择监测节点",
    options=list(range(num_nodes)),
    format_func=lambda x: f"节点 {x}",
    key="node_selector"
)

# 选择预测步长
pred_step = st.sidebar.slider(
    "选择预测步长",
    min_value=0,
    max_value=predictions.shape[1]-1,
    value=0,
    help="选择预测的第几个时间步"
)

# 7. 时序对比可视化
st.header("📈 流量 - 真实值 vs 预测值")

# 提取单个样本的数据
# 对于第sample_idx个样本，我们有12个时间步的预测
sample_pred = predictions_real[sample_idx]  # [12, 307]
sample_true = y_test_real[sample_idx]       # [12, 307]

# 提取选定节点的所有预测步长
pred_series = sample_pred[:, selected_node]
true_series = sample_true[:, selected_node]

# 创建DataFrame
df_plot = pd.DataFrame({
    "预测步长": np.arange(len(pred_series)),
    "真实值": true_series,
    "预测值": pred_series
})

# Plotly绘图
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df_plot["预测步长"], y=df_plot["真实值"],
    mode='lines+markers', name='真实值', 
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=8),
    hovertemplate="步长: %{x}<br>真实值: %{y:.2f}<extra></extra>"
))
fig_ts.add_trace(go.Scatter(
    x=df_plot["预测步长"], y=df_plot["预测值"],
    mode='lines+markers', name='预测值', 
    line=dict(color='#ff7f0e', width=3, dash='dash'),
    marker=dict(size=8, symbol='diamond'),
    hovertemplate="步长: %{x}<br>预测值: %{y:.2f}<extra></extra>"
))

fig_ts.update_layout(
    xaxis_title="预测步长",
    yaxis_title="流量",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
    height=500,
    template="plotly_white",
    title=f"样本 {sample_idx} - 节点 {selected_node} 的流量预测"
)
st.plotly_chart(fig_ts, use_container_width=True)

# 8. 单点对比
st.header("🎯 单点详细对比")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        f"真实值 (步长 {pred_step})",
        f"{true_series[pred_step]:.2f}"
    )
with col2:
    st.metric(
        f"预测值 (步长 {pred_step})",
        f"{pred_series[pred_step]:.2f}"
    )
with col3:
    error = abs(true_series[pred_step] - pred_series[pred_step])
    error_percent = (error / (true_series[pred_step] + 1e-8)) * 100
    st.metric(
        "绝对误差",
        f"{error:.2f}",
        delta=f"{error_percent:.1f}%"
    )

# 9. 统计信息
st.header("📊 统计信息")

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"真实值统计 (节点 {selected_node})")
    st.write(f"最小值: {np.min(true_series):.2f}")
    st.write(f"最大值: {np.max(true_series):.2f}")
    st.write(f"平均值: {np.mean(true_series):.2f}")
    st.write(f"标准差: {np.std(true_series):.2f}")

with col2:
    st.subheader(f"预测值统计 (节点 {selected_node})")
    st.write(f"最小值: {np.min(pred_series):.2f}")
    st.write(f"最大值: {np.max(pred_series):.2f}")
    st.write(f"平均值: {np.mean(pred_series):.2f}")
    st.write(f"标准差: {np.std(pred_series):.2f}")

# 10. 评估指标
st.header("📈 模型评估指标")

# 计算这个样本所有节点的指标
sample_pred_all = sample_pred.flatten()
sample_true_all = sample_true.flatten()

mae = mean_absolute_error(sample_true_all, sample_pred_all)
rmse = np.sqrt(mean_squared_error(sample_true_all, sample_pred_all))

# 避免除零
mask = sample_true_all != 0
if mask.sum() > 0:
    mape = np.mean(np.abs((sample_true_all[mask] - sample_pred_all[mask]) / sample_true_all[mask])) * 100
else:
    mape = 0.0

# R²
ss_res = np.sum((sample_true_all - sample_pred_all) ** 2)
ss_tot = np.sum((sample_true_all - np.mean(sample_true_all)) ** 2)
r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAPE", f"{mape:.2f}%")
col4.metric("R²", f"{r2:.4f}")

# 11. 全节点分布
st.header("🌍 全节点流量分布")

# 选择时间步
time_step = st.slider(
    "选择时间步",
    min_value=0,
    max_value=sample_true.shape[0]-1,
    value=0,
    key="time_slider"
)

# 获取该时间步所有节点的数据
all_nodes_true = sample_true[time_step, :]
all_nodes_pred = sample_pred[time_step, :]

# 创建对比图
fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 真实值分布
ax1.hist(all_nodes_true, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
ax1.set_title(f"真实值分布 (步长 {time_step})", fontsize=14)
ax1.set_xlabel("流量", fontsize=12)
ax1.set_ylabel("节点数", fontsize=12)
ax1.grid(True, alpha=0.3)

# 预测值分布
ax2.hist(all_nodes_pred, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black')
ax2.set_title(f"预测值分布 (步长 {time_step})", fontsize=14)
ax2.set_xlabel("流量", fontsize=12)
ax2.set_ylabel("节点数", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_dist)

# 12. 网络拓扑
st.header("🌐 交通网络拓扑")

# 使用邻接矩阵
adj_matrix = st.session_state['adj_matrix']
G = nx.from_numpy_array(adj_matrix)

# 设置节点颜色基于真实值
node_values = all_nodes_true

# 创建网络图
pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(num_nodes))

# 绘制边
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none', mode='lines'
)

# 绘制节点
node_x, node_y, node_color, node_text = [], [], [], []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(node_values[node])
    node_text.append(f'节点 {node}<br>流量: {node_values[node]:.2f}')

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers', hoverinfo='text', text=node_text,
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        color=node_color,
        size=8,
        colorbar=dict(thickness=15, title="流量", xanchor='left', titleside='right'),
        line_width=2
    )
)

fig_network = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f'交通网络 - 真实流量分布 (步长 {time_step})',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
)
st.plotly_chart(fig_network, use_container_width=True)

# 13. 数据表格
st.header("📋 详细数据")

# 显示当前样本的数据
df_sample = pd.DataFrame({
    "节点": list(range(num_nodes)),
    f"真实值(步长{time_step})": all_nodes_true,
    f"预测值(步长{time_step})": all_nodes_pred,
    "误差": abs(all_nodes_true - all_nodes_pred)
})

st.dataframe(df_sample.head(20), use_container_width=True)

# 14. 使用说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 系统说明：
    1. **数据来源**：直接使用你训练好的`processed_data.npz`中的测试集
    2. **模型预测**：使用训练好的`best_stgcn_model.pth`进行预测
    3. **显示结果**：所有数据都已反标准化到原始尺度
    
    ### 使用步骤：
    1. 点击"初始化系统"按钮加载数据和模型
    2. 在侧边栏选择测试样本、节点和预测步长
    3. 查看时序对比图、统计指标和网络拓扑
    
    ### 数据解释：
    - **测试样本**：测试集中的不同序列
    - **预测步长**：每个样本预测未来12个时间步
    - **节点**：307个不同的监测点
    """)