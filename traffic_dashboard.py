"""
PEMS04 Traffic Dashboard - 优化版
streamlit run traffic_dashboard.py

原版的为 https://github.com/dekanms/pems04-traffic-dashboard/blob/main/traffic_dashboard.py
本版本优化了numpy，适配新版Streamlit
"""
import streamlit as st
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

# 忽略无关警告
warnings.filterwarnings('ignore')

# --------------------------
# 1. STGCN模型定义
# --------------------------
class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2)
        )
    
    def forward(self, x):
        return self.conv(x)

class SpatialGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(SpatialGraphConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.num_nodes = num_nodes
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
    
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
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                             if in_channels != out_channels else None
    
    def forward(self, x, adj):
        residual = x
        x = self.temporal_conv1(x)
        x = F.relu(x)
        x = self.graph_conv(x, adj)
        x = F.relu(x)
        x = self.temporal_conv2(x)
        x = self.batch_norm(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        x = F.relu(x + residual)
        return x

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, 
                 num_layers=3, pred_len=12, kernel_size=3):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.start_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.st_blocks = nn.ModuleList([
            STConvBlock(hidden_channels, hidden_channels, num_nodes, kernel_size)
            for _ in range(num_layers)
        ])
        self.end_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.end_conv2 = nn.Conv2d(hidden_channels // 2, pred_len, kernel_size=1)
    
    def forward(self, x, adj):
        batch_size, seq_len, num_nodes = x.shape
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.start_conv(x)
        for block in self.st_blocks:
            x = block(x, adj)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)
        x = x.mean(dim=-1)
        return x

# --------------------------
# 2. 页面配置 & 基础设置
# --------------------------
st.set_page_config(
    page_title="交通流量预测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 3. 缓存函数
# --------------------------
@st.cache_data
def load_npz(file):
    """加载npz文件，返回data字段"""
    try:
        data = np.load(file, allow_pickle=True)
        if 'data' not in data.files:
            raise ValueError("npz文件中未找到'data'字段")
        return data['data']
    except Exception as e:
        st.error(f"加载文件失败: {str(e)}")
        return None

@st.cache_resource
def load_stgcn_model(model_path, num_nodes=307):
    """加载预训练的STGCN模型"""
    try:
        model = STGCN(
            num_nodes=num_nodes,
            in_channels=1,
            hidden_channels=64,
            num_layers=3,
            pred_len=12,
            kernel_size=3
        )
        
        # 加载权重
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        st.success(f"✅ STGCN模型加载成功 (设备: {device})")
        return model, device
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None, None

# --------------------------
# 4. 侧边栏 - 文件上传 & 配置
# --------------------------
st.sidebar.header("📤 数据上传")

actual_uploaded_file = st.sidebar.file_uploader(
    "上传真实交通数据 (.npz)",
    type="npz",
    key="actual_data"
)

st.sidebar.header("🤖 模型配置")

# 模型路径输入
model_path = st.sidebar.text_input(
    "STGCN模型路径",
    value="D:/Graduation Project/best_stgcn_model.pth",
    help="请输入预训练的STGCN模型文件路径"
)

# 加载模型按钮
if st.sidebar.button("🚀 加载STGCN模型"):
    with st.spinner("正在加载模型..."):
        model, device = load_stgcn_model(model_path)
        if model is not None:
            st.session_state['model'] = model
            st.session_state['device'] = device
            st.sidebar.success("模型已加载到会话中")

# 检查模型是否已加载
if 'model' not in st.session_state:
    st.sidebar.warning("⚠️ 请先加载STGCN模型")
else:
    st.sidebar.success("✅ 模型已加载")

# --------------------------
# 5. 核心逻辑
# --------------------------
st.title("🚦 STGCN交通流量预测系统")

# 数据校验
if actual_uploaded_file is None:
    st.info("请先在侧边栏上传**真实数据**（.npz格式）")
    st.stop()

actual_data = load_npz(actual_uploaded_file)
if actual_data is None:
    st.error("无法加载数据文件")
    st.stop()

# 提取维度信息
timesteps, num_nodes, num_features = actual_data.shape
st.success(f"✅ 数据加载成功 | 时间步: {timesteps} | 节点数: {num_nodes} | 特征数: {num_features}")

# 侧边栏 - 可视化配置
st.sidebar.header("⚙️ 可视化配置")
selected_node = st.sidebar.selectbox(
    "选择监测节点",
    options=list(range(num_nodes)),
    format_func=lambda x: f"节点 {x}",
    key="node_selector"
)

# --------------------------
# 6. 模型预测功能
# --------------------------
st.header("🤖 STGCN模型预测")

if st.button("🔮 运行流量预测", type="primary"):
    if 'model' not in st.session_state:
        st.error("请先加载STGCN模型！")
    else:
        with st.spinner("正在进行流量预测..."):
            try:
                model = st.session_state['model']
                device = st.session_state['device']
                
                # 提取流量数据（第一个特征）
                flow_data = actual_data[:, :, 0]  # [timesteps, num_nodes]
                
                # 准备输入数据（这里简化处理，实际需要按照训练时的格式）
                # 假设使用最近12个时间步预测未来12个时间步
                seq_len = 12
                pred_len = 12
                
                # 创建简单的邻接矩阵（线性拓扑）
                adj_matrix = np.zeros((num_nodes, num_nodes))
                for i in range(num_nodes):
                    if i > 0:
                        adj_matrix[i][i-1] = 1
                    if i < num_nodes - 1:
                        adj_matrix[i][i+1] = 1
                
                adj_tensor = torch.FloatTensor(adj_matrix).to(device)
                
                # 分批预测
                predictions = []
                batch_size = 32
                
                for i in range(0, timesteps - seq_len, batch_size):
                    batch_data = []
                    for j in range(i, min(i + batch_size, timesteps - seq_len)):
                        # 提取序列
                        seq = flow_data[j:j+seq_len, :]  # [seq_len, num_nodes]
                        batch_data.append(seq)
                    
                    if len(batch_data) == 0:
                        continue
                    
                    batch_tensor = torch.FloatTensor(np.array(batch_data)).to(device)
                    
                    with torch.no_grad():
                        batch_pred = model(batch_tensor, adj_tensor)
                        predictions.append(batch_pred.cpu().numpy())
                
                if predictions:
                    predictions = np.concatenate(predictions, axis=0)  # [n_samples, pred_len, num_nodes]
                    
                    # 保存预测结果到session state
                    st.session_state['predictions'] = predictions
                    
                    # 显示预测结果形状
                    st.success(f"✅ 预测完成！预测结果维度: {predictions.shape}")
                    
                    # 显示第一个样本的预测结果
                    st.subheader("📊 第一个预测样本示例")
                    sample_pred = predictions[0]  # [pred_len, num_nodes]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**预测形状:**")
                        st.write(f"时间步: {sample_pred.shape[0]}")
                        st.write(f"节点数: {sample_pred.shape[1]}")
                    
                    with col2:
                        st.write("**统计信息:**")
                        st.write(f"最小值: {np.min(sample_pred):.2f}")
                        st.write(f"最大值: {np.max(sample_pred):.2f}")
                        st.write(f"平均值: {np.mean(sample_pred):.2f}")
                else:
                    st.error("预测失败，无有效预测结果")
                    
            except Exception as e:
                st.error(f"预测过程中出错: {str(e)}")

# 检查是否有预测结果
if 'predictions' not in st.session_state:
    st.info("👆 点击上方按钮运行模型预测")
    # 如果没有预测数据，使用零数据作为占位符
    predicted_data = np.zeros_like(actual_data)
else:
    # 将预测结果转换为与真实数据相同的格式
    predictions = st.session_state['predictions']
    # 这里简化处理，实际需要根据预测的时间范围进行对齐
    predicted_data = np.zeros_like(actual_data)
    # 只填充流量通道
    pred_timesteps = min(predictions.shape[0], timesteps)
    for i in range(pred_timesteps):
        for t in range(min(predictions.shape[1], 12)):
            if i + t < timesteps:
                predicted_data[i + t, :, 0] = predictions[i, t, :]

# --------------------------
# 7. 时序对比可视化（Plotly）
# --------------------------
st.header("📈 流量 - 真实值 vs 预测值")

# 提取时序数据
actual_series = actual_data[:, selected_node, 0]  # 流量特征
predicted_series = predicted_data[:, selected_node, 0]  # 流量特征

# 构建DataFrame
df_plot = pd.DataFrame({
    "时间步": np.arange(timesteps),
    "真实值": actual_series,
    "预测值": predicted_series
})

# Plotly绘图
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df_plot["时间步"], y=df_plot["真实值"],
    mode='lines', name='真实值', line=dict(color='#1f77b4', width=2),
    hovertemplate="时间步: %{x}<br>真实值: %{y:.2f}<extra></extra>"
))
fig_ts.add_trace(go.Scatter(
    x=df_plot["时间步"], y=df_plot["预测值"],
    mode='lines', name='预测值', line=dict(color='#ff7f0e', width=2, dash='dash'),
    hovertemplate="时间步: %{x}<br>预测值: %{y:.2f}<extra></extra>"
))

fig_ts.update_layout(
    xaxis_title="时间步",
    yaxis_title="流量",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
    height=500,
    template="plotly_white"
)
st.plotly_chart(fig_ts, use_container_width=True)

# --------------------------
# 8. 统计信息 & 评估指标
# --------------------------
st.header("📊 统计信息 & 评估指标")

# 基础统计
col1, col2 = st.columns(2)
with col1:
    st.subheader("真实值统计")
    st.write(f"最小值: {np.min(actual_series):.2f}")
    st.write(f"最大值: {np.max(actual_series):.2f}")
    st.write(f"平均值: {np.mean(actual_series):.2f}")
    st.write(f"标准差: {np.std(actual_series):.2f}")

with col2:
    st.subheader("预测值统计")
    st.write(f"最小值: {np.min(predicted_series):.2f}")
    st.write(f"最大值: {np.max(predicted_series):.2f}")
    st.write(f"平均值: {np.mean(predicted_series):.2f}")
    st.write(f"标准差: {np.std(predicted_series):.2f}")

# 评估指标（只在有预测数据时计算）
if 'predictions' in st.session_state:
    st.subheader("模型评估指标")
    
    # 确保长度一致
    min_len = min(len(actual_series), len(predicted_series))
    actual_for_eval = actual_series[:min_len]
    predicted_for_eval = predicted_series[:min_len]
    
    # 计算指标
    mae = mean_absolute_error(actual_for_eval, predicted_for_eval)
    rmse = np.sqrt(mean_squared_error(actual_for_eval, predicted_for_eval))
    # 避免除零
    mask = actual_for_eval != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual_for_eval[mask] - predicted_for_eval[mask]) / actual_for_eval[mask])) * 100
    else:
        mape = 0.0
    
    col_mae, col_rmse, col_mape = st.columns(3)
    col_mae.metric("MAE (平均绝对误差)", f"{mae:.2f}")
    col_rmse.metric("RMSE (均方根误差)", f"{rmse:.2f}")
    col_mape.metric("MAPE (平均绝对百分比误差)", f"{mape:.2f}%")

# 数据样本展示
st.subheader("📋 数据样本（前50行）")
st.dataframe(df_plot.head(50), use_container_width=True)

# --------------------------
# 9. 时间步筛选 & 分布直方图
# --------------------------
st.header("⏱️ 单时间步数据分析")
timestep = st.slider(
    "选择时间步",
    min_value=0,
    max_value=timesteps-1,
    value=0,
    key="timestep_slider"
)

# 提取该时间步所有节点的流量
flow = actual_data[timestep, :, 0]

# 时间步统计
st.subheader(f"时间步 {timestep} - 全节点流量统计")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("最小值", f"{flow.min():.2f}")
with col2:
    st.metric("最大值", f"{flow.max():.2f}")
with col3:
    st.metric("平均值", f"{flow.mean():.2f}")

# 分布直方图（只保留流量）
st.subheader(f"时间步 {timestep} - 流量分布直方图")
fig_hist, ax = plt.subplots(figsize=(10, 5))
ax.hist(flow, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
ax.set_title("流量分布", fontsize=14)
ax.set_xlabel("流量", fontsize=12)
ax.set_ylabel("频次", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_hist)

# --------------------------
# 10. 网络拓扑可视化
# --------------------------
st.header("🌐 交通网络拓扑可视化")

# 构建邻接矩阵（线性拓扑）
adj_matrix = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    if i > 0:
        adj_matrix[i][i-1] = 1
    if i < num_nodes - 1:
        adj_matrix[i][i+1] = 1

# 创建网络图
G = nx.from_numpy_array(adj_matrix)
for node in G.nodes():
    G.nodes[node]['flow'] = flow[node]

# 节点位置
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
    node_color.append(G.nodes[node]['flow'])
    node_text.append(f'节点 {node}<br>流量: {node_color[-1]:.2f}')

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers', hoverinfo='text', text=node_text,
    marker=dict(
        showscale=True,
        colorscale='Viridis',
        color=node_color,
        size=10,
        colorbar=dict(
            thickness=15,
            title="流量",
            xanchor='left',
            titleside='right'
        ),
        line_width=2
    )
)

# 网络图布局
fig_network = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=f'交通网络 - 流量分布 (时间步 {timestep})',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
)
st.plotly_chart(fig_network, use_container_width=True)

# --------------------------
# 11. 说明信息
# --------------------------
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 使用步骤：
    1. **加载模型**：在侧边栏输入模型路径，点击"加载STGCN模型"
    2. **上传数据**：上传包含真实交通数据的.npz文件
    3. **运行预测**：点击"运行流量预测"按钮生成预测结果
    4. **交互查看**：选择不同节点和时间步查看详细数据
    
    ### 数据格式要求：
    - 文件格式：.npz
    - 数据形状：(时间步, 节点数, 特征数)
    - 特征：第一个特征必须是流量数据
    
    ### 功能说明：
    - **时序对比**：显示选定节点的真实值与预测值对比
    - **统计信息**：显示基本统计量和评估指标
    - **单时间步分析**：分析特定时间步的全节点流量分布
    - **网络拓扑**：可视化交通网络结构及流量分布
    """)