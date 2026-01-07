# -*- coding: utf-8 -*-
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

# 忽略无关警告
warnings.filterwarnings('ignore')

# --------------------------
# 1. 页面配置 & 基础设置
# --------------------------
st.set_page_config(
    page_title="Traffic Flow Prediction", #交通流量预测仪表盘
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体（解决matplotlib中文乱码）
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 2. 缓存函数优化（适配新版Streamlit）
# --------------------------
@st.cache_data  # 替代旧的st.cache，新版推荐用法
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

# --------------------------
# 3. 侧边栏 - 文件上传 & 配置
# --------------------------
st.sidebar.header("📤 数据上传")
actual_uploaded_file = st.sidebar.file_uploader(
    "上传真实交通数据 (.npz)",
    type="npz",
    key="actual_data"
)
predicted_uploaded_file = st.sidebar.file_uploader(
    "上传预测交通数据 (.npz)",
    type="npz",
    key="pred_data"
)

# 加载数据
actual_data = load_npz(actual_uploaded_file) if actual_uploaded_file else None
predicted_data = load_npz(predicted_uploaded_file) if predicted_uploaded_file else None

# --------------------------
# 4. 核心逻辑 - 数据校验 & 可视化
# --------------------------
st.title("🚦 交通流量预测系统")

# 数据校验
if actual_data is None or predicted_data is None:
    st.info("请先在侧边栏上传**真实数据**和**预测数据**（均为.npz格式）")
    st.stop()

if actual_data.shape != predicted_data.shape:
    st.error(f"数据维度不匹配！真实数据维度: {actual_data.shape}, 预测数据维度: {predicted_data.shape}")
    st.error("要求维度格式：(时间步, 节点数, 特征数)")
    st.stop()

# 提取维度信息
timesteps, num_nodes, num_features = actual_data.shape
st.success(f"✅ 数据加载成功 | 时间步: {timesteps} | 节点数: {num_nodes} | 特征数: {num_features}")

# 侧边栏 - 可视化配置
st.sidebar.header("⚙️ 可视化配置")
feature_options = ["Flow（流量）", "Occupancy（占用率）", "Speed（速度）"]
selected_feature = st.sidebar.selectbox("选择特征", feature_options)
feature_idx = {"Flow（流量）":0, "Occupancy（占用率）":1, "Speed（速度）":2}[selected_feature]

selected_node = st.sidebar.selectbox(
    "选择监测节点",
    options=list(range(num_nodes)),
    format_func=lambda x: f"节点 {x}",
    key="node_selector"
)

# --------------------------
# 5. 时序对比可视化（Plotly）
# --------------------------
st.header(f"📈 {selected_feature} - 真实值 vs 预测值 (节点 {selected_node})")

# 提取时序数据
actual_series = actual_data[:, selected_node, feature_idx]
predicted_series = predicted_data[:, selected_node, feature_idx]

# 构建DataFrame
df_plot = pd.DataFrame({
    "时间步": np.arange(timesteps),
    "真实值": actual_series,
    "预测值": predicted_series
})

# Plotly绘图（增强交互）
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
    yaxis_title=selected_feature,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
    height=500,
    template="plotly_white"
)
st.plotly_chart(fig_ts, use_container_width=True)

# --------------------------
# 6. 统计信息 & 评估指标
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

# 评估指标
st.subheader("模型评估指标")
mae = mean_absolute_error(actual_series, predicted_series)
rmse = np.sqrt(mean_squared_error(actual_series, predicted_series))
mape = np.mean(np.abs((actual_series - predicted_series) / (actual_series + 1e-8))) * 100  # 避免除零

col_mae, col_rmse, col_mape = st.columns(3)
col_mae.metric("MAE (平均绝对误差)", f"{mae:.2f}")
col_rmse.metric("RMSE (均方根误差)", f"{rmse:.2f}")
col_mape.metric("MAPE (平均绝对百分比误差)", f"{mape:.2f}%")

# 数据样本展示
st.subheader("📋 数据样本（前50行）")
st.dataframe(df_plot.head(50), use_container_width=True)

# --------------------------
# 7. 时间步筛选 & 分布直方图
# --------------------------
st.header("⏱️ 单时间步数据分析")
timestep = st.slider(
    "选择时间步",
    min_value=0,
    max_value=timesteps-1,
    value=0,
    key="timestep_slider"
)

# 提取该时间步所有节点的特征
flow = actual_data[timestep, :, 0]
occupancy = actual_data[timestep, :, 1]
speed = actual_data[timestep, :, 2]

# 时间步统计
st.subheader(f"时间步 {timestep} - 全节点统计")
col_flow, col_occ, col_speed = st.columns(3)
with col_flow:
    st.write("**流量**")
    st.write(f"最小值: {flow.min():.2f}")
    st.write(f"最大值: {flow.max():.2f}")
with col_occ:
    st.write("**占用率 (%)**")
    st.write(f"最小值: {occupancy.min():.4f}")
    st.write(f"最大值: {occupancy.max():.4f}")
with col_speed:
    st.write("**速度 (km/h)**")
    st.write(f"最小值: {speed.min():.2f}")
    st.write(f"最大值: {speed.max():.2f}")

# 分布直方图
st.subheader(f"时间步 {timestep} - 特征分布直方图")
fig_hist, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(flow, bins=30, color='#1f77b4', alpha=0.7)
axes[0].set_title("流量分布")
axes[0].set_xlabel("流量")
axes[0].set_ylabel("频次")

axes[1].hist(occupancy, bins=30, color='#2ca02c', alpha=0.7)
axes[1].set_title("占用率分布")
axes[1].set_xlabel("占用率 (%)")
axes[1].set_ylabel("频次")

axes[2].hist(speed, bins=30, color='#d62728', alpha=0.7)
axes[2].set_title("速度分布")
axes[2].set_xlabel("速度 (km/h)")
axes[2].set_ylabel("频次")

plt.tight_layout()
st.pyplot(fig_hist)

# --------------------------
# 8. 网络拓扑可视化
# --------------------------
st.header("🌐 交通网络拓扑可视化")
selected_feature_graph = st.selectbox(
    "选择网络节点着色特征",
    ["Flow（流量）", "Occupancy（占用率）", "Speed（速度）"],
    key="graph_feature"
)
feature_idx_graph = {"Flow（流量）":0, "Occupancy（占用率）":1, "Speed（速度）":2}[selected_feature_graph]

# 提取该时间步的特征数据
feature_data = actual_data[timestep, :, feature_idx_graph]

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
    G.nodes[node][selected_feature_graph.split("（")[0].lower()] = feature_data[node]

# 节点位置（固定seed保证布局一致）
pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(num_nodes))  # 适配节点数调整布局

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
    node_color.append(G.nodes[node][selected_feature_graph.split("（")[0].lower()])
    node_text.append(f'节点 {node}<br>{selected_feature_graph}: {node_color[-1]:.2f}')

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
            title=selected_feature_graph,
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
        title=f'交通网络 - {selected_feature_graph} (时间步 {timestep})',
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