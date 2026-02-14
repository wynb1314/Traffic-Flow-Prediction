"""
streamlit run main_system.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="STGCN 交通流量预测系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 模型定义 ====================
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


# ==================== 工具函数 ====================
@st.cache_resource
def load_model(model_path, num_nodes=307):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果使用GPU，先清理显存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    model = STGCN(
        num_nodes=num_nodes,
        hidden_channels=64,
        num_layers=3,
        pred_len=12,
        kernel_size=3
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def calculate_metrics(pred, true):
    """计算评估指标"""
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
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


def predict(model, X, adj, device, scaler, batch_size=32):
    """进行预测（批处理以节省显存）"""
    model.eval()
    predictions = []
    
    # 先将邻接矩阵移到设备上（只需一次）
    adj_tensor = torch.FloatTensor(adj).to(device)
    
    # 批处理预测
    num_samples = len(X)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i+batch_size]
            X_tensor = torch.FloatTensor(batch_X).to(device)
            
            # 预测
            pred = model(X_tensor, adj_tensor)
            predictions.append(pred.cpu().numpy())
            
            # 清理显存
            del X_tensor, pred
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # 合并所有批次的预测
    pred_np = np.concatenate(predictions, axis=0)
    
    # 清理邻接矩阵
    del adj_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 反标准化
    mean, std = scaler
    pred_real = pred_np * std + mean
    
    return pred_real


# ==================== 主应用 ====================
def main():
    # 标题
    st.markdown('<h1 class="main-header">🚗 STGCN 交通流量预测系统</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏配置
    st.sidebar.title("⚙️ 系统配置")
    
    # 模型路径
    model_path = st.sidebar.text_input(
        "模型路径",
        value=r"D:\Graduation Project\best_stgcn_model.pth"
    )
    
    # 数据加载方式选择
    st.sidebar.markdown("### 📁 数据加载")
    data_load_method = st.sidebar.radio(
        "选择数据加载方式",
        ["本地路径", "文件上传"]
    )
    
    # GPU设置
    st.sidebar.markdown("### ⚙️ GPU设置")
    use_gpu = st.sidebar.checkbox("使用GPU加速", value=torch.cuda.is_available())
    if not use_gpu:
        st.sidebar.info("💡 将使用CPU进行预测")
    
    batch_size_predict = st.sidebar.select_slider(
        "预测批次大小",
        options=[8, 16, 32, 64, 128],
        value=32,
        help="显存不足时减小批次大小"
    )
    
    uploaded_file = None
    data_path_input = None
    
    if data_load_method == "本地路径":
        data_path_input = st.sidebar.text_input(
            "数据文件路径",
            value=r"D:\Graduation Project\processed_data.npz",
            help="输入 .npz 文件的完整路径"
        )
        load_button = st.sidebar.button("🔄 加载数据", type="primary", use_container_width=True)
    else:
        st.sidebar.info("💡 如果文件超过200MB，建议使用「本地路径」方式")
        uploaded_file = st.sidebar.file_uploader(
            "上传处理后的数据文件 (.npz)",
            type=['npz'],
            help="适用于小于200MB的文件"
        )
        load_button = False
    
    # 判断是否应该加载数据
    should_load = (data_load_method == "本地路径" and load_button and data_path_input) or \
                  (data_load_method == "文件上传" and uploaded_file is not None)
    
    if should_load:
        # 加载数据
        try:
            if data_load_method == "本地路径" and data_path_input:
                if not data_path_input.endswith('.npz'):
                    st.sidebar.error("❌ 请输入有效的 .npz 文件路径")
                    return
                st.sidebar.info("⏳ 正在加载数据...")
                data = np.load(data_path_input)
            else:
                data = np.load(uploaded_file)
            
            X_test = data['X_test']
            y_test = data['y_test']
            adj_matrix = data['adj_matrix']
            scaler = (data['mean'].item(), data['std'].item())
            
            st.sidebar.success(f"✅ 数据加载成功！")
            st.sidebar.info(f"测试样本数: {len(X_test)}")
            st.sidebar.info(f"节点数量: {adj_matrix.shape[0]}")
            
        except FileNotFoundError:
            st.sidebar.error("❌ 文件未找到，请检查路径是否正确")
            return
        except Exception as e:
            st.sidebar.error(f"❌ 数据加载失败: {e}")
            return
        
        # 加载模型
        try:
            # 根据用户选择决定是否使用GPU
            actual_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            
            model, device = load_model(model_path, num_nodes=adj_matrix.shape[0])
            
            # 如果用户选择不用GPU，将模型移到CPU
            if not use_gpu and device.type == 'cuda':
                model = model.cpu()
                device = torch.device('cpu')
                torch.cuda.empty_cache()
            
            st.sidebar.success(f"✅ 模型加载成功！")
            st.sidebar.info(f"运行设备: {device}")
            
            if device.type == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.sidebar.info(f"GPU显存: {gpu_memory:.1f} GB")
                
        except Exception as e:
            st.sidebar.error(f"❌ 模型加载失败: {e}")
            return
        
        # 节点选择
        st.sidebar.markdown("### 🎯 路段选择")
        node_id = st.sidebar.selectbox(
            "选择路段/传感器 ID",
            range(adj_matrix.shape[0]),
            index=0
        )
        
        # 样本选择
        sample_idx = st.sidebar.slider(
            "选择测试样本",
            0, len(X_test)-1, 0
        )
        
        # ==================== 主界面标签页 ====================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 预测结果", 
            "📈 统计分析", 
            "🎯 模型评估", 
            "📋 数据样本",
            "🌐 网络拓扑"
        ])
        
        # ==================== Tab 1: 预测结果 ====================
        with tab1:
            st.header("🔮 实时预测与对比")
            
            # 进行预测
            with st.spinner("正在进行预测..."):
                try:
                    predictions = predict(model, X_test, adj_matrix, device, scaler, batch_size=batch_size_predict)
                    mean, std = scaler
                    y_test_real = y_test * std + mean
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        st.error("❌ GPU显存不足！请尝试：")
                        st.error("1. 减小预测批次大小（侧边栏设置）")
                        st.error("2. 取消勾选「使用GPU加速」使用CPU")
                        st.error("3. 关闭其他占用GPU的程序")
                        return
                    else:
                        raise e
            
            # 选择的节点数据
            pred_node = predictions[:, 0, node_id]
            true_node = y_test_real[:, 0, node_id]
            
            # 绘制对比图
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 使用 Plotly 绘制交互式图表
                fig = go.Figure()
                
                plot_len = min(500, len(true_node))
                x_axis = list(range(plot_len))
                
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=true_node[:plot_len],
                    mode='lines',
                    name='真实值',
                    line=dict(color='#1f77b4', width=2),
                    opacity=0.8
                ))
                
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=pred_node[:plot_len],
                    mode='lines',
                    name='预测值',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    opacity=0.8
                ))
                
                fig.update_layout(
                    title=f"路段 {node_id} 的流量预测对比",
                    xaxis_title="时间步 (小时)",
                    yaxis_title="交通流量",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📌 当前样本信息")
                st.metric("样本索引", sample_idx)
                st.metric("路段 ID", node_id)
                st.metric("真实流量", f"{true_node[sample_idx]:.2f}")
                st.metric("预测流量", f"{pred_node[sample_idx]:.2f}")
                
                error = abs(pred_node[sample_idx] - true_node[sample_idx])
                st.metric("预测误差", f"{error:.2f}")
            
            # 单样本详细分析
            st.markdown("---")
            st.subheader("🔍 单样本详细分析")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 输入序列
                input_seq = X_test[sample_idx, :, node_id] * std + mean
                
                fig_input = go.Figure()
                fig_input.add_trace(go.Scatter(
                    y=input_seq,
                    mode='lines+markers',
                    name='输入序列',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=6)
                ))
                
                fig_input.update_layout(
                    title=f"样本 {sample_idx} - 输入序列 (过去12小时)",
                    xaxis_title="时间步",
                    yaxis_title="流量",
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_input, use_container_width=True)
            
            with col2:
                # 预测对比
                true_future = y_test_real[sample_idx, :, node_id]
                pred_future = predictions[sample_idx, :, node_id]
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    y=true_future,
                    mode='lines+markers',
                    name='真实值',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                fig_pred.add_trace(go.Scatter(
                    y=pred_future,
                    mode='lines+markers',
                    name='预测值',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=6)
                ))
                
                fig_pred.update_layout(
                    title=f"样本 {sample_idx} - 预测对比 (未来12小时)",
                    xaxis_title="时间步",
                    yaxis_title="流量",
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
        
        # ==================== Tab 2: 统计分析 ====================
        with tab2:
            st.header("📊 统计信息与分布分析")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总样本数", len(predictions))
            with col2:
                st.metric("节点数量", adj_matrix.shape[0])
            with col3:
                st.metric("平均流量", f"{np.mean(y_test_real):.2f}")
            with col4:
                st.metric("流量标准差", f"{np.std(y_test_real):.2f}")
            
            st.markdown("---")
            
            # 分布对比
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=y_test_real[:, 0, node_id],
                    name='真实值',
                    opacity=0.7,
                    marker_color='#1f77b4',
                    nbinsx=50
                ))
                fig.add_trace(go.Histogram(
                    x=predictions[:, 0, node_id],
                    name='预测值',
                    opacity=0.7,
                    marker_color='#ff7f0e',
                    nbinsx=50
                ))
                
                fig.update_layout(
                    title=f"路段 {node_id} 流量分布对比",
                    xaxis_title="流量",
                    yaxis_title="频次",
                    barmode='overlay',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 散点图
                fig = go.Figure()
                
                sample_size = min(1000, len(true_node))
                indices = np.random.choice(len(true_node), sample_size, replace=False)
                
                fig.add_trace(go.Scatter(
                    x=true_node[indices],
                    y=pred_node[indices],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=np.abs(true_node[indices] - pred_node[indices]),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="误差")
                    ),
                    name='预测点'
                ))
                
                # 添加理想线
                max_val = max(true_node.max(), pred_node.max())
                fig.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='理想线'
                ))
                
                fig.update_layout(
                    title=f"路段 {node_id} 真实值 vs 预测值",
                    xaxis_title="真实值",
                    yaxis_title="预测值",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 时序统计
            st.markdown("---")
            st.subheader("📈 时序统计分析")
            
            # 计算每小时平均流量
            hours = np.arange(24)
            hourly_true = []
            hourly_pred = []
            
            for h in hours:
                mask = np.arange(len(true_node)) % 24 == h
                hourly_true.append(true_node[mask].mean())
                hourly_pred.append(pred_node[mask].mean())
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hours,
                y=hourly_true,
                name='真实值',
                marker_color='#1f77b4',
                opacity=0.7
            ))
            fig.add_trace(go.Bar(
                x=hours,
                y=hourly_pred,
                name='预测值',
                marker_color='#ff7f0e',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f"路段 {node_id} 每小时平均流量",
                xaxis_title="小时",
                yaxis_title="平均流量",
                barmode='group',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ==================== Tab 3: 模型评估 ====================
        with tab3:
            st.header("🎯 模型性能评估")
            
            # 计算全局指标
            with st.spinner("计算评估指标..."):
                mae, rmse, mape, r2 = calculate_metrics(predictions, y_test_real)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MAE", f"{mae:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE", f"{rmse:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MAPE", f"{mape:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("R²", f"{r2:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 各节点性能对比
            st.subheader("📊 各路段预测性能对比")
            
            with st.spinner("计算各节点指标..."):
                node_metrics = []
                
                # 随机选择若干节点进行展示
                selected_nodes = np.random.choice(
                    adj_matrix.shape[0], 
                    min(20, adj_matrix.shape[0]), 
                    replace=False
                )
                
                for n in selected_nodes:
                    pred_n = predictions[:, 0, n]
                    true_n = y_test_real[:, 0, n]
                    mae_n = np.mean(np.abs(pred_n - true_n))
                    rmse_n = np.sqrt(np.mean((pred_n - true_n) ** 2))
                    node_metrics.append({
                        'Node': n,
                        'MAE': mae_n,
                        'RMSE': rmse_n
                    })
                
                df_metrics = pd.DataFrame(node_metrics)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_metrics, 
                    x='Node', 
                    y='MAE',
                    title='各路段 MAE 对比',
                    color='MAE',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    df_metrics, 
                    x='Node', 
                    y='RMSE',
                    title='各路段 RMSE 对比',
                    color='RMSE',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # 误差分析
            st.markdown("---")
            st.subheader("🔍 误差分析")
            
            errors = pred_node - true_node
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=50,
                    marker_color='#d62728',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title=f"路段 {node_id} 误差分布",
                    xaxis_title="预测误差",
                    yaxis_title="频次",
                    height=350,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"""
                **误差统计:**
                - 平均误差: {np.mean(errors):.4f}
                - 误差标准差: {np.std(errors):.4f}
                - 最大正误差: {np.max(errors):.4f}
                - 最大负误差: {np.min(errors):.4f}
                """)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=errors,
                    mode='lines',
                    line=dict(color='#d62728', width=1),
                    name='误差'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                
                fig.update_layout(
                    title=f"路段 {node_id} 误差时序图",
                    xaxis_title="样本序号",
                    yaxis_title="预测误差",
                    height=350,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # ==================== Tab 4: 数据样本 ====================
        with tab4:
            st.header("📋 数据样本查看")
            
            # 创建数据表
            st.subheader(f"🔍 路段 {node_id} 前50个样本")
            
            sample_data = []
            for i in range(min(50, len(predictions))):
                sample_data.append({
                    '样本ID': i,
                    '真实值': f"{y_test_real[i, 0, node_id]:.2f}",
                    '预测值': f"{predictions[i, 0, node_id]:.2f}",
                    '绝对误差': f"{abs(predictions[i, 0, node_id] - y_test_real[i, 0, node_id]):.2f}",
                    '相对误差(%)': f"{abs(predictions[i, 0, node_id] - y_test_real[i, 0, node_id]) / max(y_test_real[i, 0, node_id], 1) * 100:.2f}"
                })
            
            df_samples = pd.DataFrame(sample_data)
            st.dataframe(df_samples, use_container_width=True, height=400)
            
            # 下载按钮
            csv = df_samples.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载样本数据 (CSV)",
                data=csv,
                file_name=f"node_{node_id}_samples.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # 输入序列查看
            st.subheader(f"📊 样本 {sample_idx} 输入序列")
            
            input_seq = X_test[sample_idx, :, node_id] * std + mean
            input_df = pd.DataFrame({
                '时间步': range(len(input_seq)),
                '流量值': input_seq
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(input_df, use_container_width=True)
            
            with col2:
                st.info(f"""
                **输入序列统计:**
                - 序列长度: {len(input_seq)}
                - 平均值: {np.mean(input_seq):.2f}
                - 最大值: {np.max(input_seq):.2f}
                - 最小值: {np.min(input_seq):.2f}
                - 标准差: {np.std(input_seq):.2f}
                """)
        
        # ==================== Tab 5: 网络拓扑 ====================
        with tab5:
            st.header("🌐 交通网络拓扑可视化")
            
            st.info("此功能展示交通传感器网络的连接关系，基于邻接矩阵构建。")
            
            # 构建网络图
            threshold = st.slider(
                "邻接矩阵阈值 (只显示连接强度大于此值的边)",
                0.0, 1.0, 0.1, 0.05
            )
            
            with st.spinner("正在构建网络图..."):
                G = nx.Graph()
                
                # 添加节点
                for i in range(min(50, adj_matrix.shape[0])):  # 限制显示节点数
                    G.add_node(i)
                
                # 添加边
                for i in range(min(50, adj_matrix.shape[0])):
                    for j in range(i+1, min(50, adj_matrix.shape[0])):
                        if adj_matrix[i, j] > threshold:
                            G.add_edge(i, j, weight=adj_matrix[i, j])
                
                # 使用spring布局
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                # 创建边的轨迹
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # 创建节点的轨迹
                node_x = []
                node_y = []
                node_text = []
                node_colors = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # 节点信息
                    degree = G.degree(node)
                    node_text.append(f"节点 {node}<br>连接数: {degree}")
                    
                    # 高亮当前选择的节点
                    if node == node_id:
                        node_colors.append('#ff0000')
                    else:
                        node_colors.append('#1f77b4')
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=[str(i) for i in G.nodes()],
                    textposition="top center",
                    hovertext=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='YlOrRd',
                        size=15,
                        color=node_colors,
                        line_width=2
                    )
                )
                
                # 创建图形
                fig = go.Figure(data=[edge_trace, node_trace],
                               layout=go.Layout(
                                   title=f'交通网络拓扑图 (前50个节点, 阈值={threshold})',
                                   titlefont_size=16,
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=0, l=0, r=0, t=40),
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   height=600
                               ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 网络统计信息
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("节点数", G.number_of_nodes())
            with col2:
                st.metric("边数", G.number_of_edges())
            with col3:
                st.metric("平均度数", f"{np.mean([d for n, d in G.degree()]):.2f}")
            with col4:
                if G.number_of_edges() > 0:
                    density = nx.density(G)
                    st.metric("网络密度", f"{density:.4f}")
                else:
                    st.metric("网络密度", "0.0000")
            
            st.markdown("---")
            
            # 节点度数分布
            st.subheader("📊 节点度数分布")
            
            degrees = [d for n, d in G.degree()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=degrees,
                nbinsx=20,
                marker_color='#1f77b4',
                opacity=0.7
            ))
            
            fig.update_layout(
                title='节点度数分布直方图',
                xaxis_title='度数',
                yaxis_title='节点数量',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 邻接矩阵热力图
            st.markdown("---")
            st.subheader("🔥 邻接矩阵热力图")
            
            sample_size = min(30, adj_matrix.shape[0])
            adj_sample = adj_matrix[:sample_size, :sample_size]
            
            fig = go.Figure(data=go.Heatmap(
                z=adj_sample,
                colorscale='Viridis',
                colorbar=dict(title="连接强度")
            ))
            
            fig.update_layout(
                title=f'邻接矩阵热力图 (前{sample_size}×{sample_size})',
                xaxis_title='节点',
                yaxis_title='节点',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 邻接矩阵统计
            st.info(f"""
            **邻接矩阵统计信息:**
            - 矩阵维度: {adj_matrix.shape[0]} × {adj_matrix.shape[1]}
            - 非零元素: {np.count_nonzero(adj_matrix)}
            - 稀疏度: {(1 - np.count_nonzero(adj_matrix) / adj_matrix.size) * 100:.2f}%
            - 最大连接强度: {np.max(adj_matrix):.4f}
            - 平均连接强度: {np.mean(adj_matrix):.4f}
            """)
    
    else:
        st.info("👈 请在侧边栏选择数据加载方式")
        
        st.markdown("""
        ### 📖 使用说明
        
        #### 1️⃣ 准备工作
        - 确保已训练好 STGCN 模型 (`.pth` 文件)
        - 准备处理后的数据文件 (`.npz` 格式)
        
        #### 2️⃣ 数据加载方式
        
        **方式一: 本地路径 (推荐 - 支持大文件)**
        - ✅ 适用于任意大小的文件
        - ✅ 加载速度快
        - 在侧边栏输入文件完整路径，例如:
          - `D:\\Graduation Project\\processed_data.npz`
          - `/home/user/data/processed_data.npz`
        
        **方式二: 文件上传**
        - ⚠️ 仅支持小于 200MB 的文件
        - 适用于小型数据集或演示
        
        #### 3️⃣ 数据文件要求
        数据文件应包含以下字段:
        - `X_test`: 测试输入数据
        - `y_test`: 测试标签数据
        - `adj_matrix`: 邻接矩阵
        - `mean`: 标准化均值
        - `std`: 标准化标准差
        
        #### 4️⃣ 功能介绍
        
        **📊 预测结果**
        - 查看任意路段的流量预测对比
        - 分析单个样本的输入序列和预测结果
        - 实时查看预测误差
        
        **📈 统计分析**
        - 流量分布对比分析
        - 预测值与真实值相关性分析
        - 时序模式分析(每小时平均流量)
        
        **🎯 模型评估**
        - 全局性能指标: MAE, RMSE, MAPE, R²
        - 各路段预测性能对比
        - 详细误差分析
        
        **📋 数据样本**
        - 查看详细的样本数据
        - 下载数据表格 (CSV 格式)
        - 检查输入序列统计信息
        
        **🌐 网络拓扑**
        - 可视化交通网络结构
        - 分析节点连接关系
        - 查看邻接矩阵热力图
        
        #### 5️⃣ 开始使用
        1. 在侧边栏选择「本地路径」方式
        2. 输入模型路径和数据文件路径
        3. 选择要分析的路段和样本
        4. 在不同标签页查看分析结果
        
        ---
        
        💡 **提示**: 
        - 所有图表都支持交互操作，可以缩放、平移、悬停查看详情！
        - 对于大文件(>200MB)，请务必使用「本地路径」方式
        """)


if __name__ == "__main__":
    main()