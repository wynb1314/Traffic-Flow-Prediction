# -*- coding: utf-8 -*-
"""
Dashboard views: 数据探索、智能预测、系统仪表盘
"""
import json
import sys
from pathlib import Path

import numpy as np
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect

# 项目根目录加入路径以便导入 STGCN
BASE_DIR = Path(settings.BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 延迟导入，避免 Django 启动时未安装 torch 报错
def _load_data():
    data_path = getattr(settings, 'PROCESSED_DATA_PATH', BASE_DIR / 'processed_data.npz')
    if not Path(data_path).exists():
        return None
    return np.load(data_path, allow_pickle=True)


def _get_adj_and_scaler():
    data = _load_data()
    if data is None:
        return None, None, None
    adj = data['adj_matrix']
    # 支持节点级和全局标准化
    mean_data = data['mean']
    std_data = data['std']
    if mean_data.ndim == 0:
        # 全局标准化（标量）
        scaler = (float(mean_data.item()), float(std_data.item()))
    else:
        # 节点级标准化（数组）
        scaler = (mean_data, std_data)
    return data, adj, scaler


def home_redirect(request):
    return redirect('data_explore')


def data_explore(request):
    return render(request, 'dashboard/data_explore.html', {'page': 'explore'})


def _get_num_cameras():
    """若存在传感器 CSV 则从中统计摄像头数，否则用节点数"""
    csv_path = getattr(settings, 'SENSORS_CSV_PATH', None)
    if csv_path is None:
        csv_path = BASE_DIR / 'PEMS04' / 'sensors.csv'
    for p in [Path(csv_path), BASE_DIR / 'sensors.csv', BASE_DIR / 'PEMS04.csv']:
        if p.exists():
            try:
                import csv as csv_module
                with open(p, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    r = csv_module.reader(f)
                    next(r, None)
                    return sum(1 for _ in r)
            except Exception:
                pass
    return None


def api_explore_stats(request):
    """摄像头数、平均流量（供数据探索页统计卡片）"""
    data, adj, scaler = _get_adj_and_scaler()
    num_nodes = int(adj.shape[0]) if adj is not None else 307
    num_cameras = _get_num_cameras()
    if num_cameras is None:
        num_cameras = num_nodes
    avg_flow = None
    if data is not None and scaler is not None:
        mean, std = scaler
        X_test = data['X_test']
        # 支持节点级标准化
        if isinstance(mean, np.ndarray) and mean.ndim == 1:
            std_expanded = std.reshape(1, 1, -1)
            mean_expanded = mean.reshape(1, 1, -1)
            avg_flow = round(float(np.mean(X_test * std_expanded + mean_expanded)), 1)
        else:
            avg_flow = round(float(np.mean(X_test * std + mean)), 1)
    return JsonResponse({'num_cameras': num_cameras, 'avg_flow': avg_flow})


def api_congestion_top10(request):
    """拥堵路段 Top10：按测试集上平均流量（反标准化）排序取前 10 个节点"""
    data, adj, scaler = _get_adj_and_scaler()
    if data is None or scaler is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    mean, std = scaler
    y_test = data['y_test']
    # 支持节点级标准化
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        std_expanded = std.reshape(1, 1, -1)
        mean_expanded = mean.reshape(1, 1, -1)
        y_real = y_test * std_expanded + mean_expanded
    else:
        y_real = y_test * std + mean
    # 每个节点在所有样本、所有预测步上的平均流量
    node_avg = np.mean(y_real, axis=(0, 1))
    top10_idx = np.argsort(node_avg)[::-1][:10]
    top10 = [{'rank': i + 1, 'node_id': int(idx), 'avg_flow': float(node_avg[idx])} for i, idx in enumerate(top10_idx)]
    return JsonResponse({'top10': top10})


def api_topology_3d(request):
    """2D 拓扑：307 个节点用颜色表示拥堵程度，节点间连线"""
    data, adj, scaler = _get_adj_and_scaler()
    if adj is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError:
        return JsonResponse({'error': '缺少 plotly 或 networkx'}, status=500)
    threshold = float(request.GET.get('threshold', 0.08))
    num_nodes = adj.shape[0]

    # 计算每个节点的平均流量作为拥堵指标
    node_avg = np.zeros(num_nodes)
    if data is not None and scaler is not None:
        mean, std = scaler
        X_test = data['X_test']
        n_samples = min(3000, X_test.shape[0])
        # 支持节点级标准化
        if isinstance(mean, np.ndarray) and mean.ndim == 1:
            std_expanded = std.reshape(1, 1, -1)
            mean_expanded = mean.reshape(1, 1, -1)
            node_avg = np.mean(X_test[:n_samples] * std_expanded + mean_expanded, axis=(0, 1))
        else:
            node_avg = np.mean(X_test[:n_samples] * std + mean, axis=(0, 1))

    # 使用 networkx spring layout 生成 2D 坐标
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] > threshold:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    # 边
    ex, ey = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] > threshold:
                ex.extend([pos[i][0], pos[j][0], None])
                ey.extend([pos[i][1], pos[j][1], None])
    edge_trace = go.Scatter(
        x=ex if ex else [0],
        y=ey if ey else [0],
        mode='lines',
        line=dict(color='rgba(78, 205, 196, 0.3)', width=0.8),
        hoverinfo='none',
        showlegend=False,
    )
    # 节点：颜色根据平均流量（拥堵程度）
    node_x = [pos[i][0] for i in range(num_nodes)]
    node_y = [pos[i][1] for i in range(num_nodes)]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=7,
            color=node_avg.tolist(),
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(
                title=dict(text='平均流量', font=dict(color='#9ca3af')),
                tickfont=dict(color='#9ca3af'),
                bgcolor='rgba(28,31,42,0.8)',
            ),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
        ),
        hovertext=[f'节点 {i}<br>平均流量: {node_avg[i]:.1f}' for i in range(num_nodes)],
        hoverinfo='text',
        showlegend=False,
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text='2D 路网拓扑（307 节点 · 颜色=拥堵程度）', font=dict(size=16, color='#e0e0e0')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(28,31,42,1)',
        plot_bgcolor='rgba(15,17,23,1)',
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        hovermode='closest',
    )
    return JsonResponse(json.loads(fig.to_json()))


def api_polar_period(request):
    """极坐标周期图：日周期（12 步或 24 段）theta = 角度, r = 平均流量"""
    data, adj, scaler = _get_adj_and_scaler()
    if data is None or scaler is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    node = int(request.GET.get('node', 0))
    node = min(max(0, node), adj.shape[0] - 1)
    mean, std = scaler
    X_test = data['X_test']
    n_samples = min(3000, X_test.shape[0])
    # 获取该节点的标准化参数
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        node_mean, node_std = mean[node], std[node]
    else:
        node_mean, node_std = mean, std
    # 12 个时段（对应 12 步）的平均流量
    day_agg = np.mean(X_test[:n_samples, :, node] * node_std + node_mean, axis=0)
    # 极坐标：每段 2*pi/12 弧度
    n_steps = 12
    thetas = (np.arange(n_steps) * 2 * np.pi / n_steps).tolist()
    # 闭合：首尾相接
    r = day_agg.tolist()
    return JsonResponse({'theta': thetas, 'r': r, 'node': node})


def predict_page(request):
    data, adj, _ = _get_adj_and_scaler()
    num_nodes = int(adj.shape[0]) if adj is not None else 307
    return render(request, 'dashboard/predict.html', {
        'page': 'predict',
        'num_nodes': num_nodes,
        'node_list': list(range(num_nodes)),
    })


def system_dashboard(request):
    metrics = {}
    base = BASE_DIR
    for name, path_key in [('stgcn', 'STGCN_RESULTS_PATH'), ('lstm', 'LSTM_RESULTS_PATH')]:
        path = getattr(settings, path_key, base / f'{name}_results.npz')
        path = Path(path)
        if path.exists():
            try:
                r = np.load(path)
                metrics[name] = {
                    'mae': float(r['mae'].item()),
                    'rmse': float(r['rmse'].item()),
                    'mape': float(r['mape'].item()),
                    'r2': float(r['r2'].item()),
                }
            except Exception:
                metrics[name] = None
        else:
            metrics[name] = None
    return render(request, 'dashboard/system_dashboard.html', {
        'page': 'dashboard',
        'metrics': metrics,
    })


def api_topology(request):
    data, adj, _ = _get_adj_and_scaler()
    if adj is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        return JsonResponse({'error': '缺少 networkx 或 plotly'}, status=500)
    threshold = float(request.GET.get('threshold', 0.1))
    max_nodes = min(60, adj.shape[0])
    G = nx.Graph()
    for i in range(max_nodes):
        G.add_node(i)
    for i in range(max_nodes):
        for j in range(i + 1, max_nodes):
            if adj[i, j] > threshold:
                G.add_edge(i, j, weight=float(adj[i, j]))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none'))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=[str(i) for i in G.nodes()],
        textposition='top center', marker=dict(size=10, color='#1f77b4', line=dict(width=1, color='white')),
        hovertext=[f'节点 {i}' for i in G.nodes()]
    ))
    fig.update_layout(
        title=f'路网拓扑 (前{max_nodes}节点, 阈值={threshold})',
        showlegend=False, hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=0, l=0, r=0, t=40), height=450
    )
    return JsonResponse(json.loads(fig.to_json()))


def api_timeseries(request):
    data, adj, scaler = _get_adj_and_scaler()
    if data is None or scaler is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    node = int(request.GET.get('node', 0))
    node = min(max(0, node), adj.shape[0] - 1)
    X_test = data['X_test']
    y_test = data['y_test']
    mean, std = scaler
    # 获取该节点的标准化参数
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        node_mean, node_std = mean[node], std[node]
    else:
        node_mean, node_std = mean, std
    # 取该节点在测试集上的最后一维（时间步）的序列：多个样本拼成长序列
    seq_len = X_test.shape[1]
    steps = min(500, X_test.shape[0])
    hist = (X_test[:steps, :, node] * node_std + node_mean).reshape(-1)
    true_next = (y_test[:steps, 0, node] * node_std + node_mean).tolist()
    x_axis = list(range(len(hist)))
    return JsonResponse({
        'x': x_axis,
        'history': hist.tolist(),
        'true_first_step': true_next,
        'node': node,
    })


def api_patterns(request):
    data, adj, scaler = _get_adj_and_scaler()
    if data is None or scaler is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    node = int(request.GET.get('node', 0))
    node = min(max(0, node), adj.shape[0] - 1)
    mean, std = scaler
    X_test = data['X_test']
    y_test = data['y_test']
    # 获取该节点的标准化参数
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        node_mean, node_std = mean[node], std[node]
    else:
        node_mean, node_std = mean, std
    # 日模式：12 个时间步上的平均流量
    steps_12 = 12
    day_agg = np.zeros(steps_12)
    count = np.zeros(steps_12)
    n_samples = min(2000, X_test.shape[0])
    for t in range(steps_12):
        day_agg[t] = np.mean((X_test[:n_samples, t, node] * node_std + node_mean))
        count[t] = n_samples
    day_agg = (day_agg / (count + 1e-8)).tolist()
    # 周模式：用样本索引模 7 当作"星期几"，每个"日"取平均
    week_agg = []
    for d in range(7):
        mask = (np.arange(X_test.shape[0]) % 7) == d
        subset = X_test[mask][:500]
        if len(subset) > 0:
            week_agg.append(float(np.mean(subset[:, :, node]) * node_std + node_mean))
        else:
            week_agg.append(0.0)
    return JsonResponse({
        'day_pattern': day_agg,
        'week_pattern': week_agg,
        'node': node,
    })


def api_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': '需要 POST'}, status=405)
    try:
        body = json.loads(request.body)
        node_id = int(body.get('node_id', 0))
        sample_idx = int(body.get('sample_idx', 0))
    except Exception:
        return JsonResponse({'error': '参数错误'}, status=400)
    data, adj, scaler = _get_adj_and_scaler()
    if data is None or adj is None or scaler is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    X_test = data['X_test']
    y_test = data['y_test']
    mean, std = scaler
    num_nodes = adj.shape[0]
    node_id = min(max(0, node_id), num_nodes - 1)
    sample_idx = min(max(0, sample_idx), X_test.shape[0] - 1)
    model_path = getattr(settings, 'STGCN_MODEL_PATH', BASE_DIR / 'best_stgcn_model.pth')
    if not Path(model_path).exists():
        return JsonResponse({'error': 'STGCN 模型文件不存在'}, status=404)
    try:
        import torch
        from STGCN import STGCN
    except ImportError as e:
        return JsonResponse({'error': f'模型加载失败: {e}'}, status=500)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN(
        num_nodes=num_nodes,
        hidden_channels=64,
        num_layers=3,
        pred_len=12,
        kernel_size=3,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    if hasattr(state, 'state_dict'):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.eval()
    adj_t = torch.FloatTensor(adj).to(device)
    x = torch.FloatTensor(X_test[sample_idx:sample_idx + 1]).to(device)
    with torch.no_grad():
        pred = model(x, adj_t)
    pred = pred.cpu().numpy()[0]
    true = y_test[sample_idx]
    # 支持节点级标准化
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        pred_real = pred * std[node_id] + mean[node_id]
        true_real = true * std[node_id] + mean[node_id]
        input_seq = (X_test[sample_idx, :, node_id] * std[node_id] + mean[node_id]).tolist()
    else:
        pred_real = pred * std + mean
        true_real = true * std + mean
        input_seq = (X_test[sample_idx, :, node_id] * std + mean).tolist()
    return JsonResponse({
        'input_seq': input_seq,
        'pred_steps': list(range(1, 13)),
        'pred': pred_real[:, node_id].tolist() if pred_real.ndim > 1 else pred_real.tolist(),
        'true': true_real[:, node_id].tolist() if true_real.ndim > 1 else true_real.tolist(),
        'node_id': node_id,
        'sample_idx': sample_idx,
    })


def api_flow_distribution(request):
    """流量分布直方图：所有节点所有时间的流量分布"""
    data, adj, scaler = _get_adj_and_scaler()
    if data is None or scaler is None:
        return JsonResponse({'error': '数据未加载'}, status=404)
    mean, std = scaler
    X_test = data['X_test']
    n_samples = min(3000, X_test.shape[0])
    # 支持节点级标准化
    if isinstance(mean, np.ndarray) and mean.ndim == 1:
        std_expanded = std.reshape(1, 1, -1)
        mean_expanded = mean.reshape(1, 1, -1)
        all_flow = (X_test[:n_samples] * std_expanded + mean_expanded).flatten()
    else:
        all_flow = (X_test[:n_samples] * std + mean).flatten()
    # 计算直方图的 bin 和 count
    counts, bin_edges = np.histogram(all_flow, bins=50)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    return JsonResponse({
        'bin_centers': bin_centers,
        'counts': counts.tolist(),
        'bin_edges': bin_edges.tolist(),
    })
