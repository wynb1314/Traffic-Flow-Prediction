import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""
方案A：基于真实地理位置的交通热力图
适用于PeMSD04数据集（加州高速公路）
"""

def generate_pems04_locations():
    """
    生成PeMSD04的307个检测器的地理位置（模拟）
    实际项目中，如果有真实GPS坐标更好
    这里基于加州湾区（旧金山-圣何塞区域）随机生成
    """
    # 加州湾区中心坐标
    center_lat = 37.7749  # 旧金山
    center_lon = -122.4194
    
    # 在湾区范围内随机分布307个点
    np.random.seed(42)
    num_nodes = 307
    
    # 生成随机偏移（约50km范围）
    lat_offset = np.random.randn(num_nodes) * 0.3
    lon_offset = np.random.randn(num_nodes) * 0.3
    
    locations = []
    for i in range(num_nodes):
        lat = center_lat + lat_offset[i]
        lon = center_lon + lon_offset[i]
        locations.append([lat, lon])
    
    return locations


def create_traffic_heatmap_basic(save_path='traffic_heatmap_basic.html'):
    """
    方法1: 基础热力图（类似你提供的图片）
    使用folium的HeatMap插件
    """
    print("生成基础热力图...")
    
    # 1. 加载数据
    data = np.load('processed_data.npz')
    scaler = (data['mean'].item(), data['std'].item())
    
    # 加载ASTGCN预测结果
    results = np.load('astgcn_results.npz')
    predictions = results['predictions']
    
    # 反标准化
    mean, std = scaler
    predictions_real = predictions * std + mean
    
    # 2. 选择某个时刻的流量（取第100个样本的第一步预测）
    time_idx = 100
    traffic_flow = predictions_real[time_idx, 0, :]  # 307个节点的流量
    
    # 3. 生成地理位置
    locations = generate_pems04_locations()
    
    # 4. 创建地图（以湾区为中心）
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # 5. 准备热力图数据 [lat, lon, weight]
    heat_data = []
    for i in range(len(locations)):
        lat, lon = locations[i]
        weight = float(traffic_flow[i]) / 100  # 归一化权重
        heat_data.append([lat, lon, weight])
    
    # 6. 添加热力图层
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_opacity=0.8,
        radius=15,
        blur=25,
        gradient={
            0.0: 'green',
            0.3: 'yellow', 
            0.5: 'orange',
            0.7: 'red',
            1.0: 'darkred'
        }
    ).add_to(m)
    
    # 7. 添加标题
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>交通流量预测热力图</h4>
    <p><b>时间步:</b> {time_idx} | <b>模型:</b> ASTGCN</p>
    <p><b>颜色:</b> 绿色(低流量) → 红色(高流量)</p>
    </div>
    '''.format(time_idx=time_idx)
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    # 8. 保存
    m.save(save_path)
    print(f"  ✓ 保存: {save_path}")
    print(f"  ✓ 用浏览器打开查看!")
    
    return m


def create_traffic_markers_map(save_path='traffic_markers_map.html'):
    """
    方法2: 节点标记图
    每个检测器显示为一个圆点，大小和颜色表示流量
    """
    print("\n生成节点标记地图...")
    
    # 1. 加载数据
    data = np.load('processed_data.npz')
    scaler = (data['mean'].item(), data['std'].item())
    
    results = np.load('astgcn_results.npz')
    predictions = results['predictions']
    true_values = results['true_values']
    
    mean, std = scaler
    predictions_real = predictions * std + mean
    true_values_real = true_values * std + mean
    
    # 2. 选择时刻
    time_idx = 100
    pred_flow = predictions_real[time_idx, 0, :]
    true_flow = true_values_real[time_idx, 0, :]
    error = np.abs(pred_flow - true_flow)
    
    # 3. 生成位置
    locations = generate_pems04_locations()
    
    # 4. 创建地图
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=10,
        tiles='CartoDB positron'
    )
    
    # 5. 颜色映射
    norm = mcolors.Normalize(vmin=pred_flow.min(), vmax=pred_flow.max())
    cmap = plt.cm.get_cmap('YlOrRd')
    
    # 6. 添加每个节点
    for i in range(len(locations)):
        lat, lon = locations[i]
        flow = pred_flow[i]
        true_f = true_flow[i]
        err = error[i]
        
        # 获取颜色
        rgba = cmap(norm(flow))
        color = mcolors.rgb2hex(rgba[:3])
        
        # 创建弹窗信息
        popup_text = f"""
        <b>节点 {i}</b><br>
        预测流量: {flow:.1f}<br>
        真实流量: {true_f:.1f}<br>
        误差: {err:.1f}
        """
        
        # 添加圆形标记
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_text, max_width=200),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # 7. 添加图例
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4 style="margin-top:0">流量图例</h4>
    <p><span style="background-color: #ffffb2; padding: 5px;">●</span> 低流量</p>
    <p><span style="background-color: #feb24c; padding: 5px;">●</span> 中流量</p>
    <p><span style="background-color: #f03b20; padding: 5px;">●</span> 高流量</p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 8. 保存
    m.save(save_path)
    print(f"  ✓ 保存: {save_path}")
    
    return m


def create_comparison_map(save_path='traffic_comparison_map.html'):
    """
    方法3: 预测 vs 真实对比地图
    左右分屏显示预测和真实流量
    """
    print("\n生成对比地图...")
    
    # 加载数据
    data = np.load('processed_data.npz')
    scaler = (data['mean'].item(), data['std'].item())
    
    results = np.load('astgcn_results.npz')
    predictions = results['predictions']
    true_values = results['true_values']
    
    mean, std = scaler
    predictions_real = predictions * std + mean
    true_values_real = true_values * std + mean
    
    time_idx = 100
    pred_flow = predictions_real[time_idx, 0, :]
    true_flow = true_values_real[time_idx, 0, :]
    
    locations = generate_pems04_locations()
    
    # 创建基础地图
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=10
    )
    
    # 预测流量热力图
    pred_heat_data = [[loc[0], loc[1], float(pred_flow[i])/100] 
                      for i, loc in enumerate(locations)]
    
    # 真实流量热力图（用不同颜色）
    true_heat_data = [[loc[0], loc[1], float(true_flow[i])/100] 
                      for i, loc in enumerate(locations)]
    
    # 添加两个图层组
    pred_layer = folium.FeatureGroup(name='预测流量')
    true_layer = folium.FeatureGroup(name='真实流量')
    
    # 添加预测热力图
    HeatMap(
        pred_heat_data,
        min_opacity=0.3,
        radius=15,
        gradient={0.0: 'blue', 0.5: 'cyan', 1.0: 'green'}
    ).add_to(pred_layer)
    
    # 添加真实热力图
    HeatMap(
        true_heat_data,
        min_opacity=0.3,
        radius=15,
        gradient={0.0: 'yellow', 0.5: 'orange', 1.0: 'red'}
    ).add_to(true_layer)
    
    pred_layer.add_to(m)
    true_layer.add_to(m)
    
    # 添加图层控制
    folium.LayerControl().add_to(m)
    
    m.save(save_path)
    print(f"  ✓ 保存: {save_path}")
    
    return m


def main():
    """生成所有地图可视化"""
    print("="*60)
    print("生成真实路网热力图")
    print("="*60)
    
    # 方法1: 基础热力图
    create_traffic_heatmap_basic()
    
    # 方法2: 节点标记图
    create_traffic_markers_map()
    
    # 方法3: 对比地图
    create_comparison_map()
    
    print("\n" + "="*60)
    print("完成! 生成了3个HTML地图文件:")
    print("  1. traffic_heatmap_basic.html - 基础热力图")
    print("  2. traffic_markers_map.html - 节点标记图（可点击查看详情）")
    print("  3. traffic_comparison_map.html - 预测vs真实对比图")
    print("\n用浏览器打开查看，支持交互式缩放和点击！")
    print("="*60)


if __name__ == "__main__":
    main()