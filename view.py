import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

# 创建输出目录
os.makedirs('visualization_results', exist_ok=True)


class TrafficVisualizer:
    """交通流量预测可视化工具"""
    
    def __init__(self):
        self.colors = {
            'LSTM': '#FF6B6B',
            'STGCN': '#4ECDC4', 
            'ASTGCN': '#45B7D1'
        }
    
    def load_results(self):
        """加载所有模型的结果"""
        print("加载模型结果...")
        
        # 加载LSTM结果
        lstm_data = np.load('lstm_results.npz')
        self.lstm_results = {
            'predictions': lstm_data['predictions'],
            'true_values': lstm_data['true_values'],
            'train_losses': lstm_data['train_losses'],
            'val_losses': lstm_data['val_losses'],
            'mae': lstm_data['mae'].item(),
            'rmse': lstm_data['rmse'].item(),
            'mape': lstm_data['mape'].item(),
            'r2': lstm_data['r2'].item()
        }
        
        # 加载STGCN结果
        stgcn_data = np.load('stgcn_results.npz')
        self.stgcn_results = {
            'predictions': stgcn_data['predictions'],
            'true_values': stgcn_data['true_values'],
            'train_losses': stgcn_data['train_losses'],
            'val_losses': stgcn_data['val_losses'],
            'mae': stgcn_data['mae'].item(),
            'rmse': stgcn_data['rmse'].item(),
            'mape': stgcn_data['mape'].item(),
            'r2': stgcn_data['r2'].item()
        }
        
        # 加载ASTGCN结果
        astgcn_data = np.load('astgcn_results.npz')
        self.astgcn_results = {
            'predictions': astgcn_data['predictions'],
            'true_values': astgcn_data['true_values'],
            'train_losses': astgcn_data['train_losses'],
            'val_losses': astgcn_data['val_losses'],
            'mae': astgcn_data['mae'].item(),
            'rmse': astgcn_data['rmse'].item(),
            'mape': astgcn_data['mape'].item(),
            'r2': astgcn_data['r2'].item()
        }
        
        # 加载标准化参数
        data = np.load('processed_data.npz')
        self.scaler = (data['mean'].item(), data['std'].item())
        
        print("结果加载完成!")
    
    def plot_metrics_comparison(self):
        """1. 模型性能对比柱状图"""
        print("\n生成性能对比图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = ['LSTM', 'STGCN', 'ASTGCN']
        
        # MAE对比
        mae_values = [self.lstm_results['mae'], 
                     self.stgcn_results['mae'], 
                     self.astgcn_results['mae']]
        axes[0, 0].bar(models, mae_values, color=[self.colors[m] for m in models], alpha=0.8)
        axes[0, 0].set_ylabel('MAE', fontsize=12)
        axes[0, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(mae_values):
            axes[0, 0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')
        
        # RMSE对比
        rmse_values = [self.lstm_results['rmse'], 
                      self.stgcn_results['rmse'], 
                      self.astgcn_results['rmse']]
        axes[0, 1].bar(models, rmse_values, color=[self.colors[m] for m in models], alpha=0.8)
        axes[0, 1].set_ylabel('RMSE', fontsize=12)
        axes[0, 1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(rmse_values):
            axes[0, 1].text(i, v + 0.8, f'{v:.2f}', ha='center', fontweight='bold')
        
        # MAPE对比
        mape_values = [self.lstm_results['mape'], 
                      self.stgcn_results['mape'], 
                      self.astgcn_results['mape']]
        axes[1, 0].bar(models, mape_values, color=[self.colors[m] for m in models], alpha=0.8)
        axes[1, 0].set_ylabel('MAPE (%)', fontsize=12)
        axes[1, 0].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(mape_values):
            axes[1, 0].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')
        
        # R²对比
        r2_values = [self.lstm_results['r2'], 
                    self.stgcn_results['r2'], 
                    self.astgcn_results['r2']]
        axes[1, 1].bar(models, r2_values, color=[self.colors[m] for m in models], alpha=0.8)
        axes[1, 1].set_ylabel('R² Score', fontsize=12)
        axes[1, 1].set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(r2_values):
            axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualization_results/1_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 1_metrics_comparison.png")
        plt.close()
    
    def plot_training_curves(self):
        """2. 训练损失曲线对比"""
        print("\n生成训练曲线图...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
        
        # 训练损失
        ax1.plot(self.lstm_results['train_losses'], label='LSTM', 
                color=self.colors['LSTM'], linewidth=2, alpha=0.8)
        ax1.plot(self.stgcn_results['train_losses'], label='STGCN', 
                color=self.colors['STGCN'], linewidth=2, alpha=0.8)
        ax1.plot(self.astgcn_results['train_losses'], label='ASTGCN', 
                color=self.colors['ASTGCN'], linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 验证损失
        ax2.plot(self.lstm_results['val_losses'], label='LSTM', 
                color=self.colors['LSTM'], linewidth=2, alpha=0.8)
        ax2.plot(self.stgcn_results['val_losses'], label='STGCN', 
                color=self.colors['STGCN'], linewidth=2, alpha=0.8)
        ax2.plot(self.astgcn_results['val_losses'], label='ASTGCN', 
                color=self.colors['ASTGCN'], linewidth=2, alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualization_results/2_training_curves.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 2_training_curves.png")
        plt.close()
    
    def plot_prediction_samples(self):
        """3. 预测样本对比（三个模型）"""
        print("\n生成预测样本对比图...")
        
        mean, std = self.scaler
        
        # 反标准化
        lstm_pred = self.lstm_results['predictions'] * std + mean
        stgcn_pred = self.stgcn_results['predictions'] * std + mean
        astgcn_pred = self.astgcn_results['predictions'] * std + mean
        true_val = self.lstm_results['true_values'] * std + mean
        
        # 选择节点0，展示前1200个时间步的第一步预测
        node_idx = 0
        plot_len = 1200
        
        lstm_flat = lstm_pred[:plot_len, 0, node_idx]
        stgcn_flat = stgcn_pred[:plot_len, 0, node_idx]
        astgcn_flat = astgcn_pred[:plot_len, 0, node_idx]
        true_flat = true_val[:plot_len, 0, node_idx]
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 1, figure=fig, hspace=0.3)
        
        # LSTM预测
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(true_flat, label='True Value', color='blue', linewidth=1.5, alpha=0.7)
        ax1.plot(lstm_flat, label='LSTM Prediction', color=self.colors['LSTM'], 
                linewidth=1.2, linestyle='--', alpha=0.8)
        ax1.set_ylabel('Traffic Flow', fontsize=11)
        ax1.set_title('LSTM Prediction vs. True Value', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # STGCN预测
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(true_flat, label='True Value', color='blue', linewidth=1.5, alpha=0.7)
        ax2.plot(stgcn_flat, label='STGCN Prediction', color=self.colors['STGCN'], 
                linewidth=1.2, linestyle='--', alpha=0.8)
        ax2.set_ylabel('Traffic Flow', fontsize=11)
        ax2.set_title('STGCN Prediction vs. True Value', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ASTGCN预测
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(true_flat, label='True Value', color='blue', linewidth=1.5, alpha=0.7)
        ax3.plot(astgcn_flat, label='ASTGCN Prediction', color=self.colors['ASTGCN'], 
                linewidth=1.2, linestyle='--', alpha=0.8)
        ax3.set_xlabel('Time Steps', fontsize=11)
        ax3.set_ylabel('Traffic Flow', fontsize=11)
        ax3.set_title('ASTGCN Prediction vs. True Value', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.savefig('visualization_results/3_prediction_samples.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 3_prediction_samples.png")
        plt.close()
    
    def plot_error_distribution(self):
        """4. 预测误差分布"""
        print("\n生成误差分布图...")
        
        mean, std = self.scaler
        
        # 计算误差
        lstm_error = (self.lstm_results['predictions'] - self.lstm_results['true_values']).flatten() * std
        stgcn_error = (self.stgcn_results['predictions'] - self.stgcn_results['true_values']).flatten() * std
        astgcn_error = (self.astgcn_results['predictions'] - self.astgcn_results['true_values']).flatten() * std
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Prediction Error Distribution', fontsize=16, fontweight='bold')
        
        # LSTM误差分布
        axes[0].hist(lstm_error, bins=100, color=self.colors['LSTM'], alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Error', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'LSTM (std={np.std(lstm_error):.2f})', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # STGCN误差分布
        axes[1].hist(stgcn_error, bins=100, color=self.colors['STGCN'], alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Error', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title(f'STGCN (std={np.std(stgcn_error):.2f})', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # ASTGCN误差分布
        axes[2].hist(astgcn_error, bins=100, color=self.colors['ASTGCN'], alpha=0.7, edgecolor='black')
        axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Error', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title(f'ASTGCN (std={np.std(astgcn_error):.2f})', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualization_results/4_error_distribution.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 4_error_distribution.png")
        plt.close()
    
    def plot_multi_step_error(self):
        """5. 多步预测误差分析"""
        print("\n生成多步预测误差图...")
        
        mean, std = self.scaler
        
        # 计算每一步的MAE
        lstm_mae_per_step = []
        stgcn_mae_per_step = []
        astgcn_mae_per_step = []
        
        for step in range(12):
            lstm_mae = np.mean(np.abs(
                (self.lstm_results['predictions'][:, step, :] - 
                 self.lstm_results['true_values'][:, step, :]) * std
            ))
            stgcn_mae = np.mean(np.abs(
                (self.stgcn_results['predictions'][:, step, :] - 
                 self.stgcn_results['true_values'][:, step, :]) * std
            ))
            astgcn_mae = np.mean(np.abs(
                (self.astgcn_results['predictions'][:, step, :] - 
                 self.astgcn_results['true_values'][:, step, :]) * std
            ))
            
            lstm_mae_per_step.append(lstm_mae)
            stgcn_mae_per_step.append(stgcn_mae)
            astgcn_mae_per_step.append(astgcn_mae)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(1, 13)
        ax.plot(steps, lstm_mae_per_step, marker='o', label='LSTM', 
               color=self.colors['LSTM'], linewidth=2.5, markersize=8)
        ax.plot(steps, stgcn_mae_per_step, marker='s', label='STGCN', 
               color=self.colors['STGCN'], linewidth=2.5, markersize=8)
        ax.plot(steps, astgcn_mae_per_step, marker='^', label='ASTGCN', 
               color=self.colors['ASTGCN'], linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('Multi-Step Prediction Error Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps)
        
        plt.tight_layout()
        plt.savefig('visualization_results/5_multi_step_error.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 5_multi_step_error.png")
        plt.close()
    
    def plot_spatial_heatmap(self):
        """6. 空间流量热力图"""
        print("\n生成空间流量热力图...")
        
        mean, std = self.scaler
        
        # 选择某个时刻的所有节点流量
        time_idx = 100
        true_val = self.lstm_results['true_values'][time_idx, 0, :] * std + mean
        astgcn_pred = self.astgcn_results['predictions'][time_idx, 0, :] * std + mean
        
        # 计算误差
        error = np.abs(astgcn_pred - true_val)
        
        # 重塑为接近正方形（307节点 -> 18x18矩阵，填充到324）
        n = 18
        true_reshaped = np.zeros((n, n))
        pred_reshaped = np.zeros((n, n))
        error_reshaped = np.zeros((n, n))
        
        for i in range(min(307, n*n)):
            row = i // n
            col = i % n
            true_reshaped[row, col] = true_val[i]
            pred_reshaped[row, col] = astgcn_pred[i]
            error_reshaped[row, col] = error[i]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Spatial Traffic Flow Heatmap (Time Step {time_idx})', 
                    fontsize=16, fontweight='bold')
        
        # 真实值热力图
        im1 = axes[0].imshow(true_reshaped, cmap='YlOrRd', aspect='auto')
        axes[0].set_title('True Traffic Flow', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Node Index (X)', fontsize=11)
        axes[0].set_ylabel('Node Index (Y)', fontsize=11)
        plt.colorbar(im1, ax=axes[0], label='Flow')
        
        # 预测值热力图
        im2 = axes[1].imshow(pred_reshaped, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('ASTGCN Prediction', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Node Index (X)', fontsize=11)
        axes[1].set_ylabel('Node Index (Y)', fontsize=11)
        plt.colorbar(im2, ax=axes[1], label='Flow')
        
        # 误差热力图
        im3 = axes[2].imshow(error_reshaped, cmap='Reds', aspect='auto')
        axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Node Index (X)', fontsize=11)
        axes[2].set_ylabel('Node Index (Y)', fontsize=11)
        plt.colorbar(im3, ax=axes[2], label='Error')
        
        plt.tight_layout()
        plt.savefig('visualization_results/6_spatial_heatmap.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 6_spatial_heatmap.png")
        plt.close()
    
    def plot_node_performance(self):
        """7. 不同节点的预测性能"""
        print("\n生成节点性能对比图...")
        
        mean, std = self.scaler
        
        # 计算每个节点的平均MAE
        num_nodes = 307
        lstm_mae_per_node = []
        stgcn_mae_per_node = []
        astgcn_mae_per_node = []
        
        for node in range(num_nodes):
            lstm_mae = np.mean(np.abs(
                (self.lstm_results['predictions'][:, :, node] - 
                 self.lstm_results['true_values'][:, :, node]) * std
            ))
            stgcn_mae = np.mean(np.abs(
                (self.stgcn_results['predictions'][:, :, node] - 
                 self.stgcn_results['true_values'][:, :, node]) * std
            ))
            astgcn_mae = np.mean(np.abs(
                (self.astgcn_results['predictions'][:, :, node] - 
                 self.astgcn_results['true_values'][:, :, node]) * std
            ))
            
            lstm_mae_per_node.append(lstm_mae)
            stgcn_mae_per_node.append(stgcn_mae)
            astgcn_mae_per_node.append(astgcn_mae)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        nodes = range(num_nodes)
        ax.plot(nodes, lstm_mae_per_node, label='LSTM', 
               color=self.colors['LSTM'], linewidth=1, alpha=0.7)
        ax.plot(nodes, stgcn_mae_per_node, label='STGCN', 
               color=self.colors['STGCN'], linewidth=1, alpha=0.7)
        ax.plot(nodes, astgcn_mae_per_node, label='ASTGCN', 
               color=self.colors['ASTGCN'], linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Node Index', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('Prediction Performance Across All Nodes', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualization_results/7_node_performance.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 7_node_performance.png")
        plt.close()
    
    def generate_summary_table(self):
        """8. 生成性能对比表格"""
        print("\n生成性能对比表格...")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        
        # 表格数据
        table_data = [
            ['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²'],
            ['LSTM', 
             f"{self.lstm_results['mae']:.4f}",
             f"{self.lstm_results['rmse']:.4f}",
             f"{self.lstm_results['mape']:.2f}",
             f"{self.lstm_results['r2']:.4f}"],
            ['STGCN', 
             f"{self.stgcn_results['mae']:.4f}",
             f"{self.stgcn_results['rmse']:.4f}",
             f"{self.stgcn_results['mape']:.2f}",
             f"{self.stgcn_results['r2']:.4f}"],
            ['ASTGCN', 
             f"{self.astgcn_results['mae']:.4f}",
             f"{self.astgcn_results['rmse']:.4f}",
             f"{self.astgcn_results['mape']:.2f}",
             f"{self.astgcn_results['r2']:.4f}"]
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for i in range(5):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置行颜色
        colors = ['#FFE5E5', '#E5F5F5', '#E5F0FF']
        for i in range(1, 4):
            for j in range(5):
                table[(i, j)].set_facecolor(colors[i-1])
        
        plt.title('Model Performance Comparison Table', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig('visualization_results/8_performance_table.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存: 8_performance_table.png")
        plt.close()
    
    def generate_all(self):
        """生成所有可视化"""
        print("\n" + "="*60)
        print("开始生成所有可视化图表")
        print("="*60)
        
        self.load_results()
        self.plot_metrics_comparison()
        self.plot_training_curves()
        self.plot_prediction_samples()
        self.plot_error_distribution()
        self.plot_multi_step_error()
        self.plot_spatial_heatmap()
        self.plot_node_performance()
        self.generate_summary_table()
        
        print("\n" + "="*60)
        print("所有可视化完成! 共生成8张图表")
        print("保存位置: visualization_results/")
        print("="*60)
        
        # 生成图表列表
        print("\n生成的图表:")
        print("  1. 1_metrics_comparison.png - 性能指标对比")
        print("  2. 2_training_curves.png - 训练曲线")
        print("  3. 3_prediction_samples.png - 预测样本对比")
        print("  4. 4_error_distribution.png - 误差分布")
        print("  5. 5_multi_step_error.png - 多步预测误差")
        print("  6. 6_spatial_heatmap.png - 空间流量热力图")
        print("  7. 7_node_performance.png - 节点性能对比")
        print("  8. 8_performance_table.png - 性能对比表格")


if __name__ == "__main__":
    visualizer = TrafficVisualizer()
    visualizer.generate_all()