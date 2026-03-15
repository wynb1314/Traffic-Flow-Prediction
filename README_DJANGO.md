# PEMS04 交通预测系统 - Web 端

基于 Django 的前端，与现有 STGCN/LSTM 训练与预测逻辑对接。

## 三个页面（顶部导航切换）

1. **数据探索** (`/explore/`)：路网拓扑、时间序列、周模式/日模式分析  
2. **智能预测** (`/predict/`)：节点选择、STGCN 模型预测、预测曲线  
3. **系统仪表盘** (`/dashboard/`)：项目介绍、模型指标（MAE/RMSE/MAPE/R²）

## 运行前准备

- 项目根目录下需有：
  - `processed_data.npz`（数据探索与预测用）
  - `best_stgcn_model.pth`（智能预测用）
  - 可选：`stgcn_results.npz`、`lstm_results.npz`（仪表盘指标）

## 安装与启动

```bash
# 在项目根目录（与 manage.py 同级）
pip install -r requirements.txt
python manage.py runserver
```

浏览器访问：<http://127.0.0.1:8000/>，默认会跳转到「数据探索」页。

## 技术说明

- 数据与模型路径在 `traffic_web/settings.py` 中通过 `PROCESSED_DATA_PATH`、`STGCN_MODEL_PATH` 等配置，默认指向项目根目录。
- 预测接口会从项目根目录动态导入 `STGCN` 并加载 `best_stgcn_model.pth` 做单样本推理。
