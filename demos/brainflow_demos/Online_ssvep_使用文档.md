# Online_ssvep.py 使用文档

## 概述

`Online_ssvep.py` 演示了使用 NeuroScan 设备的在线 SSVEP 闭环流程。核心特点：
- **离线训练 + 在线预测**：启动时先用本地 `.cnt` 数据离线训练，再接收实时数据在线预测并通过 LSL 输出反馈。
- **多进程架构**：Worker 进程负责训练和在线推理，主进程负责采集与数据转发。
- **算法**：FBTDCA（滤波器组 + TDCA），带 CCA 参考信号。

## 环境与依赖

- 依赖：`metabci`、`mne`、`pylsl`、`numpy`、`sklearn`
- 设备：NeuroScan，支持 TCP 连接
- 数据：本地已采集 `.cnt` 训练数据，路径示例 `data\\train\\sub1\\1.cnt`

## 参数速览（脚本主配置）

```python
srate = 1000                         # 采样率
stim_interval = [0.14, 0.68]         # 时间窗口（考虑0.14s视觉延迟）
stim_labels = list(range(1, 21))     # 20个目标
filepath = "data\\train\\sub1"       # 训练数据路径
runs = [1]                           # 使用的 run 列表
pick_chs = ['PZ','PO5','PO3','POZ','PO4','PO6','O1','OZ','O2']  # 通道
device_address = ('192.168.1.30', 4000)  # NeuroScan IP/端口
```

## 运行步骤（最小可行流程）

1. **准备训练数据**：确保 `data\\train\\sub1\\1.cnt` 存在，并包含刺激事件标记。
2. **连设备**：修改 `device_address` 为实际设备 IP/端口。
3. **运行脚本**：
   ```bash
   python Online_ssvep.py
   ```
4. **开始在线**：脚本会自动：
   - 启动 Worker 进程，先离线训练（`pre()`）
   - 建立 LSL 输出通道
   - `start_trans()` 后实时切片数据，`consume()` 在线预测并推送 LSL
5. **结束**：控制台按键退出，脚本会停止传输、断开连接并清理。

## 模块说明

### 数据读取与标签编码
```python
read_data(run_files, chs, interval, labels)
```
- 从 `.cnt` 文件读取数据，按事件切片，返回 `X, y, ch_picks`。
- 标签用 `label_encoder` 重排为 0…C-1。

### 训练函数
```python
train_model(X, y, srate=1000)
```
流程：
1) 重采样到 256 Hz  
2) 去均值、标准化  
3) 构造 5 个频带的滤波器组  
4) 生成 CCA 参考信号（8–16 Hz，步长 0.4 Hz，5 个谐波）  
5) 训练 FBTDCA 模型  

### 在线预测
```python
model_predict(X, srate=1000, model=None)
```
对实时片段重采样/去均值/标准化后，调用 `model.predict`。

### 验证
```python
offline_validation(X, y, srate=1000)
```
留一交叉验证，打印离线准确率（用于快速质检模型）。

### Worker 逻辑
`FeedbackWorker(ProcessWorker)` 三个钩子：
- `pre()`：读取本地 `.cnt` 训练数据 → 交叉验证 → 训练模型 → 建立 LSL 输出 → 等待消费者连接
- `consume(data)`：实时数据 → 选通道 → 预处理 → 预测 → 通过 LSL 输出标签（+1 为了从 1 开始）
- `post()`：留空，可自定义清理

进程在 `ns.up_worker()` 时启动，`pre()` 完成后阻塞等待数据；`ns.start_trans()` 开始推流后自动进入在线预测。

### 主流程
1) 创建 `NeuroScan`，连接并开始采集  
2) 注册 `worker` 与 `marker`（负责事件切片）  
3) `ns.up_worker()` 启动子进程并完成训练  
4) `ns.start_trans()` 开始推流，Worker 自动 `consume()`  
5) 按键退出，依次 `down_worker`、`stop_trans`、`stop_acq`、`close_connection`、`clear`  

## 常见修改点

- **数据路径**：`filepath`、`runs` 对应你的训练数据位置与数量  
- **通道选择**：`pick_chs` 可按实际电极配置调整  
- **时间窗口**：`stim_interval` 影响切片长度，可根据实验设计改为更长/更短  
- **频率列表**：`freqs = np.arange(8, 16, 0.4)` 如需匹配不同刺激频点可修改  
- **设备地址**：`device_address` 替换为实际 IP/端口  

## 运行时输出

- 控制台会打印：
  - “Loding train data successfully”
  - 离线准确率：`Current Model accuracy:xx.xx`
  - 实时预测：`predict_id_paradigm [k]`
- LSL 输出流名称：`meta_feedback`，类型 `Markers`，单通道 int32

## 常见问题

1. **没有预测输出/阻塞**  
   - 确认已调用 `ns.start_trans()`；设备在正常推流；Marker 能产生事件。
2. **离线准确率低**  
   - 检查训练数据质量、通道、时间窗口；必要时增加数据量或调整频带/参考信号。
3. **LSL 无消费者**  
   - 确认刺激呈现端/上位机订阅了 `meta_feedback`；`pre()` 会等待消费者连接。
4. **连接失败**  
   - 检查 `device_address`、网络与防火墙；确认 NeuroScan 在线。

## 快速参考：核心代码位置
- 数据读取与训练：`read_data` / `train_model` / `offline_validation`
- Worker 逻辑：`FeedbackWorker.pre` / `consume`
- 主流程：脚本底部 `__main__` 段

---

**文档版本**：1.0  
**最后更新**：2025年  
**维护者**：MetaBCI 团队

