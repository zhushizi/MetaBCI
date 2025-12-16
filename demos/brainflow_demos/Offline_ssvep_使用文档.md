# SSVEP 离线分析简明使用文档

适用脚本：`demos/brainflow_demos/Offline_ssvep.py`

## 核心流程
1) 数据加载：使用 `MetaBCIData` + `SSVEP` 范式，路径形如 `data/ssvep/sub1/1.cnt`，需带事件标记。  
2) 预处理：可选 `raw_hook`（示例为 7–55 Hz 滤波），范式切段区间默认 `[(0.14, 1.14)]`。  
3) 训练与验证：`offline_validation` 采用留一交叉验证；`train_model` 用 FBTDCA（滤波器组 + CCA 参考信号）。  
4) 特征分析（可选）：提供时域、频域、时频域三类分析函数（默认只调用前两类，时频域示例注释）。  

## 快速上手
```bash
python Offline_ssvep.py
```
若需调整，在 `__main__` 里修改：
- `srate = 1000`                     # 原始采样率
- `stim_interval = [(0.14, 1.14)]`   # 切片时间窗（含0.14s视觉延迟）
- `subjects = [1]`                   # 被试列表
- `pick_chs = ['PZ','PO5','PO3','POZ','PO4','PO6','O1','OZ','O2']`

## 训练细节（train_model）
- 重采样到 256 Hz；去均值、标准化。  
- 滤波器组：5 个频带（wp: [6,14,22,30,38], ws: [4,12,20,28,36]），权重 `n^-1.25 + 0.25`。  
- 参考信号：8–16 Hz，步长 0.4 Hz，5 个谐波 (`generate_cca_references`)。  
- 模型：`FBTDCA(filterbank, padding_len=3, n_components=4, filterweights=...)`。  

## 输出
- 控制台打印离线准确率：`Current Model accuracy: xx.xx`  
- 生成的图（默认）：时域模板/叠加、频域 PSD（基频与谐波）。  
- 时频分析示例保留为注释，可按需解开。  

## 常用改动
- 改时间窗：调整 `stim_interval`。短窗响应快，长窗信噪比高。  
- 改通道：修改 `pick_chs` 以匹配你的帽型/脑区。  
- 改频率列表：在 `train_model` 中修改 `freqs = np.arange(8, 16, 0.4)` 以适配不同刺激频点。  

## 常见问题速查
1) **找不到数据**：确认路径 `data/ssvep/sub{id}/{run}.cnt` 存在且带事件标记。  
2) **准确率低**：检查时间窗/通道/数据质量，或增大样本量，调整滤波器组和参考信号频点。  
3) **图不显示**：确保安装 matplotlib，必要时在非交互环境添加 `plt.show()`。  

## 参考
- FBTDCA（FilterBank Task Decomposition Component Analysis）  
- SSVEP（Steady-State Visual Evoked Potential）

文档版本：1.0（简化版） / 最后更新：2025年

