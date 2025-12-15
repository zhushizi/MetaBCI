# -*- coding: utf-8 -*-
"""
FBECCA with Dynamic Window
动态窗口FBECCA算法

参考论文：
Chen Yang et al., A Dynamic Window Recognition Algorithm for SSVEP-Based 
Brain-Computer Interfaces Using A Spatio-Temporal Equalizer, 
International Journal of Neural Systems, doi: 10.1142/S0129065718500284

核心思想：
- Dynamic Window: 从短窗口开始，逐步增加窗口长度
- 根据置信度（最大特征值/次大特征值）自适应决定何时停止
- 提高ITR（信息传输率）：简单样本提前停止，困难样本延长窗口
- 窗口长度是由“最短且已过置信度阈值”的窗口决定的
"""
import numpy as np
import time
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from metabci.brainda.algorithms.decomposition import FBECCA
from metabci.brainda.algorithms.decomposition.base import generate_filterbank, generate_cca_references
from metabci.brainda.utils.performance import Performance


# 滤波器组参数
wp = [(5, 90), (14, 90), (22, 90), (30, 90), (38, 90)]
ws = [(3, 92), (12, 92), (20, 92), (28, 92), (36, 92)]

filterbank = generate_filterbank(wp, ws, srate=250, order=15, rp=0.5)

dataset = Wang2016()

events = dataset.events.keys()
freq_list = [dataset.get_freq(event) for event in events]

# 动态窗口参数（参考STE-DW论文）
delay = 0.14  # 视觉延迟（秒）
max_duration = 1.0  # 最大窗口长度（秒）
min_duration = 0.4  # 最小窗口长度（秒），确保足够的数据点进行滤波
step_duration = 0.2  # 每次增加的步长（秒）- 增大步长以减少训练时间（0.4, 0.6, 0.8, 1.0）- 4个窗口

# 动态窗口终止参数
confidence_threshold = 1.5  # 置信度阈值（最大特征值/次大特征值）

# 提取更长的数据（用于动态窗口）
paradigm = SSVEP(
    channels=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
    intervals=[(delay, delay + max_duration)],  # 提取1秒数据用于动态窗口
    srate=250
)

# add 5-90Hz bandpass filter in raw hook
def raw_hook(raw, caches):
    raw.filter(5, 90, l_trans_bandwidth=2, h_trans_bandwidth=5,
               phase='zero-double')
    caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    return raw, caches

def epochs_hook(epochs, caches):
    caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
    return epochs, caches

def data_hook(X, y, meta, caches):
    caches['data_stage'] = caches.get('data_stage', -1) + 1
    return X, y, meta, caches

paradigm.register_raw_hook(raw_hook)
paradigm.register_epochs_hook(epochs_hook)
paradigm.register_data_hook(data_hook)

# 提取数据（包含更长的窗口）
X, y, meta = paradigm.get_data(
    dataset,
    subjects=[1],
    return_concat=True,
    n_jobs=None,
    verbose=False)

print(f"提取的数据形状: {X.shape}")  # 应该是 (n_trials, n_channels, 250个时间点，对应1秒)
print(f"总样本数: {len(X)}")

# 6-fold cross validation
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta, kfold=kfold)

# 滤波器权重
filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]


class DynamicWindowFBECCA:
    """
    动态窗口FBECCA分类器
    
    参考STE-DW论文的动态窗口思想：
    从短窗口开始，逐步增加窗口长度，根据置信度决定何时停止
    """
    
    def __init__(self, filterbank, filterweights, freq_list, srate=250,
                 min_duration=0.4, max_duration=1.0, step=0.2, 
                 n_components=1, confidence_threshold=1.5):
        self.filterbank = filterbank
        self.filterweights = filterweights
        self.freq_list = freq_list
        self.srate = srate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.step = step
        self.n_components = n_components
        self.confidence_threshold = confidence_threshold  # 置信度阈值
        
        # 为不同窗口长度训练多个模型
        self.models = {}
        self.Yf_dict = {}
        
    def fit(self, X, y):
        """
        训练多个不同窗口长度的模型
        
        为不同时间窗口训练多个分类器，支持动态窗口选择
        """
        print("训练动态窗口模型...")
        
        # 生成所有可能的窗口长度
        durations = np.round(np.arange(self.min_duration, 
                                     self.max_duration + self.step, 
                                     self.step), 2)
        
        # 计算最小所需的数据点数
        min_samples_required = 100
        
        for duration in durations:
            # 计算该窗口长度对应的采样点数
            n_samples = int(self.srate * duration)
            
            # 检查数据长度是否足够
            if n_samples < min_samples_required:
                print(f"  跳过窗口长度 {duration:.2f}s (数据点{n_samples} < 最小需求{min_samples_required})")
                continue
            
            # 提取对应长度的数据
            X_window = X[..., :n_samples]
            
            # 生成对应长度的参考信号
            Yf = generate_cca_references(
                self.freq_list, 
                srate=self.srate, 
                T=duration,
                n_harmonics=5
            )
            self.Yf_dict[duration] = Yf
            
            # 训练模型
            # 对于小数据集，n_jobs=1可能更快（避免并行开销）
            estimator = FBECCA(
                filterbank=self.filterbank,
                n_components=self.n_components,
                filterweights=np.array(self.filterweights),
                n_jobs=1  # 改为1可减少并行开销，对小数据集更快
            )
            
            # 去均值
            X_window = X_window - np.mean(X_window, axis=-1, keepdims=True)
            
            estimator.fit(X=X_window, y=y, Yf=Yf)
            self.models[duration] = estimator
            
            print(f"  完成窗口长度 {duration:.2f}s 的训练 (数据点: {n_samples})")
        print(f"  共训练了 {len(self.models)} 个窗口长度的模型")
    
    def _check_confidence(self, features):
        """
        检查置信度，决定是否满足终止条件
        
        使用简单的比值方法：最大特征值 / 次大特征值
        
        Parameters
        ----------
        features : ndarray
            特征向量，形状为 (n_classes,)
            
        Returns
        -------
        decision : bool
            是否满足终止条件（True表示可以停止）
        confidence : float
            置信度值（最大特征值/次大特征值）
        predicted_label : int
            预测的类别
        """
        if len(features) > 1:
            # 使用argpartition找到最大的两个索引（O(n)而不是O(n log n)）
            top2_idx = np.argpartition(features, -2)[-2:]
            top2_vals = features[top2_idx]
            
            # 找到最大值和次大值
            if top2_vals[0] > top2_vals[1]:
                max_val, second_max_val = top2_vals[0], top2_vals[1]
                max_idx = top2_idx[0]
            else:
                max_val, second_max_val = top2_vals[1], top2_vals[0]
                max_idx = top2_idx[1]
            
            confidence = max_val / (second_max_val + 1e-10)
            decision = confidence > self.confidence_threshold
            return decision, confidence, max_idx
        else:
            return True, features[0], np.argmax(features)
    
    def predict_with_dynamic_window(self, X_single):
        """
        对单个样本进行动态窗口预测
        
        从短窗口开始，逐步增加窗口长度，根据置信度决定何时停止
        
        Parameters
        ----------
        X_single : ndarray
            单个样本，形状为 (n_channels, n_timepoints)
            
        Returns
        -------
        label : int
            预测的类别
        duration : float
            使用的窗口长度
        confidence : float
            预测的置信度
        """
        # 只使用已训练的窗口长度
        durations = sorted(self.models.keys())
        
        # 预计算去均值后的数据（避免重复计算）
        X_centered = X_single - np.mean(X_single, axis=-1, keepdims=True)
        
        for duration in durations:
            n_samples = int(self.srate * duration)
            
            # 优化：使用view而不是copy，只对需要的部分去均值
            # 直接使用切片，避免copy开销
            X_window = X_centered[:, :n_samples]
            # 对当前窗口重新去均值（更准确）
            X_window = X_window - np.mean(X_window, axis=-1, keepdims=True)
            
            # 获取模型
            model = self.models[duration]
            
            # 获取特征（相关性系数）- 直接传入2D数组，避免newaxis
            transform_start = time.time()
            features = model.transform(X_window[np.newaxis, ...])
            transform_time = time.time() - transform_start
            features = features[0]  # 取第一个样本的特征
            
            # 检查置信度
            decision, confidence, pred_label = self._check_confidence(features)
            
            # 终止条件：置信度满足阈值或达到最大窗口
            if decision or duration >= self.max_duration:
                return pred_label, duration, confidence
        
        # 如果所有窗口都不满足条件，返回最大窗口的结果
        duration = durations[-1]
        n_samples = int(self.srate * duration)
        X_window = X_centered[:, :n_samples]
        X_window = X_window - np.mean(X_window, axis=-1, keepdims=True)
        model = self.models[duration]
        
        features = model.transform(X_window[np.newaxis, ...])
        features = features[0]
        _, confidence, pred_label = self._check_confidence(features)
        
        return pred_label, duration, confidence
    
    def predict(self, X):
        """
        批量预测 - 优化版本 (Batch Processing)
        
        优化策略：
        不再逐个样本进行动态窗口搜索，而是先对所有样本计算所有窗口长度的特征。
        这样可以利用 transform 的批量处理优势，避免单样本处理的开销。
        
        Parameters
        ----------
        X : ndarray
            数据，形状为 (n_trials, n_channels, n_timepoints)
            
        Returns
        -------
        labels : ndarray
            预测的类别
        durations : ndarray
            每个样本使用的窗口长度
        confidences : ndarray
            每个样本的置信度
        """
        n_trials = X.shape[0]
        # 预分配结果数组
        labels = np.zeros(n_trials, dtype=int)
        durations = np.zeros(n_trials, dtype=float)
        confidences = np.zeros(n_trials, dtype=float)
        
        # 记录每个样本是否已经完成分类
        finished_mask = np.zeros(n_trials, dtype=bool)
        
        # 预计算去均值（可选，transform内部通常也会做）
        X_centered = X - np.mean(X, axis=-1, keepdims=True)
        
        # 获取所有训练过的窗口长度并排序
        trained_durations = sorted(self.models.keys())
        
        print("  开始批量特征提取...")
        
        # 遍历所有窗口长度
        for duration in trained_durations:
            # 如果所有样本都已完成，提前退出
            if np.all(finished_mask):
                break
                
            n_samples = int(self.srate * duration)
            
            # 提取当前窗口长度的数据（所有样本）
            # 优化：使用切片
            X_window = X_centered[..., :n_samples]
            X_window = X_window - np.mean(X_window, axis=-1, keepdims=True)
            
            # 获取对应长度的模型
            model = self.models[duration]
            
            # 批量计算特征！这是速度提升的关键
            # transform 会并行处理 n_trials 个样本
            # 注意：只对未完成的样本计算可能更省，但为了利用批量优势，
            # 且FilterBank实现可能不支持部分索引的高效处理，
            # 我们对所有样本计算，或者只计算未完成的样本（需要重组数组）
            
            # 策略：只计算未完成的样本
            active_indices = np.where(~finished_mask)[0]
            if len(active_indices) == 0:
                break
                
            features_batch = model.transform(X_window[active_indices])
            
            # 逐个检查置信度
            for i, idx in enumerate(active_indices):
                features = features_batch[i]
                decision, confidence, pred_label = self._check_confidence(features)
                
                # 如果满足条件，或者已经是最大窗口
                if decision or duration >= self.max_duration:
                    labels[idx] = pred_label
                    durations[idx] = duration
                    confidences[idx] = confidence
                    finished_mask[idx] = True
        
        return labels, durations, confidences


# 创建动态窗口分类器
# 优化说明：
# - step_duration=0.2: 窗口数量为4个（0.4, 0.6, 0.8, 1.0s）
# - n_jobs=1: 对小数据集，单线程可能更快（避免并行开销）
estimator = DynamicWindowFBECCA(
    filterbank=filterbank,
    filterweights=filterweights,
    freq_list=freq_list,
    srate=250,
    min_duration=min_duration,
    max_duration=max_duration,
    step=step_duration,
    n_components=1,
    confidence_threshold=confidence_threshold
)

# 交叉验证
accs = []
all_durations = []
all_confidences = []
all_y_true = []
all_y_pred = []
train_times = []
predict_times = []

for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    
    # 打印样本数信息
    print(f"\nFold {k+1} 数据划分:")
    print(f"  训练样本数: {len(train_ind)}")
    print(f"  测试样本数: {len(test_ind)}")
    
    # 训练模型
    print(f"训练 Fold {k+1}...")
    train_start = time.time()
    estimator.fit(X=X[train_ind], y=y[train_ind])
    train_time = time.time() - train_start
    train_times.append(train_time)
    print(f"  训练耗时: {train_time:.2f} 秒")
    
    # 预测
    print(f"预测 Fold {k+1}...")
    predict_start = time.time()
    p_labels, durations, confidences = estimator.predict(X[test_ind])
    predict_time = time.time() - predict_start
    predict_times.append(predict_time)
    print(f"  预测耗时: {predict_time:.2f} 秒 (平均每个样本: {predict_time/len(test_ind)*1000:.2f} 毫秒)")
    
    # 计算准确率
    acc = np.mean(p_labels == y[test_ind])
    accs.append(acc)
    all_durations.extend(durations.tolist())
    all_confidences.extend(confidences.tolist())
    all_y_true.extend(y[test_ind].tolist())
    all_y_pred.extend(p_labels.tolist())
    
    print(f"Fold {k+1}: 准确率={acc:.4f}, 平均窗口长度={np.mean(durations):.2f}s, "
          f"平均置信度={np.mean(confidences):.4f}")

print("\n" + "="*70)
print("动态窗口FBECCA结果:")
print("="*70)
print(f"总样本数: {len(X)}")
print(f"测试样本总数: {len(all_y_true)}")
print(f"平均准确率: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"平均窗口长度: {np.mean(all_durations):.2f}s ± {np.std(all_durations):.2f}s")
print(f"平均置信度: {np.mean(all_confidences):.4f} ± {np.std(all_confidences):.4f}")
print(f"\n时间统计:")
print(f"  总训练时间: {np.sum(train_times):.2f} 秒 (平均每个fold: {np.mean(train_times):.2f} 秒)")
print(f"  总预测时间: {np.sum(predict_times):.2f} 秒 (平均每个fold: {np.mean(predict_times):.2f} 秒)")
print(f"  预测/训练时间比: {np.sum(predict_times)/np.sum(train_times):.2f}x")
print(f"  平均每个测试样本预测时间: {np.sum(predict_times)/len(all_y_true)*1000:.2f} 毫秒")

# 计算ITR（信息传输率）- 动态窗口的主要优势
performance = Performance(estimators_list=["Acc", "tITR"], Tw=np.mean(all_durations))
results = performance.evaluate(y_true=np.array(all_y_true), y_pred=np.array(all_y_pred))
print(f"ITR: {results.get('tITR', 'N/A'):.2f} bits/min")
print("="*70)

# 窗口长度分布统计
print("\n窗口长度分布:")
unique_durations, counts = np.unique(all_durations, return_counts=True)
for dur, count in zip(unique_durations, counts):
    percentage = count / len(all_durations) * 100
    print(f"  {dur:.2f}s: {count}次 ({percentage:.1f}%)")
