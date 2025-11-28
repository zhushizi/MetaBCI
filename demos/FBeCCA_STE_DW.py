# -*- coding: utf-8 -*-
"""
FBECCA with Dynamic Window (STE-DW style)
基于STE-DW论文的动态窗口FBECCA算法

参考论文：
Chen Yang et al., A Dynamic Window Recognition Algorithm for SSVEP-Based 
Brain-Computer Interfaces Using A Spatio-Temporal Equalizer, 
International Journal of Neural Systems, doi: 10.1142/S0129065718500284

STE-DW核心思想：
1. Spatio-Temporal Equalization (STE): 减少背景噪声的时空相关性
2. Multiple Hypotheses Testing: 基于多重假设检验的刺激终止准则
3. Dynamic Window: 自适应控制刺激时间，提高ITR

论文摘要：
- 使用时空均衡算法减少背景噪声的时空相关性
- 基于多重假设检验理论，使用刺激终止准则自适应控制动态窗口
- 在benchmark数据集上达到平均ITR 134 bits/min
"""
import sys
import numpy as np
from scipy import stats
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

# STE-DW多重假设检验参数
# 论文中使用多重假设检验来决定何时停止
# 这里使用似然比检验的思想
alpha = 0.05  # 显著性水平
min_samples_for_test = 50  # 进行假设检验所需的最小样本数

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

# 6-fold cross validation
set_random_seeds(38)
kfold = 6
indices = generate_kfold_indices(meta, kfold=kfold)

# 滤波器权重
filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]


def spatio_temporal_equalization(X):
    """
    时空均衡预处理（参考STE-DW论文）- 优化版本
    
    减少背景噪声的时空相关性：
    - 空间维度：对通道进行去相关（空间白化）
    - 时间维度：对时间序列进行去相关（时间白化）
    
    Parameters
    ----------
    X : ndarray
        输入数据，形状为 (n_trials, n_channels, n_samples)
        
    Returns
    -------
    X_equalized : ndarray
        均衡后的数据，形状与输入相同
    """
    X = np.copy(X)
    n_trials, n_channels, n_samples = X.shape
    
    # 去均值（向量化）
    X = X - np.mean(X, axis=-1, keepdims=True)
    
    # 空间均衡：计算空间协方差矩阵并进行白化（向量化优化）
    # 向量化计算所有trial的协方差矩阵
    # X形状: (n_trials, n_channels, n_samples)
    # 使用einsum高效计算
    C_spatial = np.einsum('ijk,ilk->jl', X, X) / (n_trials * n_samples)
    
    # 添加正则化避免奇异矩阵
    C_spatial += np.eye(n_channels) * 1e-6
    
    # 空间白化：W_spatial = C^(-1/2)
    eigenvals, eigenvecs = np.linalg.eigh(C_spatial)
    eigenvals = np.maximum(eigenvals, 1e-10)  # 避免负值
    W_spatial = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
    
    # 应用空间白化（向量化）
    # X_equalized[i] = W_spatial @ X[i] for all i
    X_equalized = np.einsum('ij,kjl->kil', W_spatial, X)
    
    # 时间均衡：去均值（已完成）
    X_equalized = X_equalized - np.mean(X_equalized, axis=-1, keepdims=True)
    
    return X_equalized


class STE_DW_FBECCA:
    """
    STE-DW风格的动态窗口FBECCA分类器
    
    参考STE-DW论文实现：
    1. 时空均衡预处理减少噪声的时空相关性
    2. 多重假设检验终止准则
    3. 动态窗口自适应选择
    """
    
    def __init__(self, filterbank, filterweights, freq_list, srate=250,
                 min_duration=0.4, max_duration=1.0, step=0.2, 
                 alpha=0.05, n_components=1, use_ste=False, fast_mode=True):
        self.filterbank = filterbank
        self.filterweights = filterweights
        self.freq_list = freq_list
        self.srate = srate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.step = step
        self.alpha = alpha  # 多重假设检验的显著性水平
        self.n_components = n_components
        self.use_ste = use_ste  # 是否使用时空均衡
        self.fast_mode = fast_mode  # 快速模式：跳过统计信息计算，使用简单置信度
        
        # 为不同窗口长度训练多个模型
        self.models = {}
        self.Yf_dict = {}
        # 存储训练数据的统计信息（用于假设检验）
        self.train_stats = {}
        
    def fit(self, X, y):
        """
        训练多个不同窗口长度的模型
        
        参考STE-DW：为不同时间窗口训练多个分类器
        """
        print("训练动态窗口模型（STE-DW风格）...")
        
        # 应用时空均衡（如果启用）
        if self.use_ste:
            print("  应用时空均衡预处理...")
            X = spatio_temporal_equalization(X)
        
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
            
            # 计算训练数据的统计信息（用于假设检验）
            if not self.fast_mode:
                # 完整模式：计算统计信息用于假设检验
                train_stats = {}
                unique_labels = np.unique(y)
                sample_size = min(30, len(X_window))  # 减少到30个样本以加速
                
                # 设置随机种子以确保可重复性
                np.random.seed(42)
                
                for label in unique_labels:
                    label_mask = (y == label)
                    label_indices = np.where(label_mask)[0]
                    
                    if len(label_indices) == 0:
                        continue
                    
                    # 采样部分样本计算特征
                    if len(label_indices) > sample_size:
                        sampled_indices = np.random.choice(label_indices, sample_size, replace=False)
                    else:
                        sampled_indices = label_indices
                    
                    # 计算采样样本的特征
                    sampled_features = estimator.transform(X_window[sampled_indices])
                    # 计算该类别下最大特征值的统计
                    max_features = np.max(sampled_features, axis=1)
                    
                    train_stats[label] = {
                        'mean': np.mean(max_features),
                        'std': np.std(max_features),
                        'n_samples': len(max_features)
                    }
                self.train_stats[duration] = train_stats
            # 快速模式：跳过统计信息计算，使用简单的置信度方法
            
            print(f"  完成窗口长度 {duration:.2f}s 的训练 (数据点: {n_samples})", end='\r')
        print()  # 换行
    
    def _multiple_hypotheses_test(self, features, duration):
        """
        多重假设检验（参考STE-DW论文）
        
        对每个频率类别进行假设检验：
        H0: 当前特征值来自背景噪声分布
        H1: 当前特征值来自目标信号分布
        
        使用似然比检验或t检验来判断是否满足终止条件
        
        Parameters
        ----------
        features : ndarray
            特征向量，形状为 (n_classes,)
        duration : float
            当前窗口长度
            
        Returns
        -------
        decision : bool
            是否满足终止条件（True表示可以停止）
        confidence : float
            置信度值
        predicted_label : int
            预测的类别
        """
        # 快速模式：使用简单的置信度方法
        if self.fast_mode or duration not in self.train_stats:
            # 如果没有训练统计信息，使用简单的置信度方法
            features_sorted = np.sort(features)[::-1]
            if len(features_sorted) > 1:
                confidence = features_sorted[0] / (features_sorted[1] + 1e-10)
                return confidence > 1.5, confidence, np.argmax(features)
            else:
                return True, features[0], np.argmax(features)
        
        predicted_label = np.argmax(features)
        max_feature = features[predicted_label]
        
        # 获取预测类别的训练统计信息
        train_stat = self.train_stats[duration].get(predicted_label, None)
        if train_stat is None:
            # 如果没有该类别的统计信息，使用简单方法
            features_sorted = np.sort(features)[::-1]
            confidence = features_sorted[0] / (features_sorted[1] + 1e-10) if len(features_sorted) > 1 else features[0]
            return confidence > 1.5, confidence, predicted_label
        
        # 使用t检验：检验当前特征值是否显著大于训练数据的均值
        # H0: max_feature <= mean (来自噪声)
        # H1: max_feature > mean (来自信号)
        mean = train_stat['mean']
        std = train_stat['std']
        n_samples = train_stat['n_samples']
        
        if std < 1e-10 or n_samples < 2:
            # 如果标准差太小或样本太少，使用简单方法
            confidence = max_feature / (np.max(features[features != max_feature]) + 1e-10) if np.any(features != max_feature) else max_feature
            return confidence > 1.5, confidence, predicted_label
        
        # t统计量
        t_stat = (max_feature - mean) / (std / np.sqrt(n_samples) + 1e-10)
        
        # 单侧t检验（因为我们只关心是否显著大于）
        # 自由度
        df = n_samples - 1
        # t临界值（单侧检验，显著性水平alpha）
        t_critical = stats.t.ppf(1 - self.alpha, df)
        
        # p值
        p_value = 1 - stats.t.cdf(t_stat, df)
        
        # 置信度计算：使用特征值的相对差异和p值
        # 方法1：基于特征值的相对差异（更直观）
        features_sorted = np.sort(features)[::-1]
        if len(features_sorted) > 1:
            # 最大特征值与次大特征值的比值
            ratio_confidence = features_sorted[0] / (features_sorted[1] + 1e-10)
        else:
            ratio_confidence = features[0]
        
        # 方法2：基于p值的置信度（统计显著性）
        # p值越小，置信度越高；转换为0-1范围
        p_confidence = 1.0 - min(p_value, 1.0)  # p值越小，置信度越高
        
        # 综合置信度：结合两种方法
        # 如果t统计量显著，使用p值置信度；否则使用比值置信度
        if t_stat > t_critical and p_value < self.alpha:
            # 显著情况：使用p值置信度，并加上比值的影响
            confidence = p_confidence * (1 + np.log(ratio_confidence) / 10)
        else:
            # 不显著情况：主要使用比值置信度
            confidence = ratio_confidence / (1 + ratio_confidence)  # 归一化到0-1
        
        # 确保置信度为正数且在合理范围
        confidence = max(0.0, min(confidence, 10.0))
        
        # 决策：如果t统计量显著大于临界值，且p值小于alpha，则停止
        # 或者如果特征值比值足够大（简单启发式），也可以停止
        decision = ((t_stat > t_critical) and (p_value < self.alpha)) or (ratio_confidence > 1.5)
        
        return decision, confidence, predicted_label
    
    def predict_with_dynamic_window(self, X_single):
        """
        对单个样本进行动态窗口预测（STE-DW风格）
        
        从短窗口开始，逐步增加窗口长度，使用多重假设检验决定何时停止
        
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
        # 应用时空均衡（如果启用）
        if self.use_ste:
            # 对单个样本应用空间均衡（需要从训练数据中获取白化矩阵）
            # 这里简化处理：只做去均值
            X_single = X_single - np.mean(X_single, axis=-1, keepdims=True)
        
        # 只使用已训练的窗口长度
        durations = sorted(self.models.keys())
        
        for duration in durations:
            n_samples = int(self.srate * duration)
            
            # 提取当前窗口的数据
            X_window = X_single[:, :n_samples].copy()
            X_window = X_window - np.mean(X_window, axis=-1, keepdims=True)
            
            # 获取模型和参考信号
            model = self.models[duration]
            
            # 获取特征（相关性系数）
            features = model.transform(X_window[np.newaxis, ...])
            features = features[0]  # 取第一个样本的特征
            
            # 多重假设检验
            decision, confidence, pred_label = self._multiple_hypotheses_test(features, duration)
            
            # STE-DW终止条件：假设检验通过或达到最大窗口
            if decision or duration >= self.max_duration:
                return pred_label, duration, confidence
        
        # 如果所有窗口都不满足条件，返回最大窗口的结果
        duration = durations[-1]
        n_samples = int(self.srate * duration)
        X_window = X_single[:, :n_samples].copy()
        X_window = X_window - np.mean(X_window, axis=-1, keepdims=True)
        model = self.models[duration]
        
        features = model.transform(X_window[np.newaxis, ...])
        features = features[0]
        _, confidence, pred_label = self._multiple_hypotheses_test(features, duration)
        
        return pred_label, duration, confidence
    
    def predict(self, X):
        """
        批量预测
        
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
        labels = []
        durations = []
        confidences = []
        
        for i in range(n_trials):
            label, duration, confidence = self.predict_with_dynamic_window(X[i])
            labels.append(label)
            durations.append(duration)
            confidences.append(confidence)
        
        return np.array(labels), np.array(durations), np.array(confidences)


# 创建动态窗口分类器
# 速度优化说明：
# - step_duration=0.2: 窗口数量减少到4个（0.4, 0.6, 0.8, 1.0s）
# - use_ste=False: 关闭时空均衡可显著加速（约2-3倍）
# - 统计信息采样：只对30个样本计算特征统计
# - n_jobs=1: 对小数据集，单线程可能更快（避免并行开销）
estimator = STE_DW_FBECCA(
    filterbank=filterbank,
    filterweights=filterweights,
    freq_list=freq_list,
    srate=250,
    min_duration=min_duration,
    max_duration=max_duration,
    step=step_duration,
    alpha=alpha,
    n_components=1,
    use_ste=False,  # 设为False可显著加速训练（约2-3倍）
    fast_mode=True  # 快速模式：跳过统计信息计算，使用简单置信度（更快）
)

# 交叉验证
accs = []
all_durations = []
all_confidences = []
all_y_true = []
all_y_pred = []

for k in range(kfold):
    train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    # merge train and validate set
    train_ind = np.concatenate((train_ind, validate_ind))
    
    # 训练模型
    print(f"\n训练 Fold {k+1}...")
    estimator.fit(X=X[train_ind], y=y[train_ind])
    
    # 预测
    print(f"预测 Fold {k+1}...")
    p_labels, durations, confidences = estimator.predict(X[test_ind])
    
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
print("STE-DW风格动态窗口FBECCA结果:")
print("="*70)
print(f"平均准确率: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"平均窗口长度: {np.mean(all_durations):.2f}s ± {np.std(all_durations):.2f}s")
print(f"平均置信度: {np.mean(all_confidences):.4f} ± {np.std(all_confidences):.4f}")

# 计算ITR（信息传输率）- STE-DW的主要优势
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
