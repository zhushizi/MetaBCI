# -*- coding: utf-8 -*-
"""
Ring Buffer (Circular Buffer) Implementation Demo
环形缓冲区算法演示

Ring Buffer 是一种固定大小的缓冲区，当写满时，新数据会覆盖最早的数据。
常用于实时数据流处理（如EEG信号缓冲）。

多线程环境需要加锁保护。
"""
import numpy as np
from collections import deque
import time

class RingBuffer:
    """基于numpy实现的环形缓冲区"""
    
    def __init__(self, shape, dtype=np.float64):
        """
        初始化环形缓冲区
        
        Parameters
        ----------
        shape : tuple
            缓冲区形状 (capacity, n_channels) 或 (capacity,)
            第一个维度必须是容量(capacity)
        dtype : data-type, optional
            数据类型
        """
        self.shape = shape
        self.capacity = shape[0]
        self.dtype = dtype
        self._buffer = np.zeros(shape, dtype=dtype)
        self._idx = 0  # 当前写入位置的索引
        self._full = False  # 缓冲区是否已满
        
    def append(self, data):
        """
        追加数据
        
        Parameters
        ----------
        data : ndarray
            要追加的数据，形状必须与缓冲区的除第一维外的维度匹配
            可以是单个样本，也可以是一批样本
        """
        data = np.array(data, dtype=self.dtype)
        
        # 处理单个样本的情况 (n_channels,) -> (1, n_channels)
        if data.ndim < len(self.shape):
            data = data[np.newaxis, ...]
            
        n_samples = data.shape[0]
        
        if n_samples == 0:
            return
            
        # 如果新数据比容量还大，只保留最后 capacity 个
        if n_samples >= self.capacity:
            self._buffer[:] = data[-self.capacity:]
            self._idx = 0
            self._full = True
            return
            
        # 写入数据
        # 注意：使用分段处理而不是 fancy indexing，因为回绕时索引不连续
        if self._idx + n_samples <= self.capacity:
            # 情况1：直接追加，不回绕
            self._buffer[self._idx : self._idx + n_samples] = data
        else:
            # 情况2：回绕到开头
            n_end = self.capacity - self._idx  # 填满末尾所需的数量
            self._buffer[self._idx:] = data[:n_end]
            self._buffer[:n_samples - n_end] = data[n_end:]
            
        # 更新索引和状态
        self._idx = (self._idx + n_samples) % self.capacity
        if not self._full and (self._idx < n_samples or self._idx == 0):
            self._full = True
            
    def get_all(self):
        """
        获取缓冲区中的所有数据（按时间顺序）
        
        Returns
        -------
        data : ndarray
            有序的数据，形状 (n_current_samples, ...)
        """
        if not self._full:
            return self._buffer[:self._idx].copy()
        
        # 如果已满，需要重新排列：[idx:] + [:idx]
        # idx 是下一个写入位置，也是最早的数据位置
        return np.concatenate((self._buffer[self._idx:], self._buffer[:self._idx]), axis=0)
    
    def get_last(self, n):
        """
        获取最新的 n 个样本
        """
        if n > self.capacity:
            raise ValueError(f"Requested {n} samples, but capacity is {self.capacity}")
            
        # 如果缓冲区未满且数据不足
        current_size = self.capacity if self._full else self._idx
        if n > current_size:
            raise ValueError(f"Requested {n} samples, but only {current_size} available")
            
        # 计算起始位置
        start_idx = (self._idx - n) % self.capacity
        
        if start_idx + n <= self.capacity:
            return self._buffer[start_idx : start_idx + n].copy()
        else:
            return np.concatenate((self._buffer[start_idx:], self._buffer[:self._idx]), axis=0)
            
    def is_full(self):
        return self._full
        
    def __len__(self):
        return self.capacity if self._full else self._idx
        
    def clear(self):
        self._idx = 0
        self._full = False
        self._buffer.fill(0)


# ===== 演示与测试 =====
def demo_ring_buffer():
    print("=== Ring Buffer 演示 ===\n")
    
    # 1. 创建缓冲区：容量10，3个通道
    capacity = 10
    channels = 3
    rb = RingBuffer((capacity, channels))
    print(f"创建缓冲区: 容量={capacity}, 通道={channels}")
    
    # 2. 追加少量数据
    data1 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    print(f"\n追加 3 个样本:\n{data1}")
    rb.append(data1)
    print(f"当前缓冲区内容 (len={len(rb)}):\n{rb.get_all()}")
    
    # 3. 继续追加数据，直到快满
    data2 = np.array([[i, i, i] for i in range(4, 10)])  # 4,5,6,7,8,9
    print(f"\n追加 6 个样本 (累计9个):")
    rb.append(data2)
    print(f"当前缓冲区内容 (len={len(rb)}):\n{rb.get_all()}")
    
    # 4. 追加数据导致回绕 (覆盖最早的数据)
    data3 = np.array([[10, 10, 10], [11, 11, 11], [12, 12, 12]])
    print(f"\n追加 3 个样本 (触发覆盖):")
    rb.append(data3)
    print(f"当前缓冲区内容 (len={len(rb)}):\n{rb.get_all()}")
    print("注意：最早的 [1,1,1] 和 [2,2,2] 已经被覆盖")
    
    # 5. 获取最新的 N 个样本
    n_last = 5
    print(f"\n获取最新的 {n_last} 个样本:\n{rb.get_last(n_last)}")
    
    # 6. 性能测试
    print("\n=== 性能测试 ===")
    large_rb = RingBuffer((10000, 32)) # 10000点, 32通道
    chunk_size = 100
    n_iter = 1000
    
    # 先填满一些数据以避免读取错误
    large_rb.append(np.random.randn(1000, 32))
    
    start_time = time.time()
    for _ in range(n_iter):
        chunk = np.random.randn(chunk_size, 32)
        large_rb.append(chunk)
        _ = large_rb.get_last(500) # 模拟实时读取
        
    end_time = time.time()
    print(f"处理 {n_iter} 次写入(每次{chunk_size}样本) + 读取耗时: {end_time - start_time:.4f} 秒")


if __name__ == "__main__":
    demo_ring_buffer()

