# -*- coding: utf-8 -*-
"""
MAT文件查看器
一个用于读取和展示MATLAB .mat文件详细信息的图形界面工具
"""
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from metabci.brainda.utils.io import loadmat
except ImportError:
    import scipy.io as sio
    def loadmat(mat_file):
        return sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)


class MatViewerApp:
    """MAT文件查看器主应用类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("MAT文件查看器")
        self.root.geometry("1200x800")
        
        self.mat_data = None
        self.file_path = None
        self.current_array = None
        
        self._create_widgets()
        
    def _create_widgets(self):
        """创建界面组件"""
        # 顶部文件选择区域
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)
        
        ttk.Label(file_frame, text="文件路径:").pack(side=tk.LEFT, padx=5)
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(file_frame, text="选择文件", command=self._select_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="加载文件", command=self._load_file).pack(side=tk.LEFT, padx=5)
        
        # 创建Notebook（标签页）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标签页1: 变量信息表格
        self.info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.info_frame, text="变量信息")
        self._create_info_tab()
        
        # 标签页2: 数据预览
        self.preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_frame, text="数据预览")
        self._create_preview_tab()
        
        # 标签页3: 详细信息
        self.detail_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detail_frame, text="详细信息")
        self._create_detail_tab()
        
    def _create_info_tab(self):
        """创建变量信息表格标签页"""
        # 工具栏
        toolbar = ttk.Frame(self.info_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="导出CSV", command=self._export_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="刷新", command=self._refresh_info).pack(side=tk.LEFT, padx=5)
        
        # 表格
        table_frame = ttk.Frame(self.info_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建Treeview表格
        columns = ("变量名", "类型", "形状", "大小", "数据类型", "最小值", "最大值", "均值")
        self.info_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        # 设置列标题和宽度
        column_widths = {"变量名": 150, "类型": 100, "形状": 120, "大小": 100, 
                        "数据类型": 100, "最小值": 100, "最大值": 100, "均值": 100}
        for col in columns:
            self.info_tree.heading(col, text=col)
            self.info_tree.column(col, width=column_widths.get(col, 100))
        
        # 滚动条
        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.info_tree.yview)
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.info_tree.xview)
        self.info_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 布局
        self.info_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # 双击事件
        self.info_tree.bind("<Double-1>", self._on_item_double_click)
        
    def _create_preview_tab(self):
        """创建数据预览标签页"""
        # 变量选择
        select_frame = ttk.Frame(self.preview_frame)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(select_frame, text="选择变量:").pack(side=tk.LEFT, padx=5)
        self.var_combo = ttk.Combobox(select_frame, width=30, state="readonly")
        self.var_combo.pack(side=tk.LEFT, padx=5)
        self.var_combo.bind("<<ComboboxSelected>>", self._on_var_selected)
        
        ttk.Label(select_frame, text="显示形状:").pack(side=tk.LEFT, padx=5)
        self.shape_var = tk.StringVar()
        ttk.Label(select_frame, textvariable=self.shape_var).pack(side=tk.LEFT, padx=5)
        
        # 控制按钮
        control_frame = ttk.Frame(self.preview_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_full_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="显示完整数据", 
                       variable=self.show_full_var,
                       command=self._on_var_selected).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="导出完整数据", 
                  command=self._export_full_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="保存为CSV", 
                  command=self._save_array_csv).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="保存为NPY", 
                  command=self._save_array_npy).pack(side=tk.LEFT, padx=5)
        
        # 切片控制（用于高维数组）
        slice_frame = ttk.LabelFrame(self.preview_frame, text="数组切片控制", padding=5)
        slice_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.slice_controls = []
        self.slice_frame = slice_frame
        
        # 数据预览区域
        preview_text = scrolledtext.ScrolledText(self.preview_frame, wrap=tk.NONE, 
                                                 font=("Consolas", 9))
        preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.preview_text = preview_text
        
    def _create_detail_tab(self):
        """创建详细信息标签页"""
        detail_text = scrolledtext.ScrolledText(self.detail_frame, wrap=tk.WORD,
                                                font=("Consolas", 10))
        detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detail_text = detail_text
        
    def _select_file(self):
        """选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择MAT文件",
            filetypes=[("MAT文件", "*.mat"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.file_path = file_path
            
    def _load_file(self):
        """加载MAT文件"""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("错误", "请先选择有效的文件路径！")
            return
            
        try:
            self.file_path = file_path
            self.mat_data = loadmat(file_path)
            self._update_info_table()
            self._update_preview()
            self._update_detail()
            messagebox.showinfo("成功", f"文件加载成功！\n找到 {len(self._get_valid_keys())} 个变量")
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败：\n{str(e)}")
            
    def _get_valid_keys(self):
        """获取有效的变量名（排除MATLAB内部变量）"""
        if self.mat_data is None:
            return []
        # 排除MATLAB内部变量
        exclude_keys = {'__header__', '__version__', '__globals__'}
        return [key for key in self.mat_data.keys() if key not in exclude_keys]
    
    def _get_var_info(self, var_name, var_data):
        """获取变量的详细信息"""
        info = {
            "变量名": var_name,
            "类型": type(var_data).__name__,
            "形状": "",
            "大小": "",
            "数据类型": "",
            "最小值": "",
            "最大值": "",
            "均值": ""
        }
        
        try:
            if isinstance(var_data, np.ndarray):
                info["形状"] = str(var_data.shape)
                info["大小"] = str(var_data.size)
                info["数据类型"] = str(var_data.dtype)
                
                if var_data.size > 0:
                    if np.issubdtype(var_data.dtype, np.number):
                        info["最小值"] = f"{np.nanmin(var_data):.6f}"
                        info["最大值"] = f"{np.nanmax(var_data):.6f}"
                        if var_data.size > 0:
                            info["均值"] = f"{np.nanmean(var_data):.6f}"
            elif isinstance(var_data, (list, tuple)):
                info["形状"] = f"({len(var_data)},)"
                info["大小"] = str(len(var_data))
                info["数据类型"] = "list/tuple"
            elif isinstance(var_data, dict):
                info["形状"] = f"dict({len(var_data)} keys)"
                info["大小"] = str(len(var_data))
                info["数据类型"] = "dict"
            elif isinstance(var_data, str):
                info["形状"] = f"({len(var_data)},)"
                info["大小"] = str(len(var_data))
                info["数据类型"] = "str"
            else:
                info["大小"] = "1"
                info["数据类型"] = str(type(var_data).__name__)
        except Exception as e:
            info["类型"] = f"Error: {str(e)}"
            
        return info
    
    def _update_info_table(self):
        """更新变量信息表格"""
        # 清空表格
        for item in self.info_tree.get_children():
            self.info_tree.delete(item)
            
        if self.mat_data is None:
            return
            
        valid_keys = self._get_valid_keys()
        for var_name in valid_keys:
            var_data = self.mat_data[var_name]
            info = self._get_var_info(var_name, var_data)
            self.info_tree.insert("", tk.END, values=(
                info["变量名"],
                info["类型"],
                info["形状"],
                info["大小"],
                info["数据类型"],
                info["最小值"],
                info["最大值"],
                info["均值"]
            ))
    
    def _update_preview(self):
        """更新数据预览"""
        self.preview_text.delete(1.0, tk.END)
        
        if self.mat_data is None:
            return
            
        valid_keys = self._get_valid_keys()
        self.var_combo['values'] = valid_keys
        
        if valid_keys:
            self.var_combo.current(0)
            self._on_var_selected()
    
    def _on_var_selected(self, event=None):
        """变量选择事件"""
        var_name = self.var_combo.get()
        if not var_name or self.mat_data is None:
            return
            
        var_data = self.mat_data[var_name]
        self.preview_text.delete(1.0, tk.END)
        
        # 清除旧的切片控制
        for widget in self.slice_frame.winfo_children():
            widget.destroy()
        self.slice_controls = []
        
        try:
            if isinstance(var_data, np.ndarray):
                self.shape_var.set(str(var_data.shape))
                self.current_array = var_data
                
                # 创建切片控制（对于高维数组）
                if var_data.ndim > 2:
                    self._create_slice_controls(var_data.shape)
                
                show_full = self.show_full_var.get()
                
                # 根据数组大小和用户选择决定显示方式
                if show_full or var_data.size <= 10000:
                    # 显示完整数据或小数组
                    if var_data.ndim <= 2:
                        # 2D或1D数组：完整显示
                        preview_str = np.array2string(var_data, max_line_width=120, 
                                                      precision=6, suppress_small=True,
                                                      threshold=np.inf)
                    else:
                        # 高维数组：显示完整数据（可能很大）
                        preview_str = np.array2string(var_data, max_line_width=120,
                                                      precision=6, suppress_small=True,
                                                      threshold=np.inf)
                        preview_str = f"完整数组数据 (形状: {var_data.shape}):\n\n" + preview_str
                else:
                    # 大数组：显示统计信息和切片预览
                    preview_str = f"数组形状: {var_data.shape}\n"
                    preview_str += f"数据类型: {var_data.dtype}\n"
                    preview_str += f"总元素数: {var_data.size:,}\n"
                    
                    if np.issubdtype(var_data.dtype, np.number):
                        preview_str += f"最小值: {np.nanmin(var_data):.6f}\n"
                        preview_str += f"最大值: {np.nanmax(var_data):.6f}\n"
                        preview_str += f"均值: {np.nanmean(var_data):.6f}\n"
                        preview_str += f"标准差: {np.nanstd(var_data):.6f}\n\n"
                    
                    # 显示不同维度的切片
                    if var_data.ndim == 4:
                        # 检查是否是Wang2016格式
                        if var_data.shape == (64, 1500, 40, 6):
                            preview_str += "=== 维度说明 ===\n"
                            preview_str += "数组形状: (64, 1500, 40, 6)\n"
                            preview_str += "维度0 (64): 电极索引 - 64个EEG通道，覆盖整个头皮\n"
                            preview_str += "维度1 (1500): 时间点 - 1500个采样点，对应6秒数据（采样率250Hz）\n"
                            preview_str += "维度2 (40): 目标索引 - 40个不同的闪烁频率/字符（8.0-15.8 Hz，间隔0.2 Hz）\n"
                            preview_str += "维度3 (6): 块索引 - 6个实验块（block），每个block包含40个trial\n\n"
                        
                        # 4D数组：显示几个切片的示例
                        preview_str += "=== 切片预览 ===\n\n"
                        if var_data.shape == (64, 1500, 40, 6):
                            preview_str += "data[0, :, 0, 0] (第1个通道，所有时间点，第1个目标，第1个block):\n"
                        else:
                            preview_str += "data[0, :, 0, 0] (第1个通道，第1个类别，第1个trial):\n"
                        preview_str += np.array2string(var_data[0, :, 0, 0], max_line_width=120,
                                                      precision=6, suppress_small=True) + "\n\n"
                        
                        if var_data.shape == (64, 1500, 40, 6):
                            preview_str += "data[:, 0, 0, 0] (所有通道，第1个时间点，第1个目标，第1个block):\n"
                        else:
                            preview_str += "data[:, 0, 0, 0] (所有通道，第1个时间点，第1个类别，第1个trial):\n"
                        preview_str += np.array2string(var_data[:, 0, 0, 0], max_line_width=120,
                                                      precision=6, suppress_small=True) + "\n\n"
                        
                        if var_data.shape == (64, 1500, 40, 6):
                            preview_str += "data[0, 0, :, 0] (第1个通道，第1个时间点，所有目标，第1个block):\n"
                        else:
                            preview_str += "data[0, 0, :, 0] (第1个通道，第1个时间点，所有类别，第1个trial):\n"
                        preview_str += np.array2string(var_data[0, 0, :, 0], max_line_width=120,
                                                      precision=6, suppress_small=True) + "\n\n"
                        preview_str += "提示: 勾选'显示完整数据'可查看全部数据，或使用切片控制查看特定切片\n"
                    elif var_data.ndim == 3:
                        preview_str += "=== 切片预览 ===\n\n"
                        preview_str += "data[0, :, :] (第1个切片):\n"
                        preview_str += np.array2string(var_data[0, :, :], max_line_width=120,
                                                      precision=6, suppress_small=True) + "\n\n"
                        preview_str += "提示: 勾选'显示完整数据'可查看全部数据\n"
                    else:
                        preview_str += "提示: 勾选'显示完整数据'可查看全部数据\n"
                    
            elif isinstance(var_data, dict):
                self.shape_var.set(f"dict({len(var_data)} keys)")
                self.current_array = None
                preview_str = f"字典包含 {len(var_data)} 个键:\n\n"
                for key in list(var_data.keys())[:20]:  # 只显示前20个键
                    preview_str += f"  '{key}': {type(var_data[key]).__name__}\n"
                if len(var_data) > 20:
                    preview_str += f"\n... 还有 {len(var_data) - 20} 个键"
                    
            elif isinstance(var_data, (list, tuple)):
                self.shape_var.set(f"{type(var_data).__name__}({len(var_data)})")
                self.current_array = None
                if self.show_full_var.get() or len(var_data) <= 1000:
                    preview_str = f"{type(var_data).__name__} 完整数据:\n\n"
                    for i, item in enumerate(var_data):
                        preview_str += f"  [{i}]: {str(item)}\n"
                else:
                    preview_str = f"{type(var_data).__name__} 包含 {len(var_data)} 个元素:\n\n"
                    for i, item in enumerate(var_data[:50]):  # 只显示前50个元素
                        preview_str += f"  [{i}]: {str(item)[:100]}\n"
                    preview_str += f"\n... 还有 {len(var_data) - 50} 个元素"
            else:
                self.shape_var.set("scalar")
                self.current_array = None
                preview_str = f"值: {var_data}\n类型: {type(var_data).__name__}"
                
            self.preview_text.insert(1.0, preview_str)
        except Exception as e:
            import traceback
            error_msg = f"预览错误: {str(e)}\n\n{traceback.format_exc()}"
            self.preview_text.insert(1.0, error_msg)
    
    def _create_slice_controls(self, shape):
        """为高维数组创建切片控制"""
        # 检查是否是Wang2016数据集的4维数组格式
        is_wang2016_format = (len(shape) == 4 and shape == (64, 1500, 40, 6))
        
        # 如果是Wang2016格式，显示维度说明
        if is_wang2016_format:
            info_label = ttk.Label(self.slice_frame, 
                                  text="维度说明: [电极索引(64通道), 时间点(1500点), 目标索引(40个频率), 块索引(6个block)]",
                                  foreground="blue", font=("Arial", 9, "bold"))
            info_label.pack(fill=tk.X, padx=5, pady=5)
            
            # 添加分隔线
            separator = ttk.Separator(self.slice_frame, orient=tk.HORIZONTAL)
            separator.pack(fill=tk.X, padx=5, pady=2)
        
        # 维度名称映射（针对Wang2016格式）
        dim_names = {
            (64, 1500, 40, 6): ["电极索引 (64通道)", "时间点 (1500点)", "目标索引 (40个频率)", "块索引 (6个block)"]
        }
        
        dim_labels = dim_names.get(shape, [f"维度 {i}" for i in range(len(shape))])
        
        for dim_idx, dim_size in enumerate(shape):
            control_frame = ttk.Frame(self.slice_frame)
            control_frame.pack(fill=tk.X, padx=2, pady=2)
            
            # 显示维度标签和说明
            if dim_idx < len(dim_labels):
                label_text = f"{dim_labels[dim_idx]} (大小: {dim_size}):"
            else:
                label_text = f"维度 {dim_idx} (大小: {dim_size}):"
            
            ttk.Label(control_frame, text=label_text).pack(side=tk.LEFT, padx=5)
            
            slice_var = tk.StringVar(value=":")
            entry = ttk.Entry(control_frame, textvariable=slice_var, width=20)
            entry.pack(side=tk.LEFT, padx=5)
            
            ttk.Button(control_frame, text="应用切片", 
                      command=lambda idx=dim_idx, var=slice_var: self._apply_slice(idx, var)).pack(side=tk.LEFT, padx=5)
            
            self.slice_controls.append((dim_idx, slice_var, entry))
    
    def _apply_slice(self, dim_idx, slice_var):
        """应用切片"""
        if not hasattr(self, 'current_array') or self.current_array is None:
            return
        
        try:
            # 解析切片字符串
            slice_str = slice_var.get().strip()
            if slice_str == ":":
                # 全选
                slices = [slice(None)]
            elif ":" in slice_str:
                # 范围切片，如 "0:10" 或 ":10" 或 "10:"
                parts = slice_str.split(":")
                if len(parts) == 2:
                    start = int(parts[0]) if parts[0] else None
                    end = int(parts[1]) if parts[1] else None
                    slices = [slice(start, end)]
                else:
                    messagebox.showerror("错误", f"无效的切片格式: {slice_str}")
                    return
            else:
                # 单个索引
                idx = int(slice_str)
                slices = [idx]
            
            # 构建完整的切片
            full_slice = [slice(None)] * self.current_array.ndim
            full_slice[dim_idx] = slices[0]
            
            # 应用切片
            sliced_data = self.current_array[tuple(full_slice)]
            
            # 显示切片结果
            self.preview_text.delete(1.0, tk.END)
            preview_str = f"切片结果 (维度 {dim_idx} = {slice_str}):\n"
            preview_str += f"形状: {sliced_data.shape}\n\n"
            preview_str += np.array2string(sliced_data, max_line_width=120,
                                          precision=6, suppress_small=True,
                                          threshold=np.inf)
            self.preview_text.insert(1.0, preview_str)
            
        except Exception as e:
            messagebox.showerror("错误", f"应用切片失败: {str(e)}")
    
    def _update_detail(self):
        """更新详细信息"""
        self.detail_text.delete(1.0, tk.END)
        
        if self.mat_data is None or self.file_path is None:
            return
            
        detail_str = f"文件路径: {self.file_path}\n"
        detail_str += f"文件大小: {os.path.getsize(self.file_path) / 1024:.2f} KB\n\n"
        
        valid_keys = self._get_valid_keys()
        detail_str += f"变量总数: {len(valid_keys)}\n\n"
        detail_str += "=" * 80 + "\n\n"
        
        for var_name in valid_keys:
            var_data = self.mat_data[var_name]
            detail_str += f"变量名: {var_name}\n"
            detail_str += f"类型: {type(var_data).__name__}\n"
            
            try:
                if isinstance(var_data, np.ndarray):
                    detail_str += f"形状: {var_data.shape}\n"
                    detail_str += f"大小: {var_data.size}\n"
                    detail_str += f"数据类型: {var_data.dtype}\n"
                    
                    # 如果是Wang2016格式的4维数组，添加维度说明
                    if var_data.shape == (64, 1500, 40, 6):
                        detail_str += "\n【维度说明 - Wang2016 SSVEP数据集格式】\n"
                        detail_str += "维度0 (64): 电极索引 - 64个EEG通道，覆盖整个头皮，按照国际10-20系统排列\n"
                        detail_str += "维度1 (1500): 时间点 - 1500个采样点，对应6秒数据（采样率250Hz：6秒 × 250 = 1500点）\n"
                        detail_str += "维度2 (40): 目标索引 - 40个不同的闪烁频率/字符，频率范围8-15.8 Hz，间隔0.2 Hz\n"
                        detail_str += "维度3 (6): 块索引 - 6个实验块（block），每个block包含40个trial（对应40个目标）\n"
                        detail_str += "\n数据访问示例:\n"
                        detail_str += "  data[channel_idx, time_idx, target_idx, block_idx]\n"
                        detail_str += "  - 第1个通道，所有时间点，第1个目标，第1个block: data[0, :, 0, 0]\n"
                        detail_str += "  - 所有通道，第1个时间点，第1个目标，第1个block: data[:, 0, 0, 0]\n"
                        detail_str += "  - 第1个通道，所有时间点，所有目标，第1个block: data[0, :, :, 0]\n"
                        detail_str += "  - 所有通道，所有时间点，第1个目标，所有block: data[:, :, 0, :]\n\n"
                    
                    if var_data.size > 0 and np.issubdtype(var_data.dtype, np.number):
                        detail_str += f"最小值: {np.nanmin(var_data):.6f}\n"
                        detail_str += f"最大值: {np.nanmax(var_data):.6f}\n"
                        detail_str += f"均值: {np.nanmean(var_data):.6f}\n"
                        detail_str += f"标准差: {np.nanstd(var_data):.6f}\n"
                elif isinstance(var_data, dict):
                    detail_str += f"字典键数量: {len(var_data)}\n"
                    detail_str += f"键列表: {list(var_data.keys())[:10]}\n"
                elif isinstance(var_data, (list, tuple)):
                    detail_str += f"长度: {len(var_data)}\n"
                    detail_str += f"元素类型: {type(var_data[0]).__name__ if len(var_data) > 0 else 'empty'}\n"
                else:
                    detail_str += f"值: {var_data}\n"
            except Exception as e:
                detail_str += f"错误: {str(e)}\n"
                
            detail_str += "\n" + "=" * 80 + "\n\n"
            
        self.detail_text.insert(1.0, detail_str)
    
    def _on_item_double_click(self, event):
        """双击表格项事件"""
        selection = self.info_tree.selection()
        if selection:
            item = self.info_tree.item(selection[0])
            var_name = item['values'][0]
            self.var_combo.set(var_name)
            self.notebook.select(1)  # 切换到预览标签页
            self._on_var_selected()
    
    def _refresh_info(self):
        """刷新信息"""
        if self.mat_data is not None:
            self._update_info_table()
            self._update_preview()
            self._update_detail()
    
    def _export_csv(self):
        """导出变量信息为CSV"""
        if self.mat_data is None:
            messagebox.showwarning("警告", "没有可导出的数据！")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存CSV文件",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                valid_keys = self._get_valid_keys()
                data = []
                for var_name in valid_keys:
                    var_data = self.mat_data[var_name]
                    info = self._get_var_info(var_name, var_data)
                    data.append(info)
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"已导出到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：\n{str(e)}")
    
    def _export_full_data(self):
        """导出完整数组数据到文本文件"""
        if not hasattr(self, 'current_array') or self.current_array is None:
            messagebox.showwarning("警告", "请先选择一个数组变量！")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存完整数据",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"数组形状: {self.current_array.shape}\n")
                    f.write(f"数据类型: {self.current_array.dtype}\n")
                    f.write(f"总元素数: {self.current_array.size:,}\n\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(np.array2string(self.current_array, max_line_width=120,
                                           precision=6, suppress_small=True,
                                           threshold=np.inf))
                messagebox.showinfo("成功", f"完整数据已导出到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败：\n{str(e)}")
    
    def _save_array_csv(self):
        """将数组保存为CSV文件（2D数组）"""
        if not hasattr(self, 'current_array') or self.current_array is None:
            messagebox.showwarning("警告", "请先选择一个数组变量！")
            return
        
        if self.current_array.ndim > 2:
            # 对于高维数组，需要先reshape或选择切片
            reply = messagebox.askyesno(
                "确认", 
                f"数组是{self.current_array.ndim}维的，CSV只能保存2D数据。\n"
                f"是否将其reshape为2D？\n"
                f"原始形状: {self.current_array.shape}"
            )
            if reply:
                try:
                    # 将数组reshape为2D
                    reshaped = self.current_array.reshape(-1, self.current_array.shape[-1])
                except:
                    messagebox.showerror("错误", "无法reshape数组，请使用切片功能先提取2D数据")
                    return
            else:
                return
        else:
            reshaped = self.current_array
        
        file_path = filedialog.asksaveasfilename(
            title="保存为CSV",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(reshaped)
                df.to_csv(file_path, index=False, header=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败：\n{str(e)}")
    
    def _save_array_npy(self):
        """将数组保存为NPY文件"""
        if not hasattr(self, 'current_array') or self.current_array is None:
            messagebox.showwarning("警告", "请先选择一个数组变量！")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存为NPY",
            defaultextension=".npy",
            filetypes=[("NumPy文件", "*.npy"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                np.save(file_path, self.current_array)
                messagebox.showinfo("成功", f"已保存到: {file_path}\n形状: {self.current_array.shape}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败：\n{str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = MatViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

