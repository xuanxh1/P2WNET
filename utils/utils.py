"""
通用工具函数集
包含动态加载、学习率调度、配置打印、结果记录等功能
"""
import importlib
from torch.optim.lr_scheduler import _LRScheduler
import os

def get_obj_from_str(string, reload=False):
    """
    从字符串动态加载类或函数
    
    参数:
        string: 完整的类/函数路径，如"module.submodule.ClassName"
        reload: 是否重新加载模块，默认False
    
    返回:
        加载的类或函数对象
    """
    # 从右边分割一次，分离模块路径和类名
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class WarmupStepLR(_LRScheduler):
    """
    带预热的阶梯式学习率调度器
    
    参数:
        optimizer: 优化器
        step_size: 衰减步长
        gamma: 学习率衰减系数，默认0.7
        warmup_epochs: 预热轮次，默认5
        warmup_lr_init: 预热初始学习率，默认1e-6
        last_epoch: 上次轮次，默认-1
    """
    def __init__(self, optimizer, step_size, gamma=0.7, warmup_epochs=5, warmup_lr_init=1e-6, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算当前轮次的学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * (self.last_epoch / self.warmup_epochs)
                    for base_lr in self.base_lrs]
        else:
            # 阶梯衰减阶段
            return [base_lr * self.gamma ** ((self.last_epoch - self.warmup_epochs) // self.step_size)
                    for base_lr in self.base_lrs]
        
def print_config(config, logger,indent=0):
    """
    递归打印配置字典，支持嵌套字典结构
    
    参数:
        config: 配置字典
        logger: 日志记录器
        indent: 缩进级别，默认0
    
    功能:
        顶级键占一行，嵌套字典键值对递归缩进显示
    """
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(' ' * indent + f"{key}:")
            print_config(value,logger, indent + 2)
        else:
            logger.info(' ' * indent + f"{key}: {value}")


def write_epoch_results(file_path, epoch_data, header=False):
    """
    将每轮的训练结果写入文件，自动对齐列名和数据
    
    参数:
        file_path: 输出文件路径
        epoch_data: 包含所有需要记录的数据的字典
        header: 是否写入表头，默认False
    
    功能:
        - 首次写入时自动创建表头和列宽信息
        - 后续写入根据列宽自动对齐
        - 支持浮点数、整数和特殊值(NA)的格式化
    """
    # 检查文件是否为空
    file_empty = True
    try:
        with open(file_path, 'r') as f:
            if f.read(1):
                file_empty = False  # 文件非空
    except FileNotFoundError:
        file_empty = True  # 文件不存在时视为文件为空

    # 首次写入时创建表头和列宽信息
    if header or file_empty:
        headers = list(epoch_data.keys())

        # 转换为字符串以计算列宽
        str_values = []
        for key, value in epoch_data.items():
            if key == 'epoch':
                str_values.append(f"{value:03d}")
            elif isinstance(value, (int, float)):
                if value == 100000:
                    str_values.append('NA')
                elif abs(value) < 0.0001:
                    str_values.append(f"{value:.7e}")
                else:
                    str_values.append(f"{value:.7f}")
            else:
                str_values.append(str(value))

        # 计算每列的最大宽度
        column_widths = {}
        for header, value in zip(headers, str_values):
            column_widths[header] = max(len(header), len(value))

        # 写入列宽和表头
        with open(file_path, 'w') as f:
            # 写入列宽信息(作为注释)
            widths_str = ','.join(f"{k}:{v}" for k, v in column_widths.items())
            f.write(f"#WIDTHS={widths_str}\n")
            # 写入对齐的表头
            header_line = ','.join(h.ljust(column_widths[h]) for h in headers)
            f.write(header_line + '\n')
    else:
        # 文件存在时读取列宽信息
        column_widths = {}
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#WIDTHS='):
                widths_str = first_line[8:]  # 去掉 '#WIDTHS=' 前缀
                for item in widths_str.split(','):
                    key, width = item.split(':')
                    column_widths[key] = int(width)
            else:
                # 缺少列宽信息时抛出错误
                raise ValueError("文件格式错误：缺少列宽信息")

    # 写入数据行(追加模式)
    with open(file_path, 'a') as f:
        values = []
        for key, value in epoch_data.items():
            if key == 'epoch':
                values.append(f"{value:03d}".ljust(column_widths[key]))
            elif isinstance(value, (int, float)):
                if value == 100000:
                    values.append('NA'.ljust(column_widths[key]))
                elif abs(value) < 0.0001:
                    values.append(f"{value:.7e}".ljust(column_widths[key]))
                else:
                    values.append(f"{value:.7f}".ljust(column_widths[key]))
            else:
                values.append(str(value).ljust(column_widths[key]))

        f.write(','.join(values) + '\n')

def read_epoch_results(file_path):
    """
    读取训练记录文件，将每列数据解析为列表
    
    参数:
        file_path: 训练记录文件路径
    
    返回:
        results: 字典，键为列名，值为该列的所有数据列表
    
    功能:
        - 跳过列宽信息行
        - 解析表头和数据
        - 自动转换数据类型(int/float)
    """
    results = {}
    
    with open(file_path, 'r') as f:
        # 跳过列宽信息行
        f.readline()
        # 读取表头
        headers = [h.strip() for h in f.readline().strip().split(',')]
        for header in headers:
            results[header] = []
            
        # 逐行读取数据
        for line in f:
            values = [v.strip() for v in line.strip().split(',')]
            for header, value in zip(headers, values):
                if value == 'NA':
                    results[header].append(None)
                elif header == 'epoch':
                    results[header].append(int(value))
                else:
                    try:
                        results[header].append(float(value))
                    except ValueError:
                        results[header].append(value)
                        
    return results