import shutil
import os
import datetime

def copy_directory_with_exclusions(src_dir, dst_dir, exclude_dirs=None, exclude_files=None):
    """
    将源目录下的所有文件和文件夹递归复制到目标目录中，支持排除指定的文件夹和文件。
    并在复制完成后，将生成的备份文件夹压缩为同名的 zip 包。
    
    参数:
    src_dir (str): 源目录路径
    dst_dir (str): 目标目录路径，如果不存在将被创建
    exclude_dirs (list): 要排除的文件夹名称列表
    exclude_files (list): 要排除的文件名称列表
    """
    if exclude_dirs is None:
        exclude_dirs = ['data', 'logs', 'backup', 'display', 'tmp', 'temp', 'env', 'test_out', 'watch']
    if exclude_files is None:
        exclude_files = []
    
    # 确保源目录存在
    # os.makedirs(dst_dir, exist_ok=True) # 这一行可以去掉，因为后面拼接路径后会创建
    if not os.path.exists(src_dir):
        raise ValueError(f"源目录不存在: {src_dir}")
    
    today = datetime.datetime.now().strftime("%Y%m%d")
    new_dst_dir = os.path.join(dst_dir, f"backup_{today}")
    
    # 如果当天的备份目录已存在，添加序号
    counter = 1
    original_new_dst_dir = new_dst_dir
    while os.path.exists(new_dst_dir) or os.path.exists(new_dst_dir + '.zip'):
        # 同时检查文件夹和zip文件是否存在，防止覆盖
        new_dst_dir = f"{original_new_dst_dir}_{counter}"
        counter += 1
        
    dst_dir = new_dst_dir
  
    # 创建目标目录（如果不存在）
    # 注意：copytree 会自动创建目标目录，如果 dirs_exist_ok=False (默认) 且目录已存在会报错
    # 这里我们已经确保了 dst_dir 是一个新的不冲突的路径，或者使用了 dirs_exist_ok=True
    os.makedirs(dst_dir, exist_ok=True)
    
    def ignore_func(current_dir, names):
        """
        忽略函数，用于 shutil.copytree
        """
        ignored_names = set()
        
        # 获取相对于源目录的路径
        rel_path = os.path.relpath(current_dir, src_dir)
        
        for name in names:
            # 如果是在源目录的根级别，检查是否需要排除
            if rel_path == '.':
                if name in exclude_dirs:
                    ignored_names.add(name)
                    print(f"忽略目录: {name}")
            
            # 检查文件是否需要排除
            if name in exclude_files:
                ignored_names.add(name)
                print(f"忽略文件: {name}")
        
        return ignored_names
    
    try:
        # 1. 复制目录
        shutil.copytree(src_dir, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        print(f"成功将 '{src_dir}' 复制到 '{dst_dir}'")
        print(f"已排除的目录: {exclude_dirs}")
        
        # 2. 压缩目录 (新增功能)
        print(f"正在压缩文件夹: {dst_dir} ...")
        # make_archive(base_name, format, root_dir)
        # base_name: 压缩包的路径（不带后缀），这里设为 dst_dir，生成的就会是 dst_dir.zip
        # root_dir: 要压缩的目录路径
        zip_path = shutil.make_archive(dst_dir, 'zip', dst_dir)
        print(f"成功生成压缩包: {zip_path}")
        
    except Exception as e:
        print(f"操作过程中发生错误: {e}")
        raise

def copy_directory_selective(src_dir, dst_dir, include_patterns=None, exclude_patterns=None):
    """
    更高级的复制函数，支持使用通配符模式进行包含和排除
    (也同步增加了压缩功能)
    """
    import fnmatch
    
    if include_patterns is None:
        include_patterns = ['*']  # 默认包含所有
    if exclude_patterns is None:
        exclude_patterns = []
    
    if not os.path.exists(src_dir):
        raise ValueError(f"源目录不存在: {src_dir}")
    
    # 路径处理逻辑
    today = datetime.datetime.now().strftime("%Y%m%d")
    final_dst_path = os.path.join(dst_dir, f"backup_{today}")
    
    if os.path.exists(final_dst_path) or os.path.exists(final_dst_path + ".zip"):
        counter = 1
        original_path = final_dst_path
        while os.path.exists(final_dst_path) or os.path.exists(final_dst_path + ".zip"):
            final_dst_path = f"{original_path}_{counter}"
            counter += 1
        print(f"目标路径已存在，将使用新路径: {final_dst_path}")
    
    dst_dir = final_dst_path
    os.makedirs(dst_dir, exist_ok=True)
    
    def should_include(name, is_dir=False):
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return False
        for pattern in include_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
    
    def ignore_func(current_dir, names):
        ignored_names = set()
        for name in names:
            full_path = os.path.join(current_dir, name)
            is_dir = os.path.isdir(full_path)
            if not should_include(name, is_dir):
                ignored_names.add(name)
                # print(f"忽略: {name}") # 减少日志输出
        return ignored_names
    
    try:
        shutil.copytree(src_dir, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        print(f"成功将 '{src_dir}' 复制到 '{dst_dir}'")
        
        # 新增压缩逻辑
        print(f"正在压缩文件夹: {dst_dir} ...")
        zip_path = shutil.make_archive(dst_dir, 'zip', dst_dir)
        print(f"成功生成压缩包: {zip_path}")
        
    except Exception as e:
        print(f"复制文件时发生错误: {e}")
        raise

# 使用示例
if __name__ == "__main__":
    src_dir = "/data/xieshangxuan/master/img_matching/p2wnet_icme"
    dst_dir = './backup'
    
    # 调用函数
    copy_directory_with_exclusions(src_dir, dst_dir)
