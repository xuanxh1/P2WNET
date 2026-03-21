"""
文件复制工具
用于备份项目文件，排除特定目录和文件类型
"""
import os
import shutil

def copy_files_exclude(src_dir, dst_dir):
    """
    复制源目录下的所有文件到目标目录，保持目录结构
    排除特定目录和.out文件
    
    参数:
        src_dir: 源目录路径
        dst_dir: 目标目录路径
    
    功能:
        - 保持原有目录结构
        - 排除: data, test_out, checkpoints, tmp, watch, env, backup目录
        - 排除: .out后缀的文件
    """
    
    for root, dirs, files in os.walk(src_dir, topdown=True):
        # root: 当前正在遍历的目录
        # dirs: 当前目录下的文件夹列表
        # files: 当前目录下的文件名列表
        
        # 过滤掉不需要复制的目录
        dirs[:] = [d for d in dirs if d != 'data' and d != 'test_out' and d !='checkpoints'and d !='tmp'and d !='watch'and d !='env' and d !='backup']
        # 获得相对路径
        relative_folder_path = os.path.relpath(root,src_dir)
        # 目标目录路径
        dst_folder_path = os.path.join(dst_dir,relative_folder_path)
        
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)
            
        for file in files:
            if not file.endswith('.out'):
                # 复制文件，保留元数据
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_folder_path, file)
                shutil.copy2(src_file_path, dst_file_path)

