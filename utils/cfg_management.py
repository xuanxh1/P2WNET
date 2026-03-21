"""
配置管理工具
用于合并命令行参数和配置文件
"""

def merge_args_into_config(args, config):
    """
    递归地将命令行参数合并到配置字典中
    args中的同名参数将覆盖config中的值，支持嵌套字典结构
    
    参数:
        args: 命令行参数对象或字典
        config: 配置字典
    
    返回:
        config: 合并后的配置字典
    """
    def merge_recursive(args_dict, config_dict):
        """递归合并函数"""
        # 遍历 args 中的所有键值对
        for key, value in args_dict.items():
            # 直接在当前层级找到匹配的键
            if key in config_dict:
                config_dict[key] = value
            else:
                # 递归搜索所有嵌套的字典
                for config_value in config_dict.values():
                    if isinstance(config_value, dict):
                        merge_recursive(args_dict, config_value)

    args_dict = vars(args) if hasattr(args, '__dict__') else args
    
    # 先处理顶层的键值对
    for key, value in args_dict.items():
        if key in config:
            config[key] = value
            
    # 然后递归处理嵌套的字典
    merge_recursive(args_dict, config)
    return config

