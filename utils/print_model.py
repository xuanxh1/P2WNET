"""
模型信息打印工具
使用PyTorch Lightning打印模型结构和参数统计
"""
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary import ModelSummary

class ModelWrapper(LightningModule):
    """
    将普通模型包装为PyTorch Lightning模块
    用于生成模型摘要
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

def print_one_model_summaries(model, logger):
    """
    打印单个模型的结构摘要和参数统计
    
    参数:
        model: 待打印的模型
        logger: 日志记录器
    """
    # Wrap models with Lightning Module
    pl_first_stage = ModelWrapper(model)
    pl_second_stage = ModelWrapper(model)
    
    # Generate model summaries
    logger.info("\n==== FIRST STAGE MODEL SUMMARY ====")
    summary1 = ModelSummary(pl_first_stage, max_depth=-1)
    logger.info(f"{summary1}")
    
    # Calculate and print total parameters
    total_params_first = sum(p.numel() for p in model.parameters())
    
    logger.info(f"\nTotal parameters in first stage model: {total_params_first:,}")

   