"""
学习率渐进预热调度器
实现论文'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'中的预热策略
"""
from typing import List
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    """
    学习率渐进预热调度器
    在指定轮次内线性增加学习率，之后使用后续调度器
    
    参数:
        optimizer: 优化器对象
        warmup_epochs: 预热轮次
        after_scheduler: 预热完成后使用的调度器(如ReduceLROnPlateau)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        after_scheduler: _LRScheduler
    ):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """计算当前轮次的学习率"""
        if self.last_epoch > self.warmup_epochs:
            if not self.warmup_finished:
                self.warmup_finished = True
            return self.after_scheduler.get_last_lr()

        return [base_lr * (float(self.last_epoch) / self.warmup_epochs) for base_lr in self.base_lrs]

    def step(self):
        """执行一次调度步骤"""
        if self.warmup_finished:
            self.after_scheduler.step()
            self.last_epoch = self.after_scheduler.last_epoch + self.warmup_epochs + 1
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step()
  

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100 - 10)
    scheduler = GradualWarmupScheduler(optim, warmup_epochs=10, after_scheduler=cosine_scheduler)
    a = []
    b = []

    for epoch in range(1, 100):
        scheduler.step()
        a.append(epoch)
        b.append(optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[0]['lr'])

    plt.plot(a,b)
    # plt.savefig("test.png", dpi=160)
    plt.show()