from __future__ import absolute_import

import math
from torch.optim.lr_scheduler import _LRScheduler


class ERF(_LRScheduler):
    def __init__(self, optimizer, min_lr, alpha, beta, epochs, last_epoch=-1):
        self.min_lr = min_lr
        self.alpha = alpha
        self.beta = beta
        self.epochs = epochs
        super(ERF, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.min_lr + (base_lr - self.min_lr) / 2.0 * (
                1 - math.erf((self.beta - self.alpha) * self.last_epoch / self.epochs + self.alpha))
                for base_lr in self.base_lrs]
