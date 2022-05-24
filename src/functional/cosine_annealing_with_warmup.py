#  Copyright 2022 https://github.com/gaussian37/pytorch_deep_learning_models

import math

# noinspection PyUnresolvedReferences,PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, t_0, t_mult=1, eta_max=0.1, t_up=0, gamma=1.,
                 last_epoch=-1):
        if t_0 <= 0 or not isinstance(t_0, int):
            raise ValueError(
                f"Expected positive integer T_0, but got {t_0}")
        if t_mult < 1 or not isinstance(t_mult, int):
            raise ValueError(
                f"Expected integer T_mult >= 1, but got {t_mult}")
        if t_up < 0 or not isinstance(t_up, int):
            raise ValueError(
                f"Expected positive integer T_up, but got {t_up}")
        self.T_0 = t_0
        self.T_mult = t_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = t_up
        self.T_i = t_0
        self.gamma = gamma
        self.cycle = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer,
                                                            last_epoch)
        self.T_cur = last_epoch

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(
                math.pi * (self.T_cur - self.T_up) / (
                        self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log((epoch / self.T_0 * (self.T_mult - 1) + 1),
                                 self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (
                            self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
