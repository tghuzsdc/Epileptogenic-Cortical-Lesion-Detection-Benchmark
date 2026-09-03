"""The two learning-rate schedules nnU-Net does not ship.

Both follow nnU-Net's ``PolyLRScheduler`` contract: ``step(current_epoch)`` is
called once per epoch from ``on_train_epoch_start`` and writes the new value
straight into the optimizer's parameter groups.
"""
import math

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Swin UNETR's and UNETR's schedule: linear warmup, then cosine annealing.

    This reproduces the closed form of the ``LinearWarmupCosineAnnealingLR`` that
    ships with the official pipeline
    (Project-MONAI/research-contributions/{UNETR,SwinUNETR}/BTCV/optimizers/lr_scheduler.py),
    which is the branch that runs when the epoch is passed to ``step()`` -- exactly
    how nnU-Net drives its scheduler::

        epoch <  warmup_epochs:  lr = warmup_start_lr
                                      + epoch * (lr0 - warmup_start_lr) / (warmup_epochs - 1)
        epoch >= warmup_epochs:  lr = eta_min + 0.5 * (lr0 - eta_min)
                                      * (1 + cos(pi * (epoch - warmup) / (max_epochs - warmup)))

    Note the ``warmup_epochs - 1`` denominator: epoch 0 starts at ``warmup_start_lr``
    (0 by default, not lr0/warmup_epochs) and the full learning rate is reached one
    epoch before the warmup ends. The official defaults are ``warmup_epochs=50``,
    ``warmup_start_lr=0`` and ``eta_min=0``.
    """

    def __init__(self, optimizer, initial_lr: float, max_epochs: int,
                 warmup_epochs: int = 50, warmup_start_lr: float = 0.0,
                 eta_min: float = 0.0, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.warmup_epochs = min(warmup_epochs, max(0, max_epochs - 1))
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def _lr_at(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # a 1-epoch warmup has no ramp to speak of; the official formula would
            # divide by zero, so start straight at the full learning rate
            if self.warmup_epochs <= 1:
                return self.initial_lr
            return (self.warmup_start_lr
                    + epoch * (self.initial_lr - self.warmup_start_lr) / (self.warmup_epochs - 1))
        span = max(1, self.max_epochs - self.warmup_epochs)
        progress = min(max((epoch - self.warmup_epochs) / span, 0.0), 1.0)
        return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * progress))

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self._lr_at(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class StepIterationLR(_LRScheduler):
    """V-Net's schedule: divide the learning rate by 10 every ``step_iterations``.

    Milletari et al.: "a initial learning rate of 0.0001 which decreases by one
    order of magnitude every 25K iterations."  We count iterations as
    ``epoch * iterations_per_epoch`` because nnU-Net steps once per epoch.

    Note that a full 1000 x 250 = 250K iteration nnU-Net run means ten decades of
    decay, i.e. the learning rate is numerically dead well before the end.  That
    is what the paper prescribes; ``max_decays`` is provided so the behaviour can
    be capped without touching the trainer.
    """

    def __init__(self, optimizer, initial_lr: float, iterations_per_epoch: int,
                 step_iterations: int = 25000, gamma: float = 0.1,
                 max_decays: int = None, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.iterations_per_epoch = iterations_per_epoch
        self.step_iterations = step_iterations
        self.gamma = gamma
        self.max_decays = max_decays
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        n_decays = (current_step * self.iterations_per_epoch) // self.step_iterations
        if self.max_decays is not None:
            n_decays = min(n_decays, self.max_decays)
        new_lr = self.initial_lr * (self.gamma ** n_decays)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class ConstantLR(_LRScheduler):
    """No schedule at all -- UNETR's official BTCV pipeline keeps the LR fixed."""

    def __init__(self, optimizer, initial_lr: float, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr
