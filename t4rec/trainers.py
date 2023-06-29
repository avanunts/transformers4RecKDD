import inspect

import torch
from transformers4rec.torch import Trainer
from typing import Optional, Union
from torch.optim import Optimizer
from transformers.trainer_utils import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

import schedulers


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.args.custom_scheduler_type is None:
            super().create_scheduler(num_training_steps, optimizer)
        else:
            self.create_custom_scheduler(num_training_steps, optimizer)

    def create_custom_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = t4rec_schedulers.get_custom_scheduler(
                self.args.custom_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.args.learning_rate_num_cosine_cycles_by_epoch * self.args.num_train_epochs,
                piecewise=self.args.lr_piecewise_multiplier
            )

    # override, because of the bug (TODO: add link to bug-report with fix, when prepare)
    @staticmethod
    def get_scheduler(
            name: Union[str, SchedulerType],
            optimizer: Optimizer,
            num_warmup_steps: Optional[int] = None,
            num_training_steps: Optional[int] = None,
            num_cycles: Optional[int] = 0.5,
    ):
        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
        if name == SchedulerType.CONSTANT:
            return schedule_func(optimizer)

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if "num_cycles" in inspect.signature(schedule_func).parameters:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )

        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
