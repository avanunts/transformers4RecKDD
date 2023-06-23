from functools import partial
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_with_piecewise_constant_decay_schedule_lr_lambda(
        current_step, *,
        num_warmup_steps,
        num_cycles,
        num_training_steps,
        piecewise_keys,  # constant segments will be (num_warmup_steps; piecewise_keys[0] + num_warmup_steps], (...[0], ...[1] + ...], ..., (...[n], +inf]
        piecewise_values,  # constants will be piecewise_values[0], ...[1], [n + 1], so the length must be 1 more than keys
):
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cos_multiplier = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    for i, key in enumerate(piecewise_keys):
        if current_step <= key + num_warmup_steps:
            return piecewise_values[i] * cos_multiplier
    return piecewise_values[-1] * cos_multiplier


def get_cosine_with_piecewise_constant_decay_schedule(
        optimizer: Optimizer, num_warmup_steps, num_training_steps, piecewise, num_cycles: float = 0.5, last_epoch=-1
):
    lr_lambda = partial(
        _get_cosine_with_piecewise_constant_decay_schedule_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_cycles=num_cycles,
        num_training_steps=num_training_steps,
        piecewise_keys=piecewise['keys'],
        piecewise_values=piecewise['values']
    )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    'cosine_with_piecewise_constant_decay': get_cosine_with_piecewise_constant_decay_schedule
}


def get_custom_scheduler(
        name: str,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int,
        piecewise,
):
    if name not in TYPE_TO_SCHEDULER_FUNCTION:
        raise ValueError('Required custom scheduler, but no scheduler for name {} is provided'.format(name))
    scheduler_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    return scheduler_func(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        piecewise=piecewise,
        num_cycles=num_cycles,
    )
