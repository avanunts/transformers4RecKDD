from dataclasses import dataclass, field

from transformers4rec.config.trainer import T4RecTrainingArguments


@dataclass(kw_only=True)
class CustomTrainingArguments(T4RecTrainingArguments):
    custom_scheduler_type: str = None
    lr_piecewise_multiplier: dict = None
    eval_accumulation_steps: int = 10
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 128
    learning_rate: float = 0.001
    lr_scheduler_type: str = 'cosine'
    learning_rate_num_cosine_cycles_by_epoch: float = 1.5
    report_to: list = field(default_factory=list)
    logging_steps: int = 1000
    evaluation_strategy: str = 'steps'
    eval_steps: int = 5000
    save_total_limit: int = 1
    max_sequence_length: int = 20
