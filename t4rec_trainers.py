import os

from transformers4rec.config.trainer import T4RecTrainingArguments

# params of training arguments that we fix

ENGINE = 'merlin'
DROP_LAST = False
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 128
GRADIENT_ACUM_STEPS = 1
EVAL_ACUM_STEPS = 10
NO_CUDA = False
EVAL_STRATEGY = 'steps'


'''
We divide args of T4RecTrainingArguments into 3 groups: 
1. fixed (upper case variables)
2. customizable with defaults (listed below)
3. customizable without defaults: 
- output_dir_prefix
- model_name
- num_epochs
'''

defaults = {
    'lr': 0.001,
    'scheduler': 'cosine',
    'num_cycles': 1.5,
    'max_seq_length': 20,
    'logging_steps': 1000,
    'eval_steps': 5000
}


class CustomTrainingArguments(T4RecTrainingArguments):
    def get_argument(self, arg_name):
        if arg_name in self.non_fixed_args:
            return self.non_fixed_args[arg_name]
        if arg_name not in defaults:
            raise ValueError("there is no arg with {}".format(arg_name))
        return defaults[arg_name]

    def __init__(self, non_fixed_args):
        self.non_fixed_args = non_fixed_args
        output_dir = os.path.join(self.get_argument('output_dir_prefix'), self.get_argument('model_name'))
        super().__init__(
            data_loader_engine=ENGINE,
            dataloader_drop_last=DROP_LAST,
            gradient_accumulation_steps=GRADIENT_ACUM_STEPS,
            eval_accumulation_steps=EVAL_ACUM_STEPS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            output_dir=output_dir,
            learning_rate=self.get_argument('lr'),
            lr_scheduler_type=self.get_argument('scheduler'),
            learning_rate_num_cosine_cycles_by_epoch=self.get_argument('num_cycles'),
            num_train_epochs=self.get_argument('num_epochs'),
            max_sequence_length=self.get_argument('max_seq_length'),
            report_to=[],
            logging_steps=self.get_argument('logging_steps'),
            no_cuda=NO_CUDA,
            evaluation_strategy=EVAL_STRATEGY,
            eval_steps=self.get_argument('eval_steps'),
        )
