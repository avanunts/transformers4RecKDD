from merlin.io import Dataset
from .t4rec_models import xl_net_model
from .t4rec_trainers import CustomTrainingArguments
from transformers4rec.torch import Trainer



'''
args:
- nvt_paths dict of form {'train': nvt_train_path, 'test': nvt_test_path}
- xlnet_args (see t4rec_models module)
- custom_training_non_fixed_args (see t4rec_trainers module)

returns trainer object, in which model, train, test and eval results might be found
'''


def train_and_eval_xlnet(nvt_paths, xlnet_args, custom_training_non_fixed_args):
    train = Dataset(nvt_paths['train'], engine='parquet')
    test = Dataset(nvt_paths['test'], engine='parquet')
    model = xl_net_model(xlnet_args)
    training_args = CustomTrainingArguments(custom_training_non_fixed_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        schema=train.schema,
        compute_metrics=True,
    )
    trainer.train_dataset_or_path = train
    trainer.eval_dataset_or_path = test
    trainer.train(resume_from_checkpoint=False)
    return trainer



