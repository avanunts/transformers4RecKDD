from merlin.io import Dataset
import t4rec_models
import t4rec_trainers
from transformers4rec.torch import Trainer



'''
args:
- nvt_paths dict of form {'train': nvt_train_path, 'test': nvt_test_path}
- xlnet_args (see t4rec_models module, here schema is overriden, so you don't have to assign it there)
- custom_training_non_fixed_args (see t4rec_trainers module)
- resume_from_checkpoint (if train was already launched but interrupted for some reason)

returns trainer object, in which model, train, test and eval results might be found
'''


def train_and_eval_xlnet(nvt_paths, xlnet_args, custom_training_non_fixed_args, resume_from_checkpoint):
    train = Dataset(nvt_paths['train'], engine='parquet')
    test = Dataset(nvt_paths['test'], engine='parquet')
    xlnet_args['schema'] = train.schema
    model = t4rec_models.xl_net_model(xlnet_args)
    training_args = t4rec_trainers.CustomTrainingArguments(custom_training_non_fixed_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        schema=train.schema,
        compute_metrics=True,
    )
    trainer.train_dataset_or_path = train
    trainer.eval_dataset_or_path = test
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer



