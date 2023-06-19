import json

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

MODEL_CONSTRUCTORS = {
    'xlnet': t4rec_models.xl_net_model,
}

TRAINING_ARGS_CONSTRUCTORS = {
    'custom_v1': t4rec_trainers.CustomTrainingArguments
}


def train_and_eval_xlnet(nvt_paths, model_type, model_args, ta_type, ta_args, resume_from_checkpoint):
    train = Dataset(nvt_paths['train'], engine='parquet')
    test = Dataset(nvt_paths['test'], engine='parquet')
    model = init_model(model_type, model_args, train.schema)
    training_args = init_training_args(ta_type, ta_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        schema=train.schema,
        compute_metrics=True,
    )
    trainer.train_dataset_or_path = train
    trainer.eval_dataset_or_path = test
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    params_json = build_params_json(nvt_paths, model_type, model_args, ta_type, ta_args)
    return trainer, params_json


def init_model(model_type, model_args, schema):
    if model_type not in MODEL_CONSTRUCTORS:
        raise ValueError('There is no defined constructor for model type {}'.format(model_type))
    return MODEL_CONSTRUCTORS[model_type](model_args, schema)


def init_training_args(args_type, args_args):
    if args_type not in TRAINING_ARGS_CONSTRUCTORS:
        raise ValueError('There is no defined constructor for training args type {}'.format(args_type))
    return TRAINING_ARGS_CONSTRUCTORS[args_type](args_args)


def build_params_json(nvt_paths, model_type, model_args, ta_cls_name, ta_cls_init_args):
    params = {
        'paths': nvt_paths,
        'model': {
            'type': model_type,
            'args': model_args,
        },
        'training_arguments': {
            'class_name': ta_cls_name,
            'init_args': ta_cls_init_args
        },
    }
    params_json = json.dumps(params, indent=4)
    return params_json
