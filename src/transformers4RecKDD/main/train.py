import os
import json

from merlin.io import Dataset
from ..t4rec.training_args import CustomTrainingArguments
from ..t4rec import models, trainers, model_configs
from ..paths import t4rec_nvt_ds_path, t4rec_model_path, create_folder_for_path_if_not_exists

MODEL_CONSTRUCTORS = {
    'xlnet': models.xlnet_model,
}

MODEL_CONFIGS = {
    'xlnet': model_configs.XLNetConfig
}


def train(config_path):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)

    train_path, test_path, model_output_dir_path = get_paths_from_config(config)

    train_ds = Dataset(train_path, engine='parquet')
    test_ds = Dataset(test_path, engine='parquet')

    model_config = build_model_config(config['model_type'], config['model_config'])
    model = init_model(config['model_type'], model_config, train_ds.schema)

    training_args = CustomTrainingArguments(
        output_dir=model_output_dir_path,
        **config['training_arg']
    )

    trainer = trainers.CustomTrainer(
        model=model,
        args=training_args,
        schema=train_ds.schema,
        compute_metrics=True,
    )
    trainer.train_dataset_or_path = train_ds
    trainer.eval_dataset_or_path = test_ds

    if checkpoint_exists(model_output_dir_path):
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def checkpoint_exists(dir_path):
    if not os.path.exists(dir_path):
        return False
    checkpoint_folders = list(filter(lambda x: 'checkpoint' in x, os.listdir(dir_path)))
    return len(checkpoint_folders) > 0


def get_paths_from_config(config):
    t4rec_data_folder_path = config['t4rec_data_folder_path']
    locale = config['locale']
    env = config['env']
    cu_version = config['cu_version']
    workflow_version = config['workflow_version']
    train_path = t4rec_nvt_ds_path(t4rec_data_folder_path, locale, env, 'train', cu_version, workflow_version)
    test_path = t4rec_nvt_ds_path(t4rec_data_folder_path, locale, env, 'test', cu_version, workflow_version)

    t4rec_models_folder_path = config['t4rec_models_folder_path']
    model_output_dir_path = t4rec_model_path(t4rec_models_folder_path, locale, env, config['model_name'])
    create_folder_for_path_if_not_exists(os.path.dirname(model_output_dir_path))
    return train_path, test_path, model_output_dir_path


def build_model_config(model_type, config_dict):
    if model_type not in MODEL_CONFIGS:
        raise ValueError('There is no defined config for model type {}'.format(model_type))
    return MODEL_CONFIGS[model_type](**config_dict)


def init_model(model_type, model_config, schema):
    if model_type not in MODEL_CONSTRUCTORS or model_type not in MODEL_CONFIGS:
        raise ValueError('There is no defined constructor for model type {}'.format(model_type))
    return MODEL_CONSTRUCTORS[model_type](model_config, schema)
