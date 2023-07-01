import os
import json
import time

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


def train_many(*config_paths):
    for config_path in config_paths:
        if not check_config_path_to_model_name_bijection(config_path):
            print('Skipping this train, because check is not successful')
        t1 = time.time()
        try:
            train_one(config_path)
        except Exception as err:
            print('Exception for config at {} has occurred. Continue without saving time. '
                  'Exception: {}'.format(config_path, err))
            continue
        t2 = time.time()
        print('Total training time for config at {} is {:.2f}s'.format(config_path, t2 - t1))
        save_additional_info(config_path, t1, t2)


# do not use this method directly; if you want to train one model use train_many with a length one list arg
def train_one(config_path):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)

    train_path, test_path, model_output_dir_path = get_paths_from_config(config)

    train_ds = Dataset(train_path, engine='parquet')
    test_ds = Dataset(test_path, engine='parquet')

    model_config = build_model_config(config['model_type'], config['model_config'])
    model = init_model(config['model_type'], model_config, train_ds.schema)

    training_args = CustomTrainingArguments(
        output_dir=model_output_dir_path,
        **config['training_args']
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
    t4rec_folder_path = config['t4rec_folder_path']
    locale = config['locale']
    env = config['env']
    cu_version = config['cu_version']
    workflow_version = config['workflow_version']
    train_path = t4rec_nvt_ds_path(t4rec_folder_path, locale, env, 'train', cu_version, workflow_version)
    test_path = t4rec_nvt_ds_path(t4rec_folder_path, locale, env, 'test', cu_version, workflow_version)

    model_output_dir_path = t4rec_model_path(t4rec_folder_path, locale, env, config['model_name'])
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


def save_additional_info(config_path, t1, t2):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    _, _, model_output_dir_path = get_paths_from_config(config)
    add_info_path = os.path.join(model_output_dir_path, 'add_info.json')
    if os.path.exists(add_info_path):
        return
    add_info = {
        'config_path': config_path,
        'first_iter_num_epochs': config['training_args']['num_train_epochs'],
        'time_start': t1,
        'time_end': t2,
        'total_time_seconds': t2 - t1,
        'time_per_epoch': (t2 - t1) / config['training_args']['num_train_epochs']
    }

    json_object = json.dumps(add_info, indent=4)
    with open(add_info_path, "w") as outfile:
        outfile.write(json_object)


def check_config_path_to_model_name_bijection(config_path):
    print('Start check for config_path <-> model_name bijection')
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    _, _, model_output_dir_path = get_paths_from_config(config)
    if not os.path.exists(model_output_dir_path):
        print('There is no model at path {}. Check succeeded.'.format(model_output_dir_path))
        return True
    add_info_path = os.path.join(model_output_dir_path, 'add_info.json')
    if not os.path.exists(add_info_path):
        print('Model output dir at path {} exists, '
              'but add_info at path {} doesn\'t, check failed.'.format(model_output_dir_path, add_info_path))
        return False
    with open(add_info_path, 'r') as open_file:
        add_info = json.load(open_file)
    if add_info['config_path'] != config_path:
        print('This model name already reserved for the config at path {}, '
              'trying to use it to train model with config'
              ' at path {}, check failed.'.format(add_info['config_path'], config_path))
        return False
    print('Check succeeded.')
    return True


