import os
import json
import time
import traceback

from merlin.io import Dataset
import torch
from ..t4rec.training_args import CustomTrainingArguments
from ..t4rec import models, trainers, model_configs, callbacks
from ..paths import t4rec_nvt_ds_path, t4rec_model_path, create_folder_for_path_if_not_exists
from ..metrics_utils.log_history_helpers import get_train_entries, get_train_num_epochs

MODEL_CONSTRUCTORS = {
    'xlnet': models.xlnet_model,
}

MODEL_CONFIGS = {
    'xlnet': model_configs.XLNetConfig
}


def train_many(*config_paths, drive=None):
    for config_path in config_paths:
        try:
            if not check_config_path_to_model_name_bijection(config_path):
                print('Skipping this train, because check is not successful')
                continue
            num_train_epochs_at_start = get_num_train_epochs_at_last_checkpoint(config_path)
            t1 = time.time()
            train_one(config_path, drive)
            t2 = time.time()
            torch.cuda.empty_cache()
            num_train_epochs_at_end = get_num_train_epochs_from_config(config_path)
            print('Total training time for config at {} is {:.2f}s'.format(config_path, t2 - t1))
            save_additional_info(config_path, t1, t2, num_train_epochs_at_start, num_train_epochs_at_end)
        except Exception:
            print('Exception for config at {} has occurred. Continue without saving time'.format(config_path))
            print(traceback.format_exc())
            continue


# do not use this method directly; if you want to train one model use train_many with a length one list arg
def train_one(config_path, drive=None):
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

    _callbacks = None
    if training_args.max_num_checkpoints_in_trash is not None:
        if drive is None:
            raise ValueError('max_num_checkpoints_in_trash in training args is {} (not None), but drive argument'
                             'is None'.format(training_args.max_num_checkpoints_in_trash))
        _callbacks = [callbacks.CleanDriveTrashCheckpointsCallback(drive)]

    trainer = trainers.CustomTrainer(
        model=model,
        args=training_args,
        schema=train_ds.schema,
        compute_metrics=True,
        callbacks=_callbacks,
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


def get_checkpoint_num_steps(checkpoint_name):
    return int(checkpoint_name.split('-')[1])


def get_last_checkpoint_name(model_dir_path):
    return sorted(list(filter(lambda x: 'checkpoint' in x, os.listdir(model_dir_path))), key=get_checkpoint_num_steps)[-1]


def get_paths_from_config(config):
    kdd_folder_path = config['kdd_folder_path']
    locale = config['locale']
    env = config['env']
    cu_version = config['cu_version']
    workflow_version = config['workflow_version']
    train_path = t4rec_nvt_ds_path(kdd_folder_path, locale, env, 'train', cu_version, workflow_version)
    test_path = t4rec_nvt_ds_path(kdd_folder_path, locale, env, 'test', cu_version, workflow_version)

    model_output_dir_path = t4rec_model_path(kdd_folder_path, locale, env, config['model_name'])
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


def save_additional_info(config_path, t1, t2, num_train_epochs_at_start, num_train_epochs_at_end):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    _, _, model_output_dir_path = get_paths_from_config(config)
    add_info_path = os.path.join(model_output_dir_path, 'add_info.json')
    if os.path.exists(add_info_path):
        return
    num_epochs = num_train_epochs_at_end - num_train_epochs_at_start
    add_info = {
        'config_path': config_path,
        'num_epochs': num_epochs,
        'time_start': t1,
        'time_end': t2,
        'total_time_seconds': t2 - t1,
        'time_per_epoch': (t2 - t1) / num_epochs
    }

    json_object = json.dumps(add_info, indent=4)
    with open(add_info_path, "w") as outfile:
        outfile.write(json_object)


def check_config_path_to_model_name_bijection(config_path):
    print('Start check for config_path <-> model_name bijection')
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    _, _, model_output_dir_path = get_paths_from_config(config)
    add_info_path = os.path.join(model_output_dir_path, 'add_info.json')
    if not os.path.exists(add_info_path):
        print('add_info.json at path {} doesn\'t. Will recover from the last '
              'checkpoint (if exists) and proceed or start training from scratch '
              '(if no checkpoint was found)'.format(add_info_path))
        return True
    with open(add_info_path, 'r') as open_file:
        add_info = json.load(open_file)
    if add_info['config_path'] != config_path:
        print('This model name already reserved for the config at path {}, '
              'trying to use it to train model with config'
              ' at path {}, check failed.'.format(add_info['config_path'], config_path))
        return False
    print('Check succeeded.')
    return True


def get_num_train_epochs_at_last_checkpoint(config_path):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    _, _, model_output_dir_path = get_paths_from_config(config)
    if not checkpoint_exists(model_output_dir_path):
        return 0
    last_checkpoint_path = os.path.join(model_output_dir_path, get_last_checkpoint_name(model_output_dir_path))
    last_state_path = os.path.join(last_checkpoint_path, 'trainer_state.json')
    with open(last_state_path, 'r') as open_file:
        last_state = json.load(open_file)
    train_log_entries = get_train_entries(last_state['log_history'])
    return get_train_num_epochs(train_log_entries)


def get_num_train_epochs_from_config(config_path):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    return config['training_args']['num_train_epochs']
