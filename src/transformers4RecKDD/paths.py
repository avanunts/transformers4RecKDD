import os
import json

'''
locale: DE, JP, UP, IT, ES, FR
env: dev/submit
typ: train/test
cu_version: 1, 2, 3, ... - version of cu_ds for nvt_dataset
workflow_version: 1, 2, 3, ... - version of nvt workflow
 
paths_config_path: path to config json, in which kdd_data_folder, t4rec_data_folder, t4rec_models_folder paths are specified
'''


def kdd_sessions_path(locale, env, typ, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['kdd_data_folder'], locale, env, '{}_sessions.parquet'.format(typ))


def kdd_products_path(locale, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['kdd_data_folder'], locale, 'products.parquet')


def t4rec_cu_ds_path(locale, env, typ, cu_version: int, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(
        config['t4rec_data_folder'], 'cu_datasets', locale, env,  '{}_v{}.parquet'.format(typ, cu_version)
    )


def t4rec_nvt_ds_path(locale, env, typ, cu_version: int, workflow_version: int, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(
        config['t4rec_data_folder'], 'nvt_datasets', locale, env,
        '{}_cu_v={}_workflow_v={}'.format(typ, cu_version, workflow_version)
    )


def nvt_workflow_path(locale, env, workflow_version: int, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(
        config['t4rec_data_folder'], 'nvt_workflows', locale, env,
        'workflow_v{}'.format(workflow_version)
    )
