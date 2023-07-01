import os
import json

'''
locale: DE, JP, UP, IT, ES, FR
env: dev/submit
typ: train/test
cu_version: 1, 2, 3, ... - version of cu_ds for nvt_dataset
workflow_version: 1, 2, 3, ... - version of nvt workflow
 
paths_config_path: path to config json, in which t4rec_folder_path is specified
'''


def kdd_sessions_path(t4rec_folder_path, locale, env, typ):
    return os.path.join(t4rec_folder_path, 'data', 'kdd_datasets', locale, env, '{}_sessions.parquet'.format(typ))


def kdd_products_path(t4rec_folder_path, locale):
    return os.path.join(t4rec_folder_path, 'data', 'kdd_datasets', locale, 'products.parquet')


def t4rec_cu_ds_path(t4rec_folder_path, locale, env, typ, cu_version: int):
    return os.path.join(
        t4rec_folder_path, 'data', 'cu_datasets', locale, env,  '{}_v{}.parquet'.format(typ, cu_version)
    )


def t4rec_nvt_ds_path(t4rec_folder_path, locale, env, typ, cu_version: int, workflow_version: int):
    return os.path.join(
        t4rec_folder_path, 'data', 'nvt_datasets', locale, env,
        '{}_cu_v{}_workflow_v{}'.format(typ, cu_version, workflow_version)
    )


def t4rec_model_path(t4rec_folder_path, locale, env, model_name):
    return os.path.join(
        t4rec_folder_path, 'models', 'transformers4rec', locale, env, model_name
    )


def nvt_workflow_path(t4rec_folder_path, locale, env, workflow_version: int):
    return os.path.join(
        t4rec_folder_path, 'data', 'transformers4rec', 'nvt_workflows', locale, env,
        'workflow_v{}'.format(workflow_version)
    )


def create_folder_for_path_if_not_exists(path):
    dir_path = os.path.dirname(path)
    if os.path.exists(path):
        return
    os.makedirs(dir_path, exist_ok=True)

