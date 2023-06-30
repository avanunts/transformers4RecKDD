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


def kdd_sessions_path(kdd_data_folder_path, locale, env, typ):
    return os.path.join(kdd_data_folder_path, locale, env, '{}_sessions.parquet'.format(typ))


def kdd_products_path(kdd_data_folder_path, locale):
    return os.path.join(kdd_data_folder_path, locale, 'products.parquet')


def t4rec_cu_ds_path(t4rec_data_folder_path, locale, env, typ, cu_version: int):
    return os.path.join(
        t4rec_data_folder_path, 'cu_datasets', locale, env,  '{}_v{}.parquet'.format(typ, cu_version)
    )


def t4rec_nvt_ds_path(t4rec_data_folder_path, locale, env, typ, cu_version: int, workflow_version: int):
    return os.path.join(
        t4rec_data_folder_path, 'nvt_datasets', locale, env,
        '{}_cu_v{}_workflow_v{}'.format(typ, cu_version, workflow_version)
    )


def t4rec_model_path(t4rec_models_folder_path, locale, env, model_name):
    return os.path.join(
        t4rec_models_folder_path, locale, env, model_name
    )


def nvt_workflow_path(t4rec_data_folder_path, locale, env, workflow_version: int):
    return os.path.join(
        t4rec_data_folder_path, 'nvt_workflows', locale, env,
        'workflow_v{}'.format(workflow_version)
    )


def create_folder_for_path_if_not_exists(path):
    dir_path = os.path.dirname(path)
    if os.path.exists(path):
        return
    os.makedirs(dir_path, exist_ok=True)

