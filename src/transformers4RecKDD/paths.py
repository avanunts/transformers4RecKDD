import os
import json

'''
typ: dev_train, dev_test, submit_train, submit_predict
locale: DE, JP, UP, IT, ES, FR
paths_config_path: path to config json, in which kdd_data_folder, t4rec_data_folder, t4rec_models_folder paths are specified
version: int 
'''


def kdd_sessions_path(locale, typ, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['kdd_data_folder'], '{}_sessions_{}.parquet'.format(locale, typ))


def kdd_products_path(locale, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['kdd_data_folder'], '{}_products.parquet'.format(locale))


def t4rec_cu_ds_path(locale, typ, version: int, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['t4rec_data_folder'], '{}_{}_v{}_processed.parquet'.format(locale, typ, version))


def t4rec_nvt_ds_path(locale, typ, version: int, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['t4rec_data_folder'], '{}_{}_v{}_processed_nvt'.format(locale, typ, version))


def t4rec_model_path(locale, typ, version, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['t4rec_models_folder'], '{}_{}_v{}'.format(locale, typ, version))


def nvt_workflow_path(locale, version: int, paths_config_path):
    config = json.loads(paths_config_path)
    return os.path.join(config['t4rec_data_folder'], "{}_v{}_workflow_etl".format(locale, version))
