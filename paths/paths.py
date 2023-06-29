import os
import json
from importlib_resources import files

folders = files('paths').joinpath('folders').read_text(encoding="utf-8")
entry_points = json.load(folders)
KDD_DATA_FOLDER = entry_points['kdd_data']
T4REC_DATA_FOLDER = entry_points['t4rec_data']
T4REC_MODELS_FOLDER = entry_points['t4rec_models']


def kdd_sessions_path(locale, typ):
    return os.path.join(KDD_DATA_FOLDER, '{}_sessions_{}.parquet'.format(locale, typ))


def kdd_products_path(locale):
    return os.path.join(KDD_DATA_FOLDER, '{}_products.parquet'.format(locale))


def t4rec_cu_ds_path(locale, typ, version: int):
    return os.path.join(T4REC_DATA_FOLDER, '{}_{}_v{}_processed.parquet'.format(locale, typ, version))


def t4rec_nvt_ds_path(locale, typ, version: int):
    return os.path.join(T4REC_DATA_FOLDER, '{}_{}_v{}_processed_nvt'.format(locale, typ, version))


def t4rec_model_path(locale, typ, version):
    return os.path.join(T4REC_MODELS_FOLDER, '{}_{}_v{}'.format(locale, typ, version))


def nvt_workflow_path(locale, version: int):
    return os.path.join(T4REC_DATA_FOLDER, "{}_v{}_workflow_etl".format(locale, version))
