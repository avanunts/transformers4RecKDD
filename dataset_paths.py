import os

DATA_FOLDER = 'drive/MyDrive/data'
KDD_DATA_FOLDER = os.path.join(DATA_FOLDER, 'datasets')
T4REC_DATA_FOLDER = os.path.join(DATA_FOLDER, 'transformers4rec')
T4REC_MODELS_FOLDER = 'drive/MyDrive/models/transformers4rec'


def kdd_sessions_path(locale, typ):
    return os.path.join(KDD_DATA_FOLDER, '{}_sessions_{}.parquet'.format(locale, typ))


def kdd_products_path(locale):
    return os.path.join(KDD_DATA_FOLDER, '{}_products.parquet'.format(locale))


def t4rec_cu_ds_path(locale, typ):
    return os.path.join(T4REC_DATA_FOLDER, '{}_{}_processed.parquet'.format(locale, typ))


def t4rec_nvt_ds_path(locale, typ):
    return os.path.join(T4REC_DATA_FOLDER, '{}_{}_processed_nvt'.format(locale, typ))


def t4rec_model_path(locale, typ, version):
    return os.path.join(T4REC_MODELS_FOLDER, '{}_{}_{}'.format(locale, typ, version))


def nvt_workflow_path(locale, version):
    return os.path.join(T4REC_DATA_FOLDER, "{}_{}_workflow_etl".format(locale, version))
