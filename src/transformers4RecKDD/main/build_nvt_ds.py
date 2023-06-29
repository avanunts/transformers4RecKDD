import gc
import json

import pandas as pd
import cudf
import nvtabular as nvt

from ..paths import *
from ..data_utils import kdd_datasets_processing
from ..data_utils.nvt_workflows import NVT_WORKFLOWS

from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

'''
config is a dict
{
    locale: str (DE, UK, ...)
    cu_version: int (1, 2, ...) - version of a cu_ds to use
    workflow_version: int (1, 2, ...) - version of workflow from data_utils.nvt_workflows to use
    paths_config_path: str - path to json with paths_config
}
'''


def build_nvt_ds(config_path):
    with open(config_path, 'r') as open_file:
        config = json.load(open_file)
    locale = config['locale']
    env = config['env']
    cu_version = config['cu_version']
    workflow_version = config['workflow_version']
    t4rec_data_folder_path = config['t4rec_data_folder_path']
    kdd_data_folder_path = config['kdd_data_folder_path']

    nvt_train_path = t4rec_nvt_ds_path(t4rec_data_folder_path, locale, env, 'train', cu_version, workflow_version)
    nvt_test_path = t4rec_nvt_ds_path(t4rec_data_folder_path, locale, env, 'test', cu_version, workflow_version)

    if os.path.exists(nvt_train_path):
        print('NVT dataset already exists')
        return

    cu_train_path = t4rec_cu_ds_path(t4rec_data_folder_path, locale, env, 'train', cu_version)
    cu_test_path = t4rec_cu_ds_path(t4rec_data_folder_path, locale, env, 'test', cu_version)

    if not os.path.exists(cu_train_path):
        print('Start building pd datasets')

        kdd_train_sessions_path = kdd_sessions_path(kdd_data_folder_path, locale, env, 'train')
        kdd_train_sessions = pd.read_parquet(kdd_train_sessions_path, engine='pyarrow')
        kdd_test_sessions_path = kdd_sessions_path(kdd_data_folder_path, locale, env, 'holdout')
        kdd_test_sessions = pd.read_parquet(kdd_test_sessions_path, engine='pyarrow')
        product_attributes = pd.read_parquet(kdd_products_path(kdd_data_folder_path, locale))

        pd_train, pd_test = kdd_datasets_processing.process_kdd_datasets(
            kdd_train_sessions,
            kdd_test_sessions,
            product_attributes
        )

        print('Pd train ds shape: {}'.format(pd_train.shape))
        pd_train.info(verbose=True)
        print('Pd test ds shape: {}'.format(pd_test.shape))
        pd_test.info(verbose=True)

        cu_train = cudf.from_pandas(pd_train)
        cu_train.to_parquet(cu_train_path)
        cu_test = cudf.from_pandas(pd_test)
        cu_test.to_parquet(cu_test_path)

        print('Saved cu datasets successfully to paths {}, {}'.format(cu_train_path, cu_test_path))
    else:
        print('Cu datasets already exist, reading from paths {}, {}'.format(cu_train_path, cu_test_path))
        cu_train = cudf.read_parquet(cu_train_path)
        cu_test = cudf.read_parquet(cu_test_path)

    nv_train = nvt.Dataset(cu_train)
    nv_test = nvt.Dataset(cu_test)
    workflow = NVT_WORKFLOWS[workflow_version]

    workflow.fit_transform(nv_train).to_parquet(nvt_train_path)
    workflow_path = nvt_workflow_path(t4rec_data_folder_path, locale, env, workflow_version)
    workflow.save(workflow_path)
    workflow.transform(nv_test).to_parquet(nvt_test_path)
    print('Saved nvt datasets and workflow successfully to paths {}, {}, {}'
          .format(nvt_train_path, nvt_test_path, workflow_path))
