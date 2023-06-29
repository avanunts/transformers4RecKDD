import os
import json

import pandas as pd

from src.metrics_utils import log_history_helpers

RESULTS_KV = {
    'train_num_epochs': log_history_helpers.get_train_num_epochs,
    'eval_num_epochs_last_cp': log_history_helpers.get_eval_num_epochs,
    'train_loss_last_10_step': lambda x: log_history_helpers.get_train_loss(x, 10),
    'eval_loss_min': log_history_helpers.get_eval_loss_min,
    'eval_loss_last_cp': log_history_helpers.get_eval_loss_last_cp,
    'recall@40_max': lambda x: log_history_helpers.get_recall_max(x, 40),
    'recall@40_last_cp': lambda x: log_history_helpers.get_recall_last_cp(x, 40),
    'mrr@40_max': lambda x: log_history_helpers.get_mrr_max(x, 40),
    'mrr@40_last_cp': lambda x: log_history_helpers.get_mrr_last(x, 40),
}

RESULTS_COLUMNS = ['model_name'] + list(RESULTS_KV.keys())


def save_metrics(trainer, model_name, results_file_path):
    if not os.path.exists(results_file_path):
        init_results_file(results_file_path)
    if model_found_in_results(model_name, results_file_path):
        print('Results for model {} already exists in results table, please, delete entry manually and proceed,'
              ' or leave it as is'.format(model_name))
        return
    metrics = get_metrics(trainer)
    do_save_metrics(metrics, model_name, results_file_path)


def do_save_metrics(metrics, model_name, results_file_path):
    metrics['model_name'] = model_name
    pd.DataFrame([metrics], columns=RESULTS_COLUMNS).to_csv(results_file_path, mode='a', index=False,
                                                            header=False)


def get_metrics(trainer):
    return get_metrics_from_log_history(trainer.state.log_history)


def get_metrics_from_checkpoint(checkpoint_path):
    state = None
    with open(os.path.join(checkpoint_path, 'trainer_state.json'), 'r') as open_file:
        state = json.load(open_file)
    return get_metrics_from_log_history(state['log_history'])


def get_metrics_from_log_history(log_history):
    return dict(map(
        lambda item: (item[0], item[1](log_history)),
        RESULTS_KV.items()
    ))


def init_results_file(results_file_path):
    pd.DataFrame([], columns=RESULTS_COLUMNS).to_csv(results_file_path, index=False, header=True)


def model_found_in_results(model_name, results_file_path):
    return (pd.read_csv(results_file_path).model_name == model_name).sum() > 0
