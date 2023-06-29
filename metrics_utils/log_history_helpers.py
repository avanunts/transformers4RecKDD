import numpy as np


def get_train_num_epochs(log_history):
    return log_history[-1]['epoch']


def get_eval_num_epochs(log_history):
    return list(get_eval_entries(log_history))[-1]['epoch']


def get_mrr_max(log_history, at_k):
    eval_entries = get_eval_entries(log_history)
    return max(map(lambda x: mrr_at_k(x, at_k), eval_entries))


def get_mrr_last(log_history, at_k):
    eval_entries = get_eval_entries(log_history)
    return list(map(lambda x: mrr_at_k(x, at_k), eval_entries))[-1]


def get_recall_max(log_history, at_k):
    eval_entries = get_eval_entries(log_history)
    return max(map(lambda x: recall_at_k(x, at_k), eval_entries))


def get_recall_last_cp(log_history, at_k):
    eval_entries = get_eval_entries(log_history)
    return list(map(lambda x: recall_at_k(x, at_k), eval_entries))[-1]


def get_train_loss(log_history, last_k):
    return np.mean(list(map(lambda x: x['loss'], get_train_entries(log_history)[-last_k:])))


def get_eval_loss_min(log_history):
    eval_entries = get_eval_entries(log_history)
    return min(map(loss, eval_entries))


def get_eval_loss_last_cp(log_history):
    eval_entries = get_eval_entries(log_history)
    return list(map(loss, eval_entries))[-1]


def get_train_entries(log_history):
    return list(filter(is_train_log_entry, log_history))


def get_eval_entries(log_history):
    return list(filter(is_eval_log_entry, log_history))


def is_eval_log_entry(entry):
    return 'eval_runtime' in entry


def is_train_log_entry(entry):
    return 'loss' in entry


def mrr_at_k(entry, k):
    return entry['eval_/next-item/mean_reciprocal_rank_at_{}'.format(k)]


def recall_at_k(entry, k):
    return entry['eval_/next-item/recall_at_{}'.format(k)]


def loss(entry):
    return entry['eval_/loss']
