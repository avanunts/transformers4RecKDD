import log_history_helpers


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


def get_metrics(trainer):
    log_history = trainer.state.log_history
    return dict(map(
        lambda item: (item[0], item[1](log_history)),
        RESULTS_KV.items()
    ))
