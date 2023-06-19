import log_history_helpers


def get_metrics(trainer):
    log_history = trainer.state.log_history
    return {
        'train_num_epochs': log_history_helpers.get_train_num_epochs(log_history),
        'eval_num_epochs_last_cp': log_history_helpers.get_eval_num_epochs(log_history),
        'train_loss_last_10_step': log_history_helpers.get_train_loss(log_history, 10),
        'eval_loss_min': log_history_helpers.get_eval_loss_min(log_history),
        'eval_loss_last_cp': log_history_helpers.get_eval_loss_last_cp(log_history),
        'recall@40_max': log_history_helpers.get_recall_max(log_history, 40),
        'recall@40_last_cp': log_history_helpers.get_recall_last_cp(log_history, 40),
        'mrr@40_max': log_history_helpers.get_mrr_max(log_history, 40),
        'mrr@40_last_cp': log_history_helpers.get_mrr_last(log_history, 40),
    }
