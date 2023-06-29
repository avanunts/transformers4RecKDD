import matplotlib.pyplot as plt
from src.transformers4RecKDD.metrics_utils import log_history_helpers


def plot_eval_results(trainer, metric_name, **kwargs):
    log_history = trainer.state.log_history
    plot_eval_results_from_log_history(log_history, metric_name, **kwargs)


def plot_eval_results_from_log_history(log_history, metric_name, **kwargs):
    eval_log_history = log_history_helpers.get_eval_entries(log_history)
    epochs = list(map(lambda x: x['epoch'], eval_log_history))
    plt.figure(figsize=(16, 9))

    if metric_name == 'mrr':
        for at_k in kwargs['at_k']:
            mrr = list(map(lambda x: log_history_helpers.mrr_at_k(x, at_k), eval_log_history))
            plt.plot(epochs, mrr, label='mrr@{}'.format(at_k))

    if metric_name == 'recall':
        for at_k in kwargs['at_k']:
            recall = list(map(lambda x: log_history_helpers.recall_at_k(x, at_k), eval_log_history))
            plt.plot(epochs, recall, label='recall@{}'.format(at_k))

    if metric_name == 'loss':
        loss_y = list(map(lambda x: log_history_helpers.loss(x), eval_log_history))
        plt.plot(epochs, loss_y, label='loss')

    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
