import matplotlib.pyplot as plt


def plot_eval_results(trainer, metric_name, **kwargs):
    eval_log_history = list(filter(is_eval_log_entry, trainer.state.log_history))
    epochs = list(map(lambda x: x['epoch'], eval_log_history))
    plt.figure(figsize=(16, 9))

    if metric_name == 'mrr':
        for at_k in kwargs['at_k']:
            mrr = list(map(lambda x: mrr_at_k(x, at_k), eval_log_history))
            plt.plot(epochs, mrr, label='mrr@{}'.format(at_k))

    if metric_name == 'recall':
        for at_k in kwargs['at_k']:
            recall = list(map(lambda x: recall_at_k(x, at_k), eval_log_history))
            plt.plot(epochs, recall, label='recall@{}'.format(at_k))

    if metric_name == 'loss':
        loss_y = list(map(lambda x: loss(x), eval_log_history))
        plt.plot(epochs, loss_y, label='loss')

    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def is_eval_log_entry(entry):
    return 'eval_runtime' in entry


def mrr_at_k(entry, k):
    return entry['eval_/next-item/mean_reciprocal_rank_at_{}'.format(k)]


def recall_at_k(entry, k):
    return entry['eval_/next-item/recall_at_{}'.format(k)]


def loss(entry):
    return entry['eval_/loss']
