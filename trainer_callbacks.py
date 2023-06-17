from transformers.trainer_callback import TrainerCallback


class EvaluationCallback(TrainerCallback):
    def __init__(self, n_eval_each_epoch):
        self.n_eval_each_epoch = n_eval_each_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.n_eval_each_epoch == self.n_eval_each_epoch - 1:
            control.should_evaluate = True