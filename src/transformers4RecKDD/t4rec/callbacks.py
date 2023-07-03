from transformers.trainer_callback import TrainerCallback

from ..google_colab_utils.utils import clean_trash_checkpoints, get_num_checkpoints_in_trash


class CleanDriveTrashCheckpointsCallback(TrainerCallback):
    def __init__(self, drive):
        self.drive = drive

    def on_save(self, args, state, control, **kwargs):
        max_num_checkpoints_in_trash = args.max_num_checkpoints_in_trash
        num_checkpoints_in_trash = get_num_checkpoints_in_trash(self.drive)
        if num_checkpoints_in_trash > max_num_checkpoints_in_trash:
            clean_trash_checkpoints(self.drive)
