import os
import shutil


def clear_checkpoints(folder_path, names_to_leave, dry_run=True):
    all_dir_and_file_names = os.listdir(folder_path)
    is_dir_name_to_remove = lambda x: x not in names_to_leave and os.path.isdir(os.path.join(folder_path, x))
    rm_dirs = sorted(list(filter(is_dir_name_to_remove, all_dir_and_file_names)))
    if dry_run:
        print('DRY RUN: files and dirs will not be removed, to remove them set dry_run=False')
        for dir_name in rm_dirs:
            print(dir_name)
    else:
        for dir_name in rm_dirs:
            dir_path = os.path.join(folder_path, dir_name)
            shutil.rmtree(dir_path, ignore_errors=True)
