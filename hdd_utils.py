import os


def clear_checkpoints(dir_path, names_to_leave, dry_run=True):
    all_dirs_and_files = os.listdir(dir_path)
    rm_dirs_and_files = sorted(list(filter(lambda x: x not in names_to_leave, all_dirs_and_files)))
    if dry_run:
        print('DRY RUN: files and dirs will not be removed, to remove them set dry_run=False')
        for dir_or_file_name in rm_dirs_and_files:
            print(dir_or_file_name)
    else:
        for dir_or_file_name in rm_dirs_and_files:
            path = os.path.join(dir_path, dir_or_file_name)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                os.rmdir(path)

