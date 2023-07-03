from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def drive_auth():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return GoogleDrive(gauth)


def clean_trash_checkpoints(drive):
    checkpoint_folders = list(filter(lambda x: 'checkpoint' in x['title'], drive.ListFile({'q': "trashed = true"}).GetList()))
    for checkpoint_folder in checkpoint_folders:
        checkpoint_folder_id = checkpoint_folder['id']
        file_list = drive.ListFile({'q': '\'{}\' in parents and trashed=true'.format(checkpoint_folder_id)}).GetList()
        for a_file in file_list:
            print('the file {}, is about to get deleted permanently.'.format(a_file['title']))
            a_file.Delete()
        print('the file {}, is about to get deleted permanently.'.format(checkpoint_folder['title']))
        checkpoint_folder.Delete()


def get_num_checkpoints_in_trash(drive):
    return len(list(filter(lambda x: 'checkpoint' in x['title'], drive.ListFile({'q': "trashed = true"}).GetList())))