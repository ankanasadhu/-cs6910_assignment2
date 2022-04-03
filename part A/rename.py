import os

folder_path = 'yolo/images/'

for count, sub_folder in enumerate(os.listdir(folder_path)):
    sub_folder_path = folder_path + '/' + sub_folder
    i = 0
    for count_1, filename in enumerate(os.listdir(sub_folder_path)):
        old_name = f"{sub_folder_path}/{filename}"
        name = sub_folder.lower() + '_' + str(i) + '.png'
        new_name = f"{sub_folder_path}/{name}"

        os.rename(old_name, new_name)
        i += 1